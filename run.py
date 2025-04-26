# Create a new file, e.g., clinical_analyzer.py

import logging
import re
from typing import Dict, List, Optional, Union, Any
import spacy
from spacy import displacy
# PhraseMatcher might not be needed for the core extraction class
# from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import torch
import medspacy
# Ensure necessary spaCy models are downloaded if needed (e.g., en_core_web_sm or md, en_ner_bc5cdr_md)
# import spacy.cli
# try:
#     spacy.load("en_ner_bc5cdr_md")
# except OSError:
#     logging.info("Downloading en_ner_bc5cdr_md model...")
#     spacy.cli.download("en_ner_bc5cdr_md")
# try:
#      spacy.load("en_core_web_sm") # Often needed for sentencizer/parser/tagger if not in the clinical model
# except OSError:
#      logging.info("Downloading en_core_web_sm model...")
#      spacy.cli.download("en_core_web_sm")

import os
import scispacy
from medspacy.visualization import visualize_dep
import pandas as pd

# Configure logging (can be configured by the main application instead)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClinicalTextAnalyzer:
    """
    A class to load clinical NLP models and perform analysis tasks like
    negation detection, affirmed finding extraction, and token analysis.
    """

    def __init__(self, model_name="en_ner_bc5cdr_md"):
        """
        Initializes the analyzer by loading the specified spaCy/scispaCy model
        and adding necessary components like MedspaCy context.

        Args:
            model_name (str): The name of the spaCy/scispaCy model to load.
                              Defaults to "en_ner_bc5cdr_md".
        """
        self.model_name = model_name
        self.nlp = None # Initialize to None
        # Load pipeline during init
        self._load_pipeline(self.model_name)

        if self.nlp is None:
            # Handle initialization failure more robustly
            raise RuntimeError(f"Failed to load NLP pipeline for model '{model_name}'")

        # Define keywords for categorization heuristics
        # These are used if no entities are detected or entity context isn't clear
        self.negation_keywords = ["no", "not", "without", "negative for"]
        self.normal_keywords = [
            "normal", "intact", "clear", "moist", "warm", "dry", "soft",
            "appropriate", "well", "equal", "symmetric", "unremarkable",
            "nontender", "non-tender" # Add common PE descriptors
        ]


    def _check_gpu(self):
        """Checks and logs GPU availability."""
        gpu_id = -1
        try:
            if torch.cuda.is_available() and spacy.prefer_gpu():
                try:
                    current_device_id = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_device_id)
                    logging.info(f"ClinicalTextAnalyzer using GPU (ID: {current_device_id}) - {gpu_name}")
                    return True, current_device_id
                except Exception as e:
                    # This can happen if spacy.prefer_gpu() succeeds but torch can't see the device
                    logging.warning(f"spacy.prefer_gpu() succeeded, but couldn't confirm device via torch: {e}")
                    return True, 0 # Assume device 0 if prefer_gpu succeeded
            else:
                logging.info("ClinicalTextAnalyzer using CPU.")
                spacy.require_cpu()
                return False, -1
        except Exception as e:
            logging.error(f"Error during GPU check: {e}")
            logging.info("Falling back to CPU.")
            spacy.require_cpu()
            return False, -1

    def _load_pipeline(self, model_name):
        """Loads the spaCy pipeline and adds MedspaCy components."""
        logging.info(f"Attempting to load spaCy model: {model_name}...")
        self._check_gpu() # Check GPU status during loading
        try:
            # First, try loading the specified model
            nlp = spacy.load(model_name)
            logging.info(f"Base model '{model_name}' loaded with components: {nlp.pipe_names}")

            # Add sentencizer if not present (crucial for medspaCy context)
            if 'sentencizer' not in nlp.pipe_names:
                if 'parser' in nlp.pipe_names:
                    # Add before parser if parser exists
                    nlp.add_pipe("sentencizer", before="parser")
                else:
                    # Otherwise add first
                    nlp.add_pipe("sentencizer", first=True)
                logging.info("Added 'sentencizer' pipe.")
            else:
                 logging.info("'sentencizer' already in pipeline.")


            # Add medspaCy context component if not present
            if "medspacy_context" not in nlp.pipe_names:
                logging.info("Adding 'medspacy_context' pipe...")
                # Typically added after NER if NER is present
                if 'ner' in nlp.pipe_names:
                    nlp.add_pipe("medspacy_context", after="ner")
                    logging.info("Added 'medspacy_context' pipe after NER.")
                else:
                    # Add last if NER is not present
                    nlp.add_pipe("medspacy_context", last=True)
                    logging.warning("NER component not found. Added 'medspacy_context' last.")

                if "medspacy_context" not in nlp.pipe_names:
                    logging.error("'medspacy_context' pipe could not be added!")
                    self.nlp = None # Ensure self.nlp is None on failure
                    return
                else:
                    logging.info("Verified 'medspacy_context' is in the pipeline.")
            else:
                 logging.info("'medspacy_context' already in pipeline.")

            logging.info(f"Final pipeline for '{model_name}': {nlp.pipe_names}")
            self.nlp = nlp # Assign loaded pipeline to instance variable

        except OSError as e:
            logging.error(f"Model '{model_name}' loading failed: {e}.")
            logging.info(f"Attempting to load 'en_core_web_sm' as a fallback...")
            # Try loading a smaller model if the clinical one fails
            try:
                 nlp = spacy.load("en_core_web_sm")
                 logging.info("Fallback model 'en_core_web_sm' loaded.")

                 # Add sentencizer if not present
                 if 'sentencizer' not in nlp.pipe_names:
                      nlp.add_pipe("sentencizer", first=True)
                      logging.info("Added 'sentencizer' pipe to fallback model.")

                 # Add medspaCy context component (without NER, will use default rules)
                 if "medspacy_context" not in nlp.pipe_names:
                      nlp.add_pipe("medspacy_context", last=True)
                      logging.info("Added 'medspacy_context' pipe to fallback model.")

                 logging.info(f"Final pipeline for 'en_core_web_sm' fallback: {nlp.pipe_names}")
                 self.nlp = nlp # Assign fallback pipeline
                 self.model_name = "en_core_web_sm (fallback)"
                 logging.warning(f"Using fallback model '{self.model_name}' as '{model_name}' could not be loaded.")

            except OSError as fallback_e:
                 logging.error(f"Fallback model 'en_core_web_sm' also failed to load: {fallback_e}")
                 logging.error("NLP pipeline setup failed.")
                 self.nlp = None # Ensure self.nlp is None on complete failure
            except Exception as fallback_e:
                 logging.error(f"An unexpected error occurred during fallback pipeline setup: {fallback_e}")
                 self.nlp = None
        except Exception as e:
            logging.error(f"An unexpected error occurred during pipeline setup: {e}")
            self.nlp = None


    def analyze(self, text: str) -> spacy.tokens.Doc | None:
        """
        Processes the input text using the loaded spaCy pipeline.

        Args:
            text (str): The text to analyze.

        Returns:
            spacy.tokens.Doc | None: The processed Doc object, or None if
                                     the pipeline isn't loaded or text is empty/whitespace.
        """
        if not self.nlp:
            logging.error("NLP pipeline is not loaded. Cannot analyze text.")
            return None
        if not text or not text.strip():
             logging.debug("Skipping analysis for empty or whitespace text.")
             return None

        logging.debug(f"Analyzing text: '{text[:50]}...' with model '{self.model_name}'...")
        try:
             doc = self.nlp(text)
             logging.debug("Text analysis complete.")
             return doc
        except Exception as e:
             logging.error(f"Error during NLP analysis of text: {e}", exc_info=True)
             return None


    def get_affirmed_findings_by_system(self, doc: spacy.tokens.Doc, text: str) -> dict:
        """
        Identifies system sections in the original text using regex headers and
        extracts affirmed entities from the Doc object within each section.
        Note: This method is designed for header-based reports like ROS, not
              the nested structure of the physical exam dictionary.

        Args:
            doc (spacy.tokens.Doc): The processed Doc object from analyze().
            text (str): The original text string (needed for header detection).

        Returns:
            dict: A dictionary where keys are system names (str) and values are
                  lists of tuples `(entity_text, entity_label)` for affirmed entities.
                  Returns an empty dict if doc is None or no headers are found.
        """
        if doc is None:
            logging.warning("Doc object is None. Cannot extract affirmed findings by system.")
            return {}

        logging.info("Extracting affirmed findings by system using regex headers...")
        findings = {}
        header_pattern = re.compile(r"^\s*([A-Z][a-zA-Z/\-]+(?: [A-Z][a-zA-Z/\-]+)*):\s*", re.MULTILINE)
        headers = []

        # Find all headers and their positions
        for match in header_pattern.finditer(text):
             system_name = match.group(1).strip()
             content_start_char = match.end()
             headers.append({"name": system_name, "start": match.start(), "content_start": content_start_char})

        if not headers:
            logging.warning("No system headers found in the text.")
            return {} # Return empty if no headers

        # Sort headers by start position to define sections
        headers.sort(key=lambda x: x["start"])
        text_end = len(text)

        # Filter affirmed entities from the doc
        affirmed_entities = []
        for ent in doc.ents:
             # Check if medspaCy context attribute exists and entity is not negated
             if hasattr(ent._, 'is_negated') and not ent._.is_negated:
                  affirmed_entities.append({
                      "text": ent.text,
                      "label": ent.label_,
                      "start_char": ent.start_char,
                      "end_char": ent.end_char
                  })
             elif not hasattr(ent._, 'is_negated'):
                  # If is_negated is not available (e.g., medspacy context not added),
                  # assume affirmed for simplicity in this specific function's context.
                  # In a real application, you might handle this differently.
                  logging.warning(f"Entity '{ent.text}' has no 'is_negated' attribute. Assuming affirmed.")
                  affirmed_entities.append({
                       "text": ent.text,
                       "label": ent.label_,
                       "start_char": ent.start_char,
                       "end_char": ent.end_char
                  })


        # Map affirmed entities to their respective sections
        num_headers = len(headers)
        for i, header in enumerate(headers):
             system_name = header["name"]
             section_content_start = header["content_start"]
             # The section ends either at the start of the next header or the end of the text
             section_content_end = headers[i+1]["start"] if i + 1 < num_headers else text_end

             # Initialize list for this system if it doesn't exist
             if system_name not in findings:
                  findings[system_name] = []

             # Add entities falling within this section's character range
             for ent in affirmed_entities:
                  # Check if the entity's span is fully contained within the section's content area
                  if ent["start_char"] >= section_content_start and ent["end_char"] <= section_content_end:
                       findings[system_name].append((ent["text"], ent["label"]))

        # Remove systems with no findings
        findings = {k: v for k, v in findings.items() if v}

        logging.info("Finished mapping affirmed entities to systems.")
        return findings


    def get_token_analysis_df(self, doc: spacy.tokens.Doc) -> pd.DataFrame:
        """
        Analyzes each token in the Doc and returns attributes as a Pandas DataFrame.

        Args:
            doc (spacy.tokens.Doc): The processed Doc object from analyze().

        Returns:
            pd.DataFrame: DataFrame with token attributes. Returns empty DataFrame
                          if doc is None.
        """
        if doc is None:
            logging.warning("Doc object is None. Cannot create token analysis DataFrame.")
            return pd.DataFrame()

        logging.info("Analyzing token attributes for DataFrame...")
        token_data = []
        # Create a mapping from token index to its entity span for quick lookup
        token_to_entity_map = {}
        for span in doc.ents:
             for i in range(span.start, span.end):
                  token_to_entity_map[i] = span

        for token in doc:
            entity_span = token_to_entity_map.get(token.i)
            entity_context = "N/A"
            entity_text_in_span = ""

            if entity_span:
                 entity_text_in_span = entity_span.text
                 # Check if medspaCy context attribute exists
                 if hasattr(entity_span._, 'is_negated'):
                      entity_context = "NEGATED" if entity_span._.is_negated else "AFFIRMED"
                 else:
                      entity_context = "CONTEXT_N/A" # Indicate context not available

            token_data.append({
                "TEXT": token.text,
                "LEMMA": token.lemma_,
                "POS": token.pos_,
                "TAG": token.tag_,
                "DEP": token.dep_,
                "HEAD_TEXT": token.head.text, # Use HEAD_TEXT for clarity
                "ENTITY_IOB": token.ent_iob_,
                "ENTITY_TYPE": token.ent_type_ or "O",
                "ENTITY_TEXT_SPAN": entity_text_in_span, # Add the full entity text
                "ENTITY_CONTEXT": entity_context,
                "IS_SPACE": token.is_space,
                "TOKEN_IDX": token.i
            })

        df = pd.DataFrame(token_data)
        logging.info("Finished creating token analysis DataFrame.")
        return df

    def _get_entity_options(self):
        """Internal helper to get displaCy entity options."""
        # Define colors for common entities or types you expect
        # Ensure 'ents' list includes all relevant entity types from your model
        entities = list(self.nlp.get_pipe('ner').labels) if self.nlp and 'ner' in self.nlp.pipe_names else ["ENTITY"]
        # Add default spacy entities if they might be present
        default_spacy_ents = ["PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
        entities.extend([ent for ent in default_spacy_ents if ent not in entities])

        colors = {
            "DISEASE": "linear-gradient(90deg, #ff9999, #ff6666)", # Example color
            "CHEMICAL": "linear-gradient(90deg, #ccff99, #99cc66)", # Example color
            "SIGN": "linear-gradient(90deg, #ffff99, #cccc66)", # Example color for clinical signs
            "SYMPTOM": "linear-gradient(90deg, #99ccff, #6699cc)", # Example color for symptoms
             # Add colors for other important entity types if needed
        }
        # You might want to add default colors for entity types not explicitly listed
        # using a generic palette or hash function. For now, let's stick to explicit ones.

        return {"ents": entities, "colors": colors}

    def save_entity_visualization(self, doc: spacy.tokens.Doc, output_path: str):
        """Saves the spaCy entity visualization to an HTML file."""
        if doc is None:
            logging.warning("Doc object is None. Cannot save entity visualization.")
            return

        logging.info(f"Saving entity visualization to {output_path}...")
        try:
            options = self._get_entity_options()
            html = displacy.render(doc, style='ent', options=options, page=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logging.info("Entity visualization saved.")
        except Exception as e:
            logging.error(f"Failed to save entity visualization to {output_path}: {e}")


    def save_context_visualization(self, doc: spacy.tokens.Doc, output_path: str):
        """Saves the MedspaCy context visualization to an HTML file."""
        if doc is None:
            logging.warning("Doc object is None. Cannot save context visualization.")
            return

        logging.info(f"Saving context visualization to {output_path}...")
        try:
            # medspaCy's visualize_dep requires specific dependencies and might not
            # work universally depending on spacy version/installation.
            # We added an ImportError check earlier.
            html = visualize_dep(doc) # This function generates the HTML directly
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logging.info("Context visualization saved.")
        except ImportError:
            logging.error("Failed to import medspacy.visualization. visualize_dep not available.")
        except Exception as e:
            logging.error(f"Failed to save context dependency visualization to {output_path}: {e}")

    def _add_to_nested_dict(self, nested_dict: Dict[str, Any], path: List[str], value: str):
        """
        Helper to add a value (string) to a nested dictionary structure,
        creating dictionaries or lists as needed along the path.
        If the final path segment already exists, ensures it's a list and appends.
        """
        current = nested_dict
        for i, segment in enumerate(path):
            if i == len(path) - 1:
                # This is the last segment, representing the key where the value should be stored.
                # We want to store values as a list at this final key.
                if segment not in current or not isinstance(current[segment], list):
                    current[segment] = []
                current[segment].append(value)
            else:
                # Intermediate segment, needs to be a dictionary.
                if segment not in current or not isinstance(current[segment], dict):
                    current[segment] = {}
                current = current[segment]


    def analyze_physical_exam(self, physical_exam_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Traverses a nested dictionary representing a physical exam, analyzes
        each string observation using NLP, and categorizes it as 'positive',
        'negative', or 'normal'.

        Uses medspaCy context detection and heuristic keyword matching for categorization.

        Args:
            physical_exam_data (Dict[str, Any]): The nested dictionary
                structure of the physical exam. Leaves should be strings or lists of strings.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary with keys 'positive',
                'negative', 'normal', each containing a nested dictionary
                mirroring the input structure but only with the observations
                categorized into that group.
        """
        logging.info("Analyzing physical exam data structure...")

        positive_findings = {}
        negative_findings = {}
        normal_findings = {}
        unknown_findings = {} # Optional: keep track of unclassified

        # Use sets for faster keyword lookup after tokenization (optional, regex is fine too)
        # negation_kw_set = set(self.negation_keywords)
        # normal_kw_set = set(self.normal_keywords)

        def _categorize_observation(text: str) -> str:
            """Applies categorization logic to a single observation string."""
            if not isinstance(text, str) or not text.strip():
                 logging.debug("Skipping empty or non-string observation.")
                 return "unknown" # Cannot categorize non-strings or empty strings

            doc = self.analyze(text)

            # If NLP fails, fall back solely to keyword matching
            if doc is None:
                 logging.warning(f"NLP failed for text: '{text[:50]}...'. Using keyword fallback.")
                 has_negation_keyword = any(re.search(r"\b" + keyword + r"\b", text.lower()) for keyword in self.negation_keywords)
                 has_normal_keyword = any(re.search(r"\b" + keyword + r"\b", text.lower()) for keyword in self.normal_keywords)

                 if has_negation_keyword:
                      return "negative"
                 elif has_normal_keyword:
                      return "normal"
                 else:
                      logging.debug(f"Text '{text[:50]}...' could not be categorized by keywords either.")
                      return "unknown"


            # --- Primary Categorization Logic (with NLP) ---

            # 1. Check for Affirmed Entities (Highest Priority for "positive")
            affirmed_entities_found = False
            for ent in doc.ents:
                 # Ensure the medspaCy context attribute exists
                 if hasattr(ent._, 'is_negated'):
                      if not ent._.is_negated:
                           affirmed_entities_found = True
                           break # Found one affirmed entity, classify as positive

            if affirmed_entities_found:
                 return "positive"

            # 2. Check for Negated Entities or Explicit Negation Keywords
            negated_entities_found = False
            for ent in doc.ents:
                 if hasattr(ent._, 'is_negated'):
                      if ent._.is_negated:
                           negated_entities_found = True
                           break # Found one negated entity

            has_negation_keyword = any(re.search(r"\b" + keyword + r"\b", text.lower()) for keyword in self.negation_keywords)

            if negated_entities_found or has_negation_keyword:
                 return "negative"

            # 3. Check for Normal Keywords (if not positive or negative based on steps 1 & 2)
            has_normal_keyword = any(re.search(r"\b" + keyword + r"\b", text.lower()) for keyword in self.normal_keywords)

            if has_normal_keyword:
                 return "normal"

            # 4. Default or Unknown
            # If none of the above apply, it could be a description without specific findings
            # (e.g., "Abdomen is soft."). Such descriptions are often implicitly normal
            # in a "negative exam" context. Defaulting to "normal" seems reasonable
            # if no entities, negation, or explicit normal keywords signal otherwise.
            logging.debug(f"Text '{text[:50]}...' not classified by rules 1-3. Defaulting to 'normal'.")
            return "normal" # Defaulting to normal if nothing specific is flagged


        def _traverse_and_categorize(data: Any, current_path: List[str]):
            """Recursively traverses the data structure and categorizes string leaves."""
            if isinstance(data, str):
                # Found a string observation, categorize it
                category = _categorize_observation(data)
                logging.debug(f"Categorized '{data[:50]}...' as '{category}' at path {' -> '.join(current_path)}")

                if category == "positive":
                    self._add_to_nested_dict(positive_findings, current_path, data)
                elif category == "negative":
                    self._add_to_nested_dict(negative_findings, current_path, data)
                elif category == "normal":
                    self._add_to_nested_dict(normal_findings, current_path, data)
                else:
                    self._add_to_nested_dict(unknown_findings, current_path, data) # Add to unknown if needed

            elif isinstance(data, dict):
                # Traverse dictionary items
                for key, value in data.items():
                    _traverse_and_categorize(value, current_path + [key])

            elif isinstance(data, list):
                # Traverse list items (assuming list contains strings or structures)
                # Note: The path doesn't change for items within the same list container
                for item in data:
                    _traverse_and_categorize(item, current_path)

            # Ignore other data types (e.g., numbers, booleans) if they appear

        # Start the recursive traversal
        _traverse_and_categorize(physical_exam_data, [])

        # Return the categorized findings dictionaries
        return {
            "positive": positive_findings,
            "negative": negative_findings,
            "normal": normal_findings
            # Optionally include "unknown" for debugging: "unknown": unknown_findings
        }


# --- Example Usage (can be in the same file for testing or in your main app) ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    # Use INFO for general messages, DEBUG for detailed analysis steps
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("ClinicalTextAnalyzer script starting in example usage mode...")

    try:
        # Initialize the analyzer
        # You might need to download models if they are not present.
        # Uncomment the spacy.cli.download lines at the top if needed.
        # Use a model like "en_ner_bc5cdr_md" or "en_core_web_sm" + custom NER
        # "en_ner_bc5cdr_md" recognizes diseases and chemicals, useful but might miss signs/symptoms.
        # For better PE analysis, a model trained on clinical signs/symptoms is ideal.
        # If using "en_core_web_sm" fallback, categorization relies more on keywords.
        analyzer = ClinicalTextAnalyzer(model_name="en_ner_bc5cdr_md")
        # analyzer = ClinicalTextAnalyzer(model_name="en_core_web_sm") # Test with fallback

        # Example ROS text for the original function
        ros = """
            Review of Systems: Comprehensive
            HENT: Negative for facial swelling and nosebleeds.
            Eyes: Negative for photophobia and visual disturbance.
            Respiratory: Negative for shortness of breath. No cough.
            Cardiovascular: Negative for chest pain or palpitations. Normal heart sounds.
            Gastrointestinal: Negative for nausea and vomiting. Abdomen non-tender.
            Genitourinary: Negative for flank pain and hematuria. Clear urine.
            Musculoskeletal: Positive for arthralgias (left arm pain). No back pain and neck pain. Limited range of motion in left shoulder.
                Left lateral trapezius pain with radiation down left arm
            Skin: Negative for wound or rash. Skin warm and dry.
            Neurological: Positive for numbness in fingers. Negative for dizziness, syncope, weakness and headaches. Cranial nerves intact.
            Psychiatric/Behavioral: Negative for confusion. Mood normal.
        """

        print("\n" + "="*40)
        print("--- Analyzing Sample ROS ---")
        print("="*40)

        ros_doc = analyzer.analyze(ros)

        if ros_doc:
            # 1. Extract affirmed findings by system (using the original ROS function)
            print("\n--- Extracting Affirmed Findings by System (ROS) ---")
            affirmed_findings_ros = analyzer.get_affirmed_findings_by_system(ros_doc, ros)
            print("\nAffirmed Findings Summary (ROS):")
            print("-" * 30)
            if affirmed_findings_ros:
                for system, findings_list in affirmed_findings_ros.items():
                    print(f"System: {system}")
                    if findings_list:
                        for entity_text, entity_label in findings_list:
                            # Note: en_ner_bc5cdr_md detects Diseases and Chemicals.
                            # 'arthralgias', 'numbness' etc. might not be found unless labeled as DISEASE.
                            # The output here depends *heavily* on the NER model used.
                            print(f"  - {entity_text} (Label: {entity_label})")
                    print("-" * 10)
            else:
                print("  (No affirmed findings extracted by get_affirmed_findings_by_system)")
            print("-" * 30)

            # 2. Get Token Analysis DataFrame (for ROS)
            print("\n--- Generating Token Analysis DataFrame (ROS) ---")
            token_df_ros = analyzer.get_token_analysis_df(ros_doc)
            print("\nToken Analysis DataFrame Head (ROS):")
            print(token_df_ros.head())
            # print("\nFull Token Analysis DataFrame (ROS):")
            # print(token_df_ros[['TEXT', 'ENTITY_TYPE', 'ENTITY_CONTEXT', 'HEAD_TEXT', 'DEP']].to_string())

            # 3. Save visualizations (for ROS)
            output_dir = "./nlp_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            analyzer.save_entity_visualization(ros_doc, os.path.join(output_dir, "ros_entities.html"))
            analyzer.save_context_visualization(ros_doc, os.path.join(output_dir, "ros_context.html"))
            print(f"\nROS visualizations saved to {output_dir}")


        # Example Physical Exam data for the new function
        # physical_exam_data =  {
        #   "Constitutional": {
        #     "Appearance": "She is not diaphoretic."
        #   },
        #   "HENT": {
        #     "Head": "Normocephalic and atraumatic.",
        #     "Mouth/Throat": {
        #       "Mouth": "Mucous membranes are moist."
        #     }
        #   },
        #   "Eyes": {
        #     "General": "No scleral icterus.",
        #     "Extraocular Movements": "Extraocular movements intact.",
        #     "Conjunctiva/sclera": "Conjunctivae normal."
        #   },
        #   "Neck": {
        #     "Vascular": "No JVD.",
        #     "Comments": "No midline cervical tenderness" # This should be negative via keyword
        #   },
        #   "Cardiovascular": {
        #     "Rate and Rhythm": "Normal rate and regular rhythm.",
        #     "Heart sounds": "No murmur heard." # This should be negative via keyword or entity negation
        #   },
        #   "Pulmonary": {
        #     "Effort": "No tachypnea or accessory muscle usage.", # Negative via keyword
        #     "Breath sounds": "No wheezing or rales." # Negative via keyword or entity negation
        #   },
        #   "Abdominal": {
        #     "General": "There is no distension.", # Negative via keyword
        #     "Palpations": "Abdomen is soft.", # Normal via keyword
        #     "Tenderness": "There is no abdominal tenderness." # Negative via keyword
        #   },
        #   "Musculoskeletal": {
        #     "General": [
        #       "Tenderness (Left posterior shoulder) present.", # Positive via entity or keyword
        #       "No swelling or deformity." # Negative via keyword or entity negation
        #     ],
        #     "Cervical back": [
        #       "Normal range of motion.", # Normal via keyword
        #       "No edema or erythema." # Negative via keyword or entity negation
        #     ],
        #     "Comments": [
        #       "Left posterior deltoid tenderness, left lateral trapezius tenderness, left paraspinal cervical tenderness.", # Contains positive findings (keywords or potential entities)
        #       "No midline cervical tenderness." # Negative via keyword
        #     ]
        #   },
        #   "Skin": {
        #     "General": "Skin is warm and dry.", # Normal via keyword
        #     "Capillary Refill": "Capillary refill takes less than 2 seconds.", # Could be normal (no keywords/entities though) - would be 'normal' by default
        #     "Coloration": "Skin is not pale.", # Negative via keyword
        #     "Findings": "No rash." # Negative via keyword or entity negation
        #   },
        #   "Neurological": {
        #     "General": "No focal deficit present.", # Negative via keyword or entity negation
        #     "Mental Status": [
        #       "She is alert.", # Normal (no findings/negation/normal kw - default) or could add "alert" to normal_keywords
        #       "Mental status is at baseline." # Normal (no findings/negation/normal kw - default)
        #     ],
        #     "Cranial Nerves": "Cranial nerves 2-12 are intact.", # Normal via keyword
        #     "Sensory": "No sensory deficit.", # Negative via keyword or entity negation
        #     "Motor": "No weakness.", # Negative via keyword or entity negation
        #     "Deep Tendon Reflexes": {
        #       "Reflex Scores": [
        #         "Patellar reflexes are 2+ on the right side and 2+ on the left side.", # Normal/Descriptive (no findings/negation/normal kw - default)
        #         "Achilles reflexes are 2+ on the right side and 2+ on the left side." # Normal/Descriptive (no findings/negation/normal kw - default)
        #       ]
        #     }
        #   },
        #   "Psychiatric": {
        #     "Mood and Affect": "Mood normal.", # Normal via keyword
        #     "Behavior": "Behavior normal." # Normal via keyword
        #   }
        # }
        physical_exam_data = {
            "Constitutional": {
                "Appearance": "She is not diaphoretic."
            },
            "HENT": {
                "Head": "Normocephalic and atraumatic.",
                "Mouth/Throat": {
                "Mouth": "Mucous membranes are moist."
                }
            },
            "Eyes": {
                "General": "No scleral icterus.",
                "Extraocular Movements": "Extraocular movements intact.",
                "Conjunctiva/sclera": "Conjunctivae normal."
            },
            "Neck": {
                "Vascular": "No JVD.",
                "Comments": "No midline cervical tenderness"
            },
            "Cardiovascular": {
                "Rate and Rhythm": "Normal rate and regular rhythm.",
                "Heart sounds": "No murmur heard."
            },
            "Pulmonary": {
                "Effort": "No tachypnea or accessory muscle usage.",
                "Breath sounds": "No wheezing or rales."
            },
            "Abdominal": {
                "General": "There is no distension.",
                "Palpations": "Abdomen is soft.",
                "Tenderness": "There is no abdominal tenderness."
            },
            "Musculoskeletal": {
                "General": [
                "Tenderness (Left posterior shoulder) present.",
                "No swelling or deformity."
                ],
                "Cervical back": [
                "Normal range of motion.",
                "No edema or erythema."
                ],
                "Comments": [
                "Left posterior deltoid tenderness, left lateral trapezius tenderness, left paraspinal cervical tenderness.",
                "No midline cervical tenderness."
                ]
            },
            "Skin": {
                "General": "Skin is warm and dry.",
                "Capillary Refill": "Capillary refill takes less than 2 seconds.",
                "Coloration": "Skin is not pale.",
                "Findings": "No rash."
            },
            "Neurological": {
                "General": "No focal deficit present.",
                "Mental Status": [
                "She is alert.",
                "Mental status is at baseline."
                ],
                "Cranial Nerves": "Cranial nerves 2-12 are intact.",
                "Sensory": "No sensory deficit.",
                "Motor": "No weakness.",
                "Deep Tendon Reflexes": {
                "Reflex Scores": [
                    "Patellar reflexes are 2+ on the right side and 2+ on the left side.",
                    "Achilles reflexes are 2+ on the right side and 2+ on the left side."
                 ]
                }
            },
            "Psychiatric": {
                "Mood and Affect": "Mood normal.",
                "Behavior": "Behavior normal."
            }
        }
        
        print("\n" + "="*40)
        print("--- Analyzing Sample Physical Exam ---")
        print("="*40)

        # 4. Analyze the physical exam dictionary
        pe_analysis_results = analyzer.analyze_physical_exam(physical_exam_data)

        print("\n--- Physical Exam Analysis Summary ---")
        print("\nPositive Findings:")
        print("-" * 30)
        import json # Use json for pretty printing dictionaries
        print(json.dumps(pe_analysis_results.get("positive", {}), indent=2))
        print("-" * 30)

        print("\nNegative Findings:")
        print("-" * 30)
        print(json.dumps(pe_analysis_results.get("negative", {}), indent=2))
        print("-" * 30)

        print("\nNormal Findings:")
        print("-" * 30)
        print(json.dumps(pe_analysis_results.get("normal", {}), indent=2))
        print("-" * 30)

        # Note on expected output: The accuracy of "positive" findings depends heavily
        # on whether the NER model ("en_ner_bc5cdr_md" in this case) identifies the
        # specific clinical findings (like "tenderness", "swelling", "deformity",
        # "wheezing", "rales", "murmur", "diaphoretic", "icterus", "JVD", etc.) as
        # entities. "en_ner_bc5cdr_md" primarily finds Diseases and Chemicals.
        # Therefore, the categorization will rely more on the keyword heuristics
        # and negation detection around non-entity phrases unless a more appropriate
        # clinical NER model is used. The provided code uses a combination of medspacy
        # context (if entities are found) and keyword heuristics (if no entities or
        # context detection isn't sufficient).


    except RuntimeError as e:
        logging.error(f"Initialization failed: {e}. Please ensure models are downloaded.")
        logging.info("Attempt to download necessary models:")
        # Example of how you might download models if they are missing
        # import subprocess
        # try:
        #     subprocess.run(["python", "-m", "spacy", "download", "en_ner_bc5cdr_md"], check=True)
        #     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        # except Exception as download_e:
        #     logging.error(f"Model download failed: {download_e}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during script execution: {e}", exc_info=True)

    logging.info("ClinicalTextAnalyzer script finished.")