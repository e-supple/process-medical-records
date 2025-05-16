import logging
import re
from typing import Dict, List, Any
import spacy
import scispacy
from medspacy.context import ConTextComponent
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClinicalTextAnalyzerV2:
    """
    A class to analyze clinical text using the en_ner_clinical_large model,
    focusing on physical exam data categorization and detailed observation analysis.
    """

    def __init__(self, model_name="en_ner_clinical_large"):
        """
        Initializes the analyzer with the specified spaCy/scispaCy model.

        Args:
            model_name (str): The name of the spaCy/scispaCy model. Defaults to "en_ner_clinical_large".
        """
        self.model_name = model_name
        self.nlp = None
        self._load_pipeline(model_name)

        if self.nlp is None:
            raise RuntimeError(f"Failed to load NLP pipeline for model '{model_name}'")

        # Define keywords for categorization heuristics
        self.negation_keywords = ["no", "not", "without", "negative for"]
        self.normal_keywords = [
            "normal", "intact", "clear", "moist", "warm", "dry", "soft",
            "appropriate", "well", "equal", "symmetric", "unremarkable",
            "nontender", "non-tender",
            "2+"  # Added for reflex scores
        ]

    def _load_pipeline(self, model_name):
        """Loads the spaCy pipeline with MedspaCy context component."""
        logging.info(f"Loading spaCy model: {model_name}...")
        try:
            self.nlp = spacy.load(model_name)
            logging.info(f"Base model '{model_name}' loaded with components: {self.nlp.pipe_names}")

            # Add sentencizer if not present
            if 'sentencizer' not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer", first=True)
                logging.info("Added 'sentencizer' pipe.")

            # Add MedspaCy context component
            if "medspacy_context" not in self.nlp.pipe_names:
                context = ConTextComponent(self.nlp, add_attrs=True)
                self.nlp.add_pipe("medspacy_context", last=True)
                logging.info("Added 'medspacy_context' pipe.")

            logging.info(f"Final pipeline: {self.nlp.pipe_names}")
        except Exception as e:
            logging.error(f"Failed to load model '{model_name}': {e}")
            self.nlp = None

    def analyze(self, text: str) -> spacy.tokens.Doc | None:
        """
        Processes text using the loaded spaCy pipeline.

        Args:
            text (str): The text to analyze.

        Returns:
            spacy.tokens.Doc | None: The processed Doc object or None if invalid.
        """
        if not self.nlp:
            logging.error("NLP pipeline not loaded.")
            return None
        if not text or not text.strip():
            logging.debug("Skipping empty text.")
            return None
        try:
            doc = self.nlp(text)
            return doc
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            return None

    def _add_to_nested_dict(self, nested_dict: Dict[str, Any], path: List[str], value: str):
        """Helper to add a value to a nested dictionary, ensuring lists at leaf nodes."""
        current = nested_dict
        for i, segment in enumerate(path):
            if i == len(path) - 1:
                if segment not in current or not isinstance(current[segment], list):
                    current[segment] = []
                current[segment].append(value)
            else:
                if segment not in current or not isinstance(current[segment], dict):
                    current[segment] = {}
                current = current[segment]

    def analyze_physical_exam(self, physical_exam_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes a nested physical exam dictionary, categorizing observations as
        'positive', 'negative', or 'normal'.

        Args:
            physical_exam_data (Dict[str, Any]): Nested dictionary of physical exam data.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with 'positive', 'negative', 'normal' keys,
                                       each containing categorized observations.
        """
        logging.info("Analyzing physical exam data...")
        positive_findings = {}
        negative_findings = {}
        normal_findings = {}

        def _categorize_observation(text: str) -> str:
            """Categorizes a single observation string."""
            if not isinstance(text, str) or not text.strip():
                return "unknown"

            doc = self.analyze(text)
            if doc is None:
                logging.warning(f"NLP failed for text: '{text[:50]}...'. Using keyword fallback.")
                has_negation = any(re.search(r"\b" + kw + r"\b", text.lower()) for kw in self.negation_keywords)
                has_normal = any(re.search(r"\b" + kw + r"\b", text.lower()) for kw in self.normal_keywords)
                if has_negation:
                    return "negative"
                elif has_normal:
                    return "normal"
                return "unknown"

            # Check for affirmed entities
            for ent in doc.ents:
                if hasattr(ent._, 'is_negated') and not ent._.is_negated:
                    return "positive"

            # Check for negated entities or keywords
            for ent in doc.ents:
                if hasattr(ent._, 'is_negated') and ent._.is_negated:
                    return "negative"
            if any(re.search(r"\b" + kw + r"\b", text.lower()) for kw in self.negation_keywords):
                return "negative"

            # Check for normal keywords
            if any(re.search(r"\b" + kw + r"\b", text.lower()) for kw in self.normal_keywords):
                return "normal"

            # Default to normal
            logging.debug(f"Text '{text[:50]}...' defaulting to 'normal'.")
            return "normal"

        def _traverse_and_categorize(data: Any, current_path: List[str]):
            """Recursively traverses and categorizes observations."""
            if isinstance(data, str):
                category = _categorize_observation(data)
                self.print_observation_info(data, current_path)  # Print detailed info
                if category == "positive":
                    self._add_to_nested_dict(positive_findings, current_path, data)
                elif category == "negative":
                    self._add_to_nested_dict(negative_findings, current_path, data)
                elif category == "normal":
                    self._add_to_nested_dict(normal_findings, current_path, data)
            elif isinstance(data, dict):
                for key, value in data.items():
                    _traverse_and_categorize(value, current_path + [key])
            elif isinstance(data, list):
                for item in data:
                    _traverse_and_categorize(item, current_path)

        _traverse_and_categorize(physical_exam_data, [])
        return {
            "positive": positive_findings,
            "negative": negative_findings,
            "normal": normal_findings
        }

    def print_observation_info(self, text: str, path: List[str]):
        """
        Prints detailed information about a text observation, including entities,
        negation status, and categorization.

        Args:
            text (str): The observation text.
            path (List[str]): The path in the nested dictionary.
        """
        logging.info(f"\n--- Observation Analysis ---")
        logging.info(f"Path: {' -> '.join(path)}")
        logging.info(f"Text: {text}")

        doc = self.analyze(text)
        if doc is None:
            logging.info("NLP Analysis: Failed to process text.")
            logging.info("Entities: None")
            logging.info("Categorization: Determined by keyword fallback")
            return

        # Extract entities and their context
        entities = []
        for ent in doc.ents:
            negation_status = "Negated" if hasattr(ent._, 'is_negated') and ent._.is_negated else "Affirmed"
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "negation": negation_status
            })

        logging.info("Entities Detected:")
        if entities:
            for ent in entities:
                logging.info(f"  - {ent['text']} (Label: {ent['label']}, Status: {ent['negation']})")
        else:
            logging.info("  - None")

        # Categorize for display
        category = "unknown"
        for ent in doc.ents:
            if hasattr(ent._, 'is_negated') and not ent._.is_negated:
                category = "positive"
                break
        if category != "positive":
            for ent in doc.ents:
                if hasattr(ent._, 'is_negated') and ent._.is_negated:
                    category = "negative"
                    break
            if category != "negative" and any(re.search(r"\b" + kw + r"\b", text.lower()) for kw in self.negation_keywords):
                category = "negative"
            elif any(re.search(r"\b" + kw + r"\b", text.lower()) for kw in self.normal_keywords):
                category = "normal"
            else:
                category = "normal"  # Default

        logging.info(f"Categorization: {category}")
        logging.info("--- End Observation Analysis ---\n")

if __name__ == "__main__":
    try:
        # Initialize analyzer with en_ner_clinical_large
        analyzer = ClinicalTextAnalyzerV2(model_name="")

        # Physical exam data (same as original)
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

        # Analyze physical exam
        results = analyzer.analyze_physical_exam(physical_exam_data)

        # Print summary
        print("\n=== Physical Exam Analysis Summary ===")
        import json
        print("\nPositive Findings:")
        print(json.dumps(results.get("positive", {}), indent=2))
        print("\nNegative Findings:")
        print(json.dumps(results.get("negative", {}), indent=2))
        print("\nNormal Findings:")
        print(json.dumps(results.get("normal", {}), indent=2))

    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)