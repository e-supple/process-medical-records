import logging
import re
import spacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import torch
import medspacy # Keep medspacy for ConText
import os
import scispacy # Import scispacy (good practice, though loading does the work)
from medspacy.visualization import visualize_dep # <--- IMPORT THIS
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simplified GPU check (Keep as before)
def is_gpu_available():
    """Checks and logs GPU availability."""
    gpu_id = -1
    try:
        if torch.cuda.is_available():
            if spacy.prefer_gpu():
                try:
                     current_device_id = torch.cuda.current_device()
                     gpu_name = torch.cuda.get_device_name(current_device_id)
                     logging.info(f"spaCy utilizing GPU (ID: {current_device_id}) - {gpu_name}")
                     return True, current_device_id
                except Exception as e:
                     logging.warning(f"spacy.prefer_gpu() succeeded, but couldn't confirm device via torch: {e}")
                     return True, 0 # Placeholder ID
            else:
                logging.info("spacy.prefer_gpu() returned False. Using CPU.")
                spacy.require_cpu()
                return False, -1
        else:
            logging.info("No CUDA GPU available via torch. Using CPU.")
            spacy.require_cpu()
            return False, -1
    except Exception as e:
        logging.error(f"Error during GPU check: {e}")
        logging.info("Falling back to CPU.")
        spacy.require_cpu()
        return False, -1


# Pipeline loading function - MODIFIED to handle sciSpacy better
def load_spacy_pipeline_medspacy_std(model_name="en_core_sci_lg"): # <-- Default changed
    """Loads a spaCy model (recommend scispaCy for clinical) and adds MedspaCy ConTextComponent."""
    logging.info(f"Loading spaCy model: {model_name}...")
    try:
        # Load the spaCy/scispaCy model
        nlp = spacy.load(model_name)
        logging.info(f"Base model '{model_name}' loaded with components: {nlp.pipe_names}")

        # Add sentencizer if not present (good practice)
        if 'sentencizer' not in nlp.pipe_names:
            # Check if parser exists, add sentencizer before it if so
            if 'parser' in nlp.pipe_names:
                 logging.info("Adding 'sentencizer' pipe before parser.")
                 nlp.add_pipe("sentencizer", before="parser")
            else:
                 logging.info("Adding 'sentencizer' pipe first.")
                 nlp.add_pipe("sentencizer", first=True)


        # Add medspacy_context AFTER the NER component
        logging.info("Adding 'medspacy_context' pipe...")
        if 'ner' in nlp.pipe_names:
            nlp.add_pipe("medspacy_context", after="ner")
            logging.info("Added 'medspacy_context' pipe after NER.")
        else:
            # This case is less likely with standard models but handles edge cases
            logging.warning("NER component not found in the pipeline. Adding 'medspacy_context' last.")
            nlp.add_pipe("medspacy_context", last=True)

        # Verify medspacy_context was added
        if "medspacy_context" not in nlp.pipe_names:
             logging.error("'medspacy_context' pipe was not successfully added! Negation detection will fail.")
             # You might want to raise an error or return None here
        else:
             logging.info("Verified 'medspacy_context' is in the pipeline.")

        logging.info(f"Final pipeline: {nlp.pipe_names}")
        return nlp

    except OSError as e:
        logging.error(f"Model '{model_name}' loading failed: {e}.")
        logging.error(f"Ensure the model is installed (e.g., pip install <URL_for_{model_name}>) and the model name is correct.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during pipeline setup: {e}")
        return None

# get_entity_options function (Keep as before - add scispaCy entities if needed)
def get_entity_options():
    """Returns configuration for displaCy NER visualization."""
    # Combine standard entities with potential scispaCy entities
    # Check the specific scispaCy model's documentation for its entity labels
    # Example common scispaCy entities: ENTITY (generic), DISEASE, CHEMICAL
    entities = ["ENTITY", "DISEASE", "CHEMICAL", # Common scispaCy
                "PERSON", "ORG", "GPE", "DATE", "FAC", "LOC", "PRODUCT", "EVENT",
                "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY",
                "QUANTITY", "ORDINAL", "CARDINAL",
                "NEG_ENTITY"] # Your custom negation label
    colors = {
        "NEG_ENTITY": 'linear-gradient(90deg, #ffff66, #ff6600)', # Keep your negation color
        "PERSON": "linear-gradient(90deg, #ff9999, #ff6666)",
        "DISEASE": "linear-gradient(90deg, #66ccff, #3399ff)", # Example color for DISEASE
        "CHEMICAL": "linear-gradient(90deg, #ccff99, #99cc66)" # Example color for CHEMICAL
        # Add other colors as needed
    }
    options = {"ents": entities, "colors": colors}
    return options

# create_neg_matcher function (Keep as before)
def create_neg_matcher(nlp, negated_entity_texts):
    """Creates a PhraseMatcher for explicitly found negated entity texts."""
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in negated_entity_texts if text]
    if patterns:
        matcher.add("NEG_ENTITY", patterns)
        logging.info(f"Created PhraseMatcher for {len(negated_entity_texts)} negated entity texts.")
    else:
        logging.info("No valid negated entity texts provided to create matcher.")
    return matcher

# relabel_negated_entities function (Keep as before)
def relabel_negated_entities(doc, matcher, nlp):
    """Relabels entities matched by the PhraseMatcher as NEG_ENTITY."""
    matches = matcher(doc)
    if not matches:
        # If no phrase matches, return the doc as is (medspacy attributes are still there)
        logging.info("No phrase matches found for relabeling.")
        # We still want to potentially visualize the original doc with medspacy negations
        # So let's keep the original ents from the doc processed by medspacy
        return doc # Return doc with original ents (which have ._.is_negated)

    spans_to_relabel = []
    matched_texts = set()
    for match_id, start, end in matches:
        # Ensure the label is a string (spaCy expects string labels for Span)
        label_str = nlp.vocab.strings[match_id]
        span = Span(doc, start, end, label=label_str) # Use the string label
        if span.text not in matched_texts:
            spans_to_relabel.append(span)
            matched_texts.add(span.text)

    original_ents = list(doc.ents)
    filtered_orig_ents = []
    for ent in original_ents:
        overlaps = False
        for neg_span in spans_to_relabel:
            # Check for overlap
            if (ent.start < neg_span.end and ent.end > neg_span.start):
                overlaps = True
                logging.debug(f"Entity '{ent.text}' ({ent.label_}) overlaps with NEG_ENTITY span '{neg_span.text}'. Removing original.")
                break
        if not overlaps:
            filtered_orig_ents.append(ent)

    # Combine the non-overlapping original entities with the new NEG_ENTITY spans
    combined_spans = filtered_orig_ents + spans_to_relabel

    # Filter for valid spans (handles potential duplicates or overlaps within the combined list)
    # This is crucial as the relabeling might create overlapping spans if not careful
    final_ents = spacy.util.filter_spans(combined_spans)

    try:
        # Create a *new* doc to avoid modifying the original doc's entities
        # OR modify in place carefully if memory is a concern
        # For simplicity here, let's try modifying in place, but beware of side effects
        # If issues arise, creating a copy is safer: doc_copy = doc[:] # shallow copy
        doc.ents = final_ents
        logging.info(f"Relabeled entities. Original count: {len(original_ents)}, Final count: {len(doc.ents)}")
    except ValueError as e:
        logging.error(f"Error setting doc.ents after filtering/relabeling: {e}.")
        logging.warning("Falling back to original entities from medspacy.")
        # Revert to original ents if setting fails
        doc.ents = tuple(original_ents)

    return doc

def remove_negated_entity_text(original_text, doc):
    """
    Removes text spans corresponding to negated entities from the original text.

    Args:
        original_text (str): The original text string.
        doc (spacy.Doc): The processed spaCy Doc object containing entities
                         with MedspaCy's ._.is_negated attribute.

    Returns:
        str: A new string with the negated entity text removed,
             or the original string if no negated entities are found.
    """
    logging.info("Attempting to remove negated entity text...")
    negated_spans_indices = []
    for ent in doc.ents:
        # Check if the entity is negated
        if hasattr(ent._, 'is_negated') and ent._.is_negated:
            negated_spans_indices.append((ent.start_char, ent.end_char, ent.text)) # Store text for logging

    if not negated_spans_indices:
        logging.info("No negated entities found to remove.")
        return original_text

    # Sort spans by start character index to process sequentially
    negated_spans_indices.sort(key=lambda x: x[0])

    logging.info(f"Found {len(negated_spans_indices)} negated spans to remove:")
    for start, end, text in negated_spans_indices:
        logging.info(f"  - Span: [{start}:{end}], Text: '{text}'")


    kept_parts = []
    last_end = 0
    for start, end, _ in negated_spans_indices:
        # Keep the text from the end of the last removed span up to the start of this one
        if start > last_end: # Ensure we don't append empty strings if spans are adjacent
             kept_parts.append(original_text[last_end:start])
        # Update the position to skip the current negated span
        last_end = end

    # Add the remaining part of the string after the last negated span
    kept_parts.append(original_text[last_end:])

    # Join the kept parts
    filtered_text = "".join(kept_parts)

    # Optional: Clean up extra whitespace potentially left by removals
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()

    logging.info("Finished removing negated entity text.")
    return filtered_text

# --- New Function to Extract Affirmed Findings by System ---
def extract_affirmed_findings_by_system(text, doc):
    """
    Identifies system sections and extracts affirmed entities within each section.

    Args:
        text (str): The original text string (e.g., ROS).
        doc (spacy.Doc): The processed spaCy Doc with NER and MedspaCy context.

    Returns:
        dict: A dictionary where keys are system names (str) and values are
              lists of tuples `(entity_text, entity_label)` for affirmed entities.
    """
    logging.info("Extracting affirmed findings by system...")
    findings = {}

    # 1. Find System Headers using Regex
    # This pattern looks for a newline, optional whitespace,
    # capitalized words (potentially with slashes/hyphens), followed by a colon.
    # It captures the system name (group 1).
    header_pattern = re.compile(r"^\s*([A-Z][a-zA-Z/\-]+(?: [A-Z][a-zA-Z/\-]+)*):\s*", re.MULTILINE)
    # header_pattern = re.compile(r"\n\s*([A-Z][a-zA-Z/\-]+):\s*") # Simpler alternative if names are single words

    headers = []
    for match in header_pattern.finditer(text):
        system_name = match.group(1).strip()
        # Use end of the *match* as the start of the content for this section
        content_start_char = match.end()
        headers.append({"name": system_name, "start": match.start(), "content_start": content_start_char})

    if not headers:
        logging.warning("No system headers found using the defined pattern.")
        # Optionally, process the whole doc as one section
        # return {"UNCATEGORIZED": [(ent.text, ent.label_) for ent in doc.ents if hasattr(ent._, 'is_negated') and not ent._.is_negated]}
        return {}

    # Sort headers by their start position
    headers.sort(key=lambda x: x["start"])

    # Add a dummy end-marker for the last section
    text_end = len(text)

    # 2. Get Affirmed Entities
    affirmed_entities = []
    for ent in doc.ents:
        if hasattr(ent._, 'is_negated') and not ent._.is_negated:
            # Store text, label, and character positions
            affirmed_entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

    logging.info(f"Found {len(affirmed_entities)} affirmed entities.")

    # 3. Map Entities to Sections
    num_headers = len(headers)
    for i, header in enumerate(headers):
        system_name = header["name"]
        section_content_start = header["content_start"]

        # Determine the end of this section's content
        if i + 1 < num_headers:
            section_content_end = headers[i+1]["start"] # Ends where the next header begins
        else:
            section_content_end = text_end # Last section goes to the end of the text

        logging.debug(f"Processing section: '{system_name}' ({section_content_start}-{section_content_end})")
        findings[system_name] = []

        for ent in affirmed_entities:
            # Check if the entity falls within the character span of this section's content
            if ent["start_char"] >= section_content_start and ent["end_char"] <= section_content_end:
                logging.debug(f"  - Matched affirmed entity: '{ent['text']}' ({ent['label']})")
                findings[system_name].append((ent["text"], ent["label"]))

    logging.info("Finished mapping affirmed entities to systems.")
    # Filter out systems with no affirmed findings
    return {k: v for k, v in findings.items() if v}

def generate_visual(ros_doc):
    # --- Visualizations (Optional) ---
    # --- Visualizations (Optional) ---
    # (Keep the visualization code as before if desired)
    print("\n--- Visualization: Entities Only ---")
    print("(Check ros_visualization_entities_only.html)")
    options_ent = get_entity_options()
    html_ent = displacy.render(ros_doc, style='ent', options=options_ent, page=True)
    output_path_ent = "ros_visualization_entities_only.html"
    try:
        with open(output_path_ent, "w", encoding="utf-8") as f:
            f.write(html_ent)
        logging.info(f"Entity-only visualization saved to {output_path_ent}")
    except Exception as e:
        logging.error(f"Failed to save entity-only visualization: {e}")

# --- MODIFIED Function to Analyze Token Attributes (Returns DataFrame) ---
def analyze_token_attributes(doc):
    """
    Iterates through tokens in a Doc, extracts attributes, and returns
    them as a Pandas DataFrame.

    Args:
        doc (spacy.Doc): The processed spaCy Doc object.

    Returns:
        pandas.DataFrame: A DataFrame containing token attributes.
    """
    logging.info("Analyzing token attributes for DataFrame...")

    token_data = [] # List to hold data for each token

    # Pre-build a map from token index to its entity span
    token_to_entity_map = {}
    for entity_span in doc.ents:
        for i in range(entity_span.start, entity_span.end):
            token_to_entity_map[i] = entity_span

    # Iterate through each token
    for token in doc:
        # Basic spaCy attributes
        text = token.text
        lemma = token.lemma_
        pos = token.pos_
        tag = token.tag_
        dep = token.dep_
        head = token.head.text
        is_space = token.is_space # Good to know if it's just whitespace

        # Entity information (NER)
        ent_type = token.ent_type_ or "O"
        ent_iob = token.ent_iob_

        # MedspaCy Context
        entity_context = "N/A"
        entity_span = token_to_entity_map.get(token.i)
        if entity_span:
            if hasattr(entity_span._, 'is_negated'):
                entity_context = "NEGATED" if entity_span._.is_negated else "AFFIRMED"

        # Add data for this token to the list
        token_data.append({
            "TEXT": text,
            "LEMMA": lemma,
            "POS": pos,
            "TAG": tag,
            "DEP": dep,
            "HEAD": head,
            "ENTITY_IOB": ent_iob,
            "ENTITY_TYPE": ent_type,
            "ENTITY_CONTEXT": entity_context,
            "IS_SPACE": is_space,
            "TOKEN_IDX": token.i # Add token index for reference
        })

    # Create DataFrame
    df = pd.DataFrame(token_data)
    logging.info("Finished analyzing token attributes into DataFrame.")
    return df

# Main execution block - MODIFIED to call the new function
def main():
    logging.info("Script starting...")

    gpu_available, gpu_id = is_gpu_available()
    print(f"gpu_available: {gpu_available}")
    print(f"GPU ID: {gpu_id}")

    model_to_use = "en_ner_bc5cdr_md" # Using the robust clinical model
    nlp = load_spacy_pipeline_medspacy_std(model_name=model_to_use)
    if nlp is None:
        logging.error("Exiting due to NLP pipeline loading failure.")
        exit()

    ros = """
        Review of Systems
        HENT: Negative for facial swelling and nosebleeds.
        Eyes: Negative for photophobia and visual disturbance.
        Respiratory: Negative for shortness of breath.
        Cardiovascular: Negative for chest pain.
        Gastrointestinal: Negative for nausea and vomiting.
        Genitourinary: Negative for flank pain and hematuria.
        Musculoskeletal: Positive for arthralgias (left arm pain). Negative for back pain and neck pain.
            Left lateral trapezius pain with radiation down left arm
        Skin: Negative for wound.
        Neurological: Positive for numbness. Negative for dizziness, syncope, weakness and headaches.
        Psychiatric/Behavioral: Negative for confusion.
    """

    # --- Processing ROS ---
    print("\n--- Processing Review of Systems (ROS) ---")
    logging.info(f"Running ROS through {model_to_use} NER and Context pipeline...")
    ros_doc = nlp(ros) # Process the original ROS text

    # --- Extract Negated Entities using MedspaCy ---
    # (Keep the detailed printout section as before for diagnosis)
    print("\nEntities found by NER and their negation status (via MedspaCy):")
    ros_negated_entities = []
    ros_affirmed_entities = []
    has_negation_info = False
    if not ros_doc.ents:
         print(" - No entities identified by the NER model.")
    else:
        for ent in ros_doc.ents:
            negation_status = "UNKNOWN"
            is_negated = False
            if hasattr(ent._, 'is_negated'):
                has_negation_info = True
                if ent._.is_negated:
                    negation_status = "NEGATED"
                    ros_negated_entities.append(ent)
                    is_negated = True
                else:
                    negation_status = "AFFIRMED"
                    ros_affirmed_entities.append(ent)
            elif hasattr(ent._, 'context_attributes'):
                 if 'is_negated' in ent._.context_attributes and ent._.context_attributes['is_negated']:
                      has_negation_info = True
                      negation_status = "NEGATED"
                      ros_negated_entities.append(ent)
                      is_negated = True
                 else:
                      has_negation_info = True
                      negation_status = "AFFIRMED"
                      ros_affirmed_entities.append(ent)
            print(f"- '{ent.text}' (Label: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}, Negation: {negation_status})")
        if not has_negation_info:
             logging.warning("MedspaCy context attributes not found. Check pipeline.")

    # --- Print Summary of Negated Entities ---
    print("\nSummary: Negated Entities identified by MedspaCy:")
    if ros_negated_entities:
         for ent in ros_negated_entities:
            print(f"- {ent.text} (Original Label: {ent.label_})")
    else:
        print(" - None identified.")

    # --- Processing ROS ---
    print("\n--- Processing Review of Systems (ROS) ---")
    logging.info(f"Running ROS through {model_to_use} NER and Context pipeline...")
    ros_doc = nlp(ros)

    # --- Extract and Print Affirmed Findings by System ---
    print("\n--- Extracting Affirmed Findings by System ---")
    affirmed_by_system = extract_affirmed_findings_by_system(ros, ros_doc)

    if affirmed_by_system:
        print("\nAffirmed Findings Summary:")
        print("-" * 30)
        for system, findings_list in affirmed_by_system.items():
            print(f"System: {system}")
            if findings_list:
                for entity_text, entity_label in findings_list:
                    print(f"  - {entity_text} ({entity_label})")
            else:
                # This part might not be reached if empty lists are filtered out earlier
                print("  (No affirmed findings detected in this section)")
            print("-" * 10) # Separator between systems
    else:
        print("\nNo affirmed findings were extracted or associated with system headers.")

    # --- Detailed Token Analysis into DataFrame ---
    print("\n--- Analyzing Tokens into DataFrame ---")
    token_df = analyze_token_attributes(ros_doc)

    # --- Output the DataFrame ---
    # Option 1: Print to console (might be wide)
    # print("\nToken Analysis DataFrame (Console Output):")
    # pd.set_option('display.max_rows', None) # Show all rows
    # pd.set_option('display.max_columns', None) # Show all columns
    # pd.set_option('display.width', 1000) # Try to increase width
    # print(token_df)

    # Option 2: Save to HTML (Recommended for viewing)
    output_html_path = "token_analysis.html"
    print(f"\nSaving Token Analysis DataFrame to HTML: {output_html_path}")
    try:
        # index=False prevents writing the DataFrame index as a column
        token_df.to_html(output_html_path, index=False, border=1)
        logging.info(f"DataFrame successfully saved to {output_html_path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to HTML: {e}")


    # generate html visualizer
    # generate_visual(ros_doc)
    
    # --- Detailed Token Analysis --- ADDED THIS CALL
    analyze_token_attributes(ros_doc)
    
    logging.info("Script finished.")

if __name__ == "__main__":
    main()