import nltk
import re
import json
import logging
from transformers import pipeline
from collections import defaultdict

# Download NLTK resources
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Download NLTK resources
nltk.download('punkt')

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", model="blaze999/Medical-NER")

# Relevant entity types for findings
RELEVANT_ENTITY_TYPES = ["SIGN_SYMPTOM", "DISEASE_DISORDER"]

# Negative indicators to exclude normal/negative findings
NEGATIVE_INDICATORS = ["no", "not", "normal", "none", "absent", "negative"]

# Physical exam text
pe_text = """
      Physical Exam
      Vitals and nursing note reviewed.
      Constitutional:
       Appearance: She is not diaphoretic.
      HENT:
       Head: Normocephalic and atraumatic.
       Mouth/Throat:
       Mouth: Mucous membranes are moist.
      Eyes:
       General: No scleral icterus.
       Extraocular Movements: Extraocular movements intact.
       Conjunctiva/sclera: Conjunctivae normal.
      Neck:
       Vascular: No JVD.
       Comments: No midline cervical tenderness
      Cardiovascular:
       Rate and Rhythm: Normal rate and regular rhythm.
       Heart sounds: No murmur heard.
      Pulmonary:
       Effort: No tachypnea or accessory muscle usage.
       Breath sounds: No wheezing or rales.
      Abdominal:
       General: There is no distension.
       Palpations: Abdomen is soft.
       Tenderness: There is no abdominal tenderness.
      Musculoskeletal:
       General: Tenderness (Left posterior shoulder) present. No swelling or deformity.
       Cervical back: Normal range of motion. No edema or erythema.
       Comments: Left posterior deltoid tenderness, left lateral trapezius tenderness, left paraspinal cervical tenderness. No midline cervical tenderness
      Skin:
       General: Skin is warm and dry.
       Capillary Refill: Capillary refill takes less than 2 seconds.
       Coloration: Skin is not pale.
       Findings: No rash.
      Neurological:
       General: No focal deficit present.
       Mental Status: She is alert. Mental status is at baseline.
       Cranial Nerves: Cranial nerves 2-12 are intact.
       Sensory: No sensory deficit.
       Motor: No weakness.
       Deep Tendon Reflexes:
       Reflex Scores:
        Patellar reflexes are 2+ on the right side and 2+ on the left side.
        Achilles reflexes are 2+ on the right side and 2+ on the left side.
      Psychiatric:
       Mood and Affect: Mood normal.
       Behavior: Behavior normal.
"""

# --- Preprocessing Functions ---
def preprocess_physical_exam(pe_text: str) -> list[str]:
    logging.info("Preprocessing Physical Exam text...")
    lines = pe_text.strip().splitlines()
    processed_lines_output = []
    found_start = False
    ignored_headers = {"physical exam"}
    outer_indent = -1

    first_header_index = -1
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line and stripped_line.endswith(':') and stripped_line[0].isupper():
            header_text = stripped_line[:-1].strip().lower()
            if header_text not in ignored_headers:
                outer_indent = len(line) - len(line.lstrip(' '))
                first_header_index = i
                logging.info(f"Found start of structured PE list at line {i}: '{stripped_line}'. Detected outer_indent = {outer_indent}")
                found_start = True
                break

    if not found_start:
        logging.warning("Could not find the start of the structured list in the PE text.")
        return []

    logging.info(f"Processing lines starting from index {first_header_index}, removing outer indent of {outer_indent}...")
    header_pattern = re.compile(r"^[A-Z].*?:\s*$")

    for i in range(first_header_index, len(lines)):
        line = lines[i]
        stripped_line = line.strip()

        if not stripped_line:
            logging.debug(f" Line {i}: Skipping blank line.")
            continue

        leading_spaces = len(line) - len(line.lstrip(' '))
        normalized_line = line[outer_indent:] if leading_spaces >= outer_indent else line
        normalized_indent = max(0, leading_spaces - outer_indent)

        logging.debug(f" Line {i}: Orig='{line}', Norm='{normalized_line}', NormIndent={normalized_indent}")

        is_header_format = header_pattern.match(stripped_line) is not None

        if normalized_indent == 0 and not is_header_format and processed_lines_output:
            logging.debug(f"  -> Treating as continuation of previous line.")
            processed_lines_output[-1] += " " + stripped_line
            logging.debug(f"     Merged result: '{processed_lines_output[-1]}'")
        else:
            processed_lines_output.append(normalized_line)
            logging.debug(f"  -> Added as new line.")

    logging.info("Finished preprocessing PE text.")
    print(f"Processed lines output: {processed_lines_output}")
    return processed_lines_output

def parse_section_text(text):
    """Parse a section's text into a nested dictionary, handling subheaders and sentence lists."""
    result = {}
    lines = text.strip().splitlines()
    current_header = None
    current_key = None
    
    header_pattern = re.compile(r"^\s*([^:]+?):\s*$")
    key_value_pattern = re.compile(r"^\s*([^:]+?):\s*(.+)$")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        header_match = header_pattern.match(line)
        if header_match:
            header_name = header_match.group(1).strip()
            if current_header is None:
                current_header = header_name
                result[current_header] = {}
            else:
                result[current_header][header_name] = []
                current_key = header_name
            continue

        kv_match = key_value_pattern.match(line)
        if kv_match:
            key, value = kv_match.groups()
            key = key.strip()
            value = value.strip()
            if key and value:
                target_dict = result[current_header] if current_header else result
                sentences = sent_tokenize(value)
                target_dict[key] = sentences if len(sentences) >= 2 else value
                current_key = key
            continue

        if current_header and current_key:
            sentences = sent_tokenize(line)
            target = result[current_header][current_key]
            if isinstance(target, str):
                result[current_header][current_key] = [target] + sentences
            elif isinstance(target, list):
                result[current_header][current_key].extend(sentences)

    return result

def process_value(value):
    """Recursively process a value to handle nested key:value pairs or sentence lists."""
    if isinstance(value, list):
        return [process_value(item) for item in value]
    if not isinstance(value, str):
        return value
    if ': ' in value:
        try:
            new_key, new_value = value.split(': ', 1)
            return {new_key: process_value(new_value)}
        except ValueError:
            return value
    return value

def process_dictionary(data):
    """Process the dictionary to handle nested structures and sentence lists."""
    result = {}
    for section, text in data.items():
        if isinstance(text, str):
            items = parse_section_text(text)
        else:
            items = text
        result[section] = {}
        for key, value in items.items():
            result[section][key] = process_value(value)
    return result

# --- Findings Extraction ---
def aggregate_entities(entities):
    """Aggregate subword tokens into full entities."""
    aggregated = []
    current_entity = []
    current_type = None
    current_sentence = None

    for entity in entities:
        label = entity['entity']
        entity_type = label.split('-')[-1] if '-' in label else label

        if entity_type in RELEVANT_ENTITY_TYPES:
            if current_type == entity_type and (label.startswith('I-') or current_entity[-1]['end'] >= entity['start'] - 10):
                current_entity.append(entity)
            else:
                if current_entity:
                    full_text = "".join(e['word'].replace("▁", " ") for e in current_entity).strip()
                    if len(full_text) > 3 and not any(bad in full_text.lower() for bad in ["condition", "reflexes", "murmur"]):
                        aggregated.append((full_text, current_type, current_sentence))
                current_entity = [entity]
                current_type = entity_type
                current_sentence = entity.get('sentence', current_sentence)
        else:
            if current_entity:
                full_text = "".join(e['word'].replace("▁", " ") for e in current_entity).strip()
                if len(full_text) > 3 and not any(bad in full_text.lower() for bad in ["condition", "reflexes", "murmur"]):
                    aggregated.append((full_text, current_type, current_sentence))
                current_entity = []
                current_type = None

    if current_entity:
        full_text = "".join(e['word'].replace("▁", " ") for e in current_entity).strip()
        if len(full_text) > 3 and not any(bad in full_text.lower() for bad in ["condition", "reflexes", "murmur"]):
            aggregated.append((full_text, current_type, current_sentence))

    return aggregated

def extract_findings_from_dict(section_dict):
    """Extract positive/abnormal findings from the Musculoskeletal section of the dictionary."""
    findings = []

    def process_value(value, parent_key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                process_value(v, f"{parent_key}.{k}" if parent_key else k)
        elif isinstance(value, list):
            for item in value:
                process_value(item, parent_key)
        elif isinstance(value, str) and value.strip():
            # Split list-like sentences (e.g., "Left posterior deltoid tenderness, ...")
            sub_sentences = [s.strip() for s in value.split(',') if s.strip()]
            for sub_sentence in sub_sentences:
                # Skip if contains negative indicators
                sub_sentence_lower = sub_sentence.lower()
                if any(indicator in sub_sentence_lower for indicator in NEGATIVE_INDICATORS):
                    continue

                # Only process Musculoskeletal-related values
                if "musculoskeletal" not in parent_key.lower() and "tenderness" not in sub_sentence_lower:
                    continue

                # Extract entities
                entities = ner_pipeline(sub_sentence)
                for entity in entities:
                    entity['sentence'] = sub_sentence
                aggregated_entities = aggregate_entities(entities)

                # Filter relevant entities
                for text, entity_type, sentence in aggregated_entities:
                    if entity_type in RELEVANT_ENTITY_TYPES:
                        # Extract location
                        location = None
                        if "(" in sentence and ")" in sentence:
                            location_match = re.search(r"\((.*?)\)", sentence)
                            if location_match:
                                location = location_match.group(1).strip()
                        else:
                            location_match = re.search(r"(left\s+[\w\s]+)\s+tenderness", sentence, re.IGNORECASE)
                            if location_match:
                                location = location_match.group(1).strip()

                        finding = {
                            "Entity": text,
                            "Type": entity_type,
                            "Location": location,
                            "Context": sentence.strip()
                        }
                        findings.append(finding)

    # Process the Musculoskeletal section
    musculoskeletal = section_dict.get("Musculoskeletal", {})
    process_value(musculoskeletal, "Musculoskeletal")

    return findings

def preprocess_and_extract(text):
    """Preprocess text and extract findings."""
    # Preprocess the text
    processed_lines = preprocess_physical_exam(text)
    if not processed_lines:
        print("Preprocessing returned empty list.")
        return []

    # Convert to string for parsing
    processed_text = "\n".join(processed_lines)
    sections = {}
    current_section = None
    for line in processed_lines:
        line = line.strip()
        if line.endswith(':'):
            current_section = line[:-1].strip()
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line)

    # Parse each section into a nested dictionary
    section_dict = {}
    for section_name, lines in sections.items():
        section_text = "\n".join(lines)
        parsed_dict = parse_section_text(section_text)
        section_dict[section_name] = parsed_dict.get(section_name, {})

    print(f"Parsed section dictionary: {json.dumps(section_dict, indent=2)}")

    # Extract findings
    findings = extract_findings_from_dict(section_dict)
    return findings

def save_findings(findings, output_path="findings.json"):
    """Save findings to a JSON file."""
    print(f"Saving findings to {output_path}")
    print(f"Findings: {json.dumps(findings, indent=2)}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    findings = preprocess_and_extract(pe_text)
    save_findings(findings)
    print("Findings extracted and saved to findings.json")