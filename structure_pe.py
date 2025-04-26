import json
import logging


def preprocess_physical_exam(pe_text: str) -> list[str]:
    """
    Preprocesses Physical Exam text to normalize indentation, preserving hierarchy.
    Removes 6 spaces from all lines, adjusting relative indentation.

    Args:
        pe_text (str): The raw physical exam text.

    Returns:
        list[str]: A list of lines with normalized indentation.
    """
    logging.info("Preprocessing Physical Exam text...")
    lines = pe_text.strip().splitlines()
    processed_lines = []
    outer_indent = 6  # Base indentation in input text

    for line in lines:
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces >= outer_indent:
            # Remove outer_indent spaces
            processed_line = line[outer_indent:]
        else:
            processed_line = line
        processed_lines.append(processed_line)

    logging.info("Finished preprocessing PE text.")
    return processed_lines

def parse_physical_exam(lines: list[str]) -> dict:
    """
    Parse preprocessed Physical Exam text into a nested dictionary.

    Args:
        lines (list[str]): List of preprocessed lines with normalized indentation.

    Returns:
        dict: A nested dictionary representing the exam structure.
    """
    def get_indent_level(line):
        return len(line) - len(line.lstrip())

    def split_sentences(value):
        if isinstance(value, str):
            sentences = [s.strip() + ('.' if not s.endswith('.') else '') for s in value.split('.') if s.strip()]
            return sentences if len(sentences) > 1 else value
        return value

    result = {"Physical Exam": {"General Notes": ""}}
    current_section = None
    current_subsection = None
    current_subsubsection = None

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue

        indent_level = get_indent_level(line)
        content = line.strip()

        # Root-level note (e.g., "Vitals and nursing note reviewed.")
        if indent_level == 0 and not content.endswith(":"):
            result["Physical Exam"]["General Notes"] = content
            index += 1
            continue

        # Main section (e.g., "Constitutional:", "HENT:")
        if indent_level == 0 and content.endswith(":"):
            current_section = content[:-1]
            result["Physical Exam"][current_section] = {}
            current_subsection = None
            current_subsubsection = None
            index += 1
            continue

        # Subsection (e.g., "Appearance:", "Mouth/Throat:")
        if indent_level == 1 and content.endswith(":"):
            current_subsection = content[:-1]
            result["Physical Exam"][current_section][current_subsection] = {} if current_subsection != "Comments" else []
            current_subsubsection = None
            index += 1
            continue

        # Sub-subsection (e.g., "Mouth:", "Reflex Scores:")
        if indent_level == 2 and content.endswith(":"):
            current_subsubsection = content[:-1]
            result["Physical Exam"][current_section][current_subsection][current_subsubsection] = {} if current_subsubsection != "Reflex Scores" else []
            index += 1
            continue

        # Findings
        if indent_level == 1 and current_subsection == "Comments":
            findings = [s.strip() + ('.' if not s.endswith('.') else '') for s in content.split(',') if s.strip()]
            if findings and findings[-1].endswith("tenderness. No midline cervical tenderness"):
                findings[-1] = findings[-1].replace("tenderness. No midline cervical tenderness", "").strip()
                if findings[-1]:
                    findings[-1] += "."
                findings.append("No midline cervical tenderness.")
            result["Physical Exam"][current_section][current_subsection] = [f for f in findings if f]
            index += 1
            continue

        if indent_level >= 1 and current_subsection:
            target = result["Physical Exam"][current_section][current_subsection]
            if current_subsubsection:
                target = target[current_subsubsection]
            if indent_level == 2 and current_subsubsection == "Reflex Scores":
                if isinstance(target, list):
                    target.append(content)
                else:
                    target = [content]
                result["Physical Exam"][current_section][current_subsection][current_subsubsection] = target
            elif indent_level == 1 and current_subsection in ["General", "Mental Status", "Cervical back"]:
                if isinstance(target, dict):
                    target = [content]
                else:
                    target.append(content)
                result["Physical Exam"][current_section][current_subsection] = target
            else:
                if isinstance(target, dict):
                    target = content
                else:
                    target += " " + content
                if current_subsubsection:
                    result["Physical Exam"][current_section][current_subsection][current_subsubsection] = target
                else:
                    result["Physical Exam"][current_section][current_subsection] = target
            index += 1
            continue

        index += 1

    return result

def main():
    # Example Physical Exam text
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

    # Preprocess and parse
    processed_lines = preprocess_physical_exam(pe_text)
    result = parse_physical_exam(processed_lines)

    # Print and save result
    print(f"\nFinal result:\n{json.dumps(result, indent=2)}")
    with open("physical_exam.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to physical_exam.json")

if __name__ == "__main__":
    main()