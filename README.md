# Medical Text Analysis with spaCy, MedspaCy & scispaCy

This repository contains a Python script (`run.py`) demonstrating various Natural Language Processing (NLP) techniques for extracting and analyzing information from clinical text, specifically focusing on a sample Review of Systems (ROS). It leverages the power of spaCy, scispaCy (for clinical NER), MedspaCy (for clinical context analysis like negation), and Pandas (for data structuring).

This script is intended primarily for **testing and demonstrating** different methods of processing subsections of medical text, potentially pre-processed by other means (like extracting specific sections from a larger note).

## Goal

The main goal is to showcase how to:
1. Identify clinical entities within the text using a specialized NER model.
2. Determine the context of these entities (e.g., negated vs. affirmed).
3. Extract specific types of information based on context (e.g., only affirmed findings).
4. Analyze token-level attributes provided by the NLP pipeline.
5. Visualize the results in different ways.

## Features

*   **Clinical Named Entity Recognition (NER):** Uses scispaCy models (e.g., `en_ner_bc5cdr_md`) trained on biomedical/clinical text to identify entities like diseases, chemicals, symptoms.
*   **Context Detection:** Integrates MedspaCy's `ConTextComponent` to detect negation modifiers and determine if entities are affirmed or negated.
*   **Affirmed Findings Extraction:** Identifies system headers (e.g., "HENT:", "Neurological:") within the ROS and extracts only the *affirmed* entities associated with each system.
*   **Detailed Token Analysis:** Processes each token (word/punctuation) to extract attributes like lemma, Part-of-Speech (POS) tag, dependency relation, entity IOB tag, and the derived MedspaCy context (Affirmed/Negated). Outputs this analysis into a structured HTML table using Pandas.
*   **Visualization:**
    *   Generates standard entity highlighting using spaCy's `displaCy`.
    *   Generates context dependency visualization using MedspaCy's `visualize_dep` to show negation relationships (arrows).
*   **GPU Acceleration:** Automatically utilizes an available CUDA GPU via `spacy.prefer_gpu()` for faster processing. Falls back to CPU if GPU is unavailable or fails.

## Prerequisites

*   Python (>= 3.8 recommended)
*   pip (Python package installer)
*   Git (for cloning the repository)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv spacy-env
    # On Windows
    spacy-env\Scripts\activate
    # On macOS/Linux
    source spacy-env/bin/activate
    ```

3.  **Install Python packages:**
    A `requirements.txt` file *should* be created for easier installation. If not present, install manually:
    ```bash
    pip install spacy torch medspacy scispacy pandas
    # Adjust torch installation based on your system/CUDA version if needed
    # See: https://pytorch.org/get-started/locally/
    ```

4.  **Download required spaCy/scispaCy models:** The script currently defaults to `en_ner_bc5cdr_md`. Install the specific model(s) you intend to use:
    ```bash
    # Example for en_ner_bc5cdr_md (Check scispaCy docs for latest URL if needed)
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz

    # Example for en_core_sci_lg
    # pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
    ```
    *(Note: Ensure the model name in `run.py` matches an installed model).*

## Usage

Simply run the script from your terminal within the activated virtual environment:

```bash
python run.py