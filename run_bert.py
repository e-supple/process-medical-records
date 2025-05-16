import os
import glob
from pathlib import Path
from typing import List, Union
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# import scikit-learn
from pathlib import Path
import pdfplumber
import fitz

def pdf_to_text(pdf_path, txt_path):
    """
    Converts text content of a PDF file to a plain text file,
    attempting to preserve the original layout (lines, spacing, paragraphs).

    Args:
        pdf_path (str): The path to the input PDF file.
        txt_path (str): The path to the output text file.
    """
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        all_text = ""

        # Iterate through each page
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text(sort=True) # Explicitly request sorting

            all_text += page_text
            if page_num < doc.page_count - 1:
                all_text += "\n\n--- Page Break ---\n\n"
                # Or just: all_text += "\n\n"

        # Close the document
        doc.close()

        # Write the extracted text to the output file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(all_text)

        print(f"Successfully converted '{pdf_path}' to '{txt_path}'.")

    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def check_gpu_availability():
    """Check if a GPU is available and return the device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def summarize_text(text, tokenizer, model, device, max_length=150, min_length=50):
    """Generate a summary of the input text using the summarization model."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_files(text_files):
    for text_file_path in text_files:
        print(f"Processing {text_file_path}...")
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        print(f"Embedding shape for {text_file_path}: {embeddings.shape}")
        # Save or process embeddings as needed

def read_text_file(file_path):
    """Read the content of a text file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
        print(
            f"Text file loaded successfully. \n {text[:100]}..."
        )  # Print first 100 characters for verification
    except Exception as e:
        print(f"Error reading text file: {e}")
        exit(1)

def save_text_file(text: Union[str, List[str]], output_file: str) -> None:
    """
    Writes a list of strings or a string to a text file.

    Args:
        data: A list of strings or a string to write to the file.
        output_file: The name of the file to write to.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        if isinstance(text, list):
            f.writelines(f"{str(item)}\n" for item in text)
        elif isinstance(text, str):
            f.write(text)
        else:
            raise TypeError("data must be a string or a list")
    print(f"Saved preprocessed text to {output_file}")
    
def rec_similarity(emb1, emb2):
    """
    Calculate the cosine similarity between two embeddings.
    
    Args:
        emb1 (numpy.ndarray): First embedding vector.
        emb2 (numpy.ndarray): Second embedding vector.
        
    Returns:
        float: Cosine similarity score between the two embeddings.
    """
    # Assuming you have another embedding (e.g., from another record)
    embedding1 = embeddings.numpy()  # Convert to numpy
    embedding2 = embeddings.numpy()  # Replace with another embedding
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Similarity between records: {similarity[0][0]}")
    return similarity

def run_summarization(text_file_path):
    # Load the BERT tokenizer and model for embeddings
    try:
        print("Loading BERT model and tokenizer...")
        bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT").to(device)
        print("BERT model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading BERT model or tokenizer: {e}")
        exit(1)

    # Load the summarization model and tokenizer
    try:
        print("Loading summarization model and tokenizer...")
        summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        print("Summarization model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading summarization model or tokenizer: {e}")
        exit(1)

    # Read the text file
    try:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        print("Text file loaded successfully.")
    except Exception as e:
        print(f"Error reading text file: {e}")
        exit(1)

    # Generate summary
    try:
        print("Generating summary...")
        summary = summarize_text(text, summ_tokenizer, summ_model, device)
        print("Summary generated successfully.")
        print("\nSummary:")
        print(summary)
    except Exception as e:
        print(f"Error generating summary: {e}")
        exit(1)

    # Tokenize the text for BERT embeddings
    try:
        inputs = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        print("Text tokenized successfully for BERT.")
    except Exception as e:
        print(f"Error tokenizing text for BERT: {e}")
        exit(1)

    # Run the BERT model for embeddings
    try:
        with torch.no_grad():
            outputs = bert_model(**inputs)
        print("BERT model inference completed.")
        last_hidden_states = outputs.last_hidden_state
        print(f"Output shape: {last_hidden_states.shape}")  # [batch_size, sequence_length, hidden_size]
        embeddings = torch.mean(last_hidden_states, dim=1)  # Average across tokens
        print(f"Embedding shape: {embeddings.shape}")  # [batch_size, hidden_size]
    except Exception as e:
        print(f"Error during BERT model inference: {e}")
        exit(1)

if __name__ == "__main__":
    # Load the tokenizer and model from the Hugging Face Hub
    # You can replace "emilyalsentzer/Bio_Discharge_Summary_BERT" with any other model name
    # available on the Hugging Face Model Hub.
    BASE_DIR = Path(__file__).resolve().parent
    print(f"Base Directory: {BASE_DIR}")
    RECORDS_DIR = BASE_DIR / "data" / "records"
    geisinger_cmc_file = RECORDS_DIR / "Gragirene" / "Geisinger CMC" / "Thornhill, Janice - Geisinger.txt"
    northeast_rehab_file = RECORDS_DIR / "Gragirene" / "Northeast Rehab.pdf"
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    

    # Load the BERT tokenizer and model for embeddings
    try:
        print("Loading BERT model and tokenizer...")
        bert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
        bert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT").to(device)
        print("BERT model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading BERT model or tokenizer: {e}")
        exit(1)

    # Load the summarization model and tokenizer
    try:
        print("Loading summarization model and tokenizer...")
        summ_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        summ_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        print("Summarization model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading summarization model or tokenizer: {e}")
        exit(1)

    # Read the text file
    try:
        with open(geisinger_cmc_file, 'r', encoding='utf-8') as file:
            text = file.read()
        print("Text file loaded successfully.")
    except Exception as e:
        print(f"Error reading text file: {e}")
        exit(1)

    # Generate summary
    try:
        print("Generating summary...")
        summary = summarize_text(text, summ_tokenizer, summ_model, device)
        print("Summary generated successfully.")
        print("\nSummary:")
        print(summary)
    except Exception as e:
        print(f"Error generating summary: {e}")
        exit(1)

    # Tokenize the text for BERT embeddings
    try:
        inputs = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        print("Text tokenized successfully for BERT.")
    except Exception as e:
        print(f"Error tokenizing text for BERT: {e}")
        exit(1)

    # Run the BERT model for embeddings
    try:
        with torch.no_grad():
            outputs = bert_model(**inputs)
        print("BERT model inference completed.")
        last_hidden_states = outputs.last_hidden_state
        print(f"Output shape: {last_hidden_states.shape}")  # [batch_size, sequence_length, hidden_size]
        embeddings = torch.mean(last_hidden_states, dim=1)  # Average across tokens
        print(f"Embedding shape: {embeddings.shape}")  # [batch_size, hidden_size]
    except Exception as e:
        print(f"Error during BERT model inference: {e}")
        exit(1)
    
    
    
    