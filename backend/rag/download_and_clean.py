import pandas as pd
import re
from datasets import load_dataset
import os

def clean_text(text: str) -> str:
    """
    A simple cleaning function to:
    1. Remove HTML-like tags (e.g., <p>, <b>).
    2. Remove URLs.
    3. Normalize whitespace (remove newlines, tabs, and extra spaces).
    4. Strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML-like tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    
    # 3. Normalize whitespace
    text = re.sub(r'[\n\t]+', ' ', text) # Remove newlines and tabs
    text = re.sub(r'\s{2,}', ' ', text)   # Replace 2+ spaces with a single space
    
    # 4. Strip leading/trailing whitespace
    return text.strip()

def process_counsel_chat():
    """
    Loads the Counsel-Chat dataset, cleans the 'answer' column, 
    and saves the "therapist talk" to a CSV.
    """
    print("--- 1. Processing English: 'nbertagnolli/counsel-chat' ---")
    try:
        # Load the dataset using the CORRECT path
        dataset = load_dataset("nbertagnolli/counsel-chat", split='train')
        df = dataset.to_pandas()
        
        print(f"Loaded {len(df)} rows from Counsel-Chat.")
        
        # We only care about the "therapist talk" (the 'answerText')
        df_cleaned = pd.DataFrame()
        df_cleaned['therapist_text'] = df['answerText'].apply(clean_text)
        
        # Drop any empty rows that resulted from cleaning
        df_cleaned = df_cleaned[df_cleaned['therapist_text'].str.len() > 0]
        
        # Save to CSV
        output_file = "counsel_chat_cleaned.csv"
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Successfully cleaned and saved {len(df_cleaned)} rows to {output_file}")
        
    except Exception as e:
        print(f"❌ ERROR: Could not process Counsel-Chat. {e}")

def process_arabic_empathetic_conversations():
    """
    Loads the 'arbml/arabic_empathetic_conversations' dataset as our 
    Arabic base layer. Cleans the correct columns and saves to CSV.
    """
    print("\n--- 2. Processing Arabic: 'arbml/arabic_empathetic_conversations' ---")
    
    try:
        # Load the dataset
        dataset = load_dataset("arbml/arabic_empathetic_conversations", split='train')
        df = dataset.to_pandas() 
        
        print(f"Loaded {len(df)} rows from arabic_empathetic_conversations.")
        
        # --- THIS IS THE FIX ---
        # The columns are 'context' and 'response', not 'text'
        # We will grab text from BOTH columns to get all conversational data.
        
        lines_from_context = df['context'].apply(clean_text)
        lines_from_response = df['response'].apply(clean_text)
        
        # Combine both columns into a single list
        all_therapist_lines = pd.concat([lines_from_context, lines_from_response])

        df_cleaned = pd.DataFrame(all_therapist_lines, columns=['therapist_text'])
        
        # Filter out short/junk lines
        df_cleaned = df_cleaned[df_cleaned['therapist_text'].str.split().str.len() > 3]
        
        # Drop duplicates
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
        
        # Drop any final empty rows
        df_cleaned = df_cleaned[df_cleaned['therapist_text'].str.len() > 0]
        
        # Save to CSV
        output_file = "arabic_empathetic_conversations_cleaned.csv"
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Successfully cleaned and saved {len(df_cleaned)} unique lines to {output_file}")
        
    except Exception as e:
        print(f"❌ ERROR: Could not process arbml/arabic_empathetic_conversations. {e}")

def main():
    print("--- Starting RAG Dataset Download & Cleaning ---")
    print("This script will create the 'base layer' CSVs for your RAG system.\n")
    
    # Process the English dataset
    process_counsel_chat()
    
    # Process the Arabic dataset
    process_arabic_empathetic_conversations()
    
    print("\n--- Process Complete ---")
    print("You can now add the following files to your GitHub repository:")
    print("1. counsel_chat_cleaned.csv (English)")
    print("2. arabic_empathetic_conversations_cleaned.csv (Arabic)")
    print("3. This script, download_and_clean.py")

if __name__ == "__main__":
    main()