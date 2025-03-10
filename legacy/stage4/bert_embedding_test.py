import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import logging
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bert_embedding_test.log", mode="w"),
        logging.StreamHandler()
    ]
)

def load_data(input_filepath, subset_size=100):
    """
    Load a subset of 100 files of the processed, deduplicated data for testing.

    Args:
        input_filepath(str): Path to the input CSV file, in this case, deduplicated_output.csv from stage3
        subset_size (int): numbers of rows to load, 100 for now
    
    Returns:
        DataFrame: A subset of the processed, deduplicated data.
    """
    try:
        logging.info(f"Loading data from {input_filepath}")
        df = pd.read_csv(input_filepath)

        # Check if 'Processed Text' column exists
        if 'Processed Text' not in df.columns:
            raise ValueError("Column 'Processed Text' is missing from the input file.")

        subset_df = df.head(subset_size)
        logging.info(f"Loaded {len(subset_df)} rows for testing.")
        return subset_df
    except Exception as e:
        logging.error(f"Error loading data {e}.")
        raise

def generate_embeddings(text_list, tokenizer, model, batch_size=8):
    """
    Generate BERT embeddings for a list of texts.
    Args:
        text_list (list): List of input texts
        tokenizer: BERT tokenizer
        model: Pretrained BERT model.
        batch_size (int): Batch sizing for processing, 8 for now. 

    Returns:
        list: List of embeddings.    
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model.to(device)

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} with {len(batch_texts)} texts.")

        try:
            # Tokenization
            # 512 is a hard limit on max length for BERT
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512) 
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

            with torch.no_grad():
                outputs = model(**encoded_input)

            # Extract the [CLS] token embeddings (first token for each sequence)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)

        except Exception as e:
            logging.error(f"Error processing batch {i // batch_size +1}: {e}")
            continue

    logging.info(f"Generated embeddings for all texts.")
    return embeddings

def main():
    load_dotenv()

    # File paths
    input_filepath = os.getenv("DEDUPLICATED_OUTPUT_PATH", "deduplicated_output.csv")
    output_filepath = os.getenv("BERT_EMBED_SUBSET_OUTPUT_PATH", "bert_embeddings_subset.csv")

    # Load a subset of the data
    data_subset = load_data(input_filepath, subset_size=100)
    # Replace NaN with empty strings even I have done it before
    texts = data_subset['Processed Text'].fillna("").tolist()

    # This is where the memory gets used more bc of the tolist!

    # Load pre-trained BERT model and tokenizer
    logging.info("Loading pre-trained BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    logging.info("BERT model and tokenizer loaded successfully.")

    # Generate embeddings
    embeddings = generate_embeddings(texts, tokenizer, model)

    # Save embeddings to a new DF
    embedding_columns = [f"embedding_{i}" for i in range(len(embeddings[0]))]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
    output_df = pd.concat([data_subset[['Record ID', 'DOI', 'File Name', 'Processed Text']], embedding_df], axis=1)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Embeddings saved to {output_filepath}")

if __name__ == "__main__":
    main()