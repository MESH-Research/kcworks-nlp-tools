# Copyright 2025, Mesh Research
#
# kcworks_nlp_tools is free software, released under the MIT license

import os
import pandas as pd
import logging
import psutil
from dotenv import load_dotenv
from .logging_config import set_up_logging
# from memory_profiler import profile


def log_memory_usage(message):
    """
    Logs current memory usage of the system.

    Arg:
        message(str): Contextual message for the log entry.
    """
    memory = psutil.virtual_memory()
    logging.info(
        f"{message} - Memory usage: {memory.percent}% used, "
        f"{memory.available // (1024 * 1024)} MB available"
    )


# TODO: Check variables across the scripts; think about if i keep df everywhere.
def clean_text(text):
    """
    Make sure any NaN values are replaced with string

    Arg:
        text(str): Raw text input.

    Returns:
        str: Cleaned text.
    """
    if pd.isnull(text):
        logging.warning("Encountered NaN value, replacing with empty string.")
        return ""
    return text.strip()


def filter_and_process_batch(df):
    """
    Filter and process batch, only take 'eng' and 'failed = 0' files.

    Arg:
        df (DataFrame): Input DataFrame chunk.

    Returns:
        DataFrame: Processed DataFrame
    """
    # Log memory before processing the batch
    log_memory_usage("Before filtering and processing batch.")

    # Filter for English language and files are processed
    df = df[(df['Languages'] == 'eng') & (df['Failed'] == 0)]
    logging.info(
        f"Filtering English rows and successfully extracted text "
        f"from batch of size {len(df)}."
    )

    # Minimal cleaning
    df['Processed Text'] = df['Extracted Text'].apply(clean_text)

    # Drop the unnecessary column 'Extracted Text'
    columns_to_keep = [
        'Record ID',
        'Languages',
        'DOI',
        'File Name',
        'Processed Text',
        'Failed',
        'Supported'
    ]
    processed_df = df[columns_to_keep]

    # Log memory after processing the batch
    log_memory_usage("After filtering and processing batch.")

    return processed_df


#TODO: maybe thinking about creating another DataFrame that keeps all the files for future use?
def process_csv_in_batches(input_filepath, output_filepath, batch_size=100):
    """
    Process a large CSV file in batches with minimal preprocessing.

    Args:
        input_filepath (str): Path to the input CSV.
        output_filepath (str): Path to save the processed CSV.
        batch_size(int): Number of rows per batch.
    """
    chunk_number = 0
    cumulative_count = 0 # Track total processed rows

    try:
        for chunk in pd.read_csv(input_filepath, chunksize=batch_size):
            chunk_number += 1
            log_memory_usage(f"Before processing batch {chunk_number}")

            processed_chunk = filter_and_process_batch(chunk)
            batch_count = len(processed_chunk)
            cumulative_count += batch_count

            # Write processed data to the output file
            if chunk_number == 1:
                processed_chunk.to_csv(output_filepath, index=False)
            else:
                processed_chunk.to_csv(output_filepath, index=False,
                                       mode='a', header=False)

            logging.info(
                f"Processed {batch_count} rows in batch {chunk_number}. "
                f"Total processed so far: {cumulative_count} rows."
            )
            log_memory_usage(f"After processing batch {chunk_number}.")

        logging.info(
            f"All batches processed successfully. Total rows "
            f"processed: {cumulative_count}."
        )

    except Exception as e:
        logging.error(f"Error processing batch {chunk_number}:{e}.")


def main():
    """Download record files and extract text into csv for NLP uses
    """
    # Load environment variables
    load_dotenv()

    # File paths from .env
    input_filepath = os.getenv("OUTPUT_CSV")
    output_filepath = os.getenv("PREPROCESS_PATH")

    # Set batch size
    batch_size = int(os.getenv("BATCH_SIZE", 100))

    # Log initial memory
    log_memory_usage("Before starting the preprocessing.")

    # Process CSV
    process_csv_in_batches(input_filepath, output_filepath, batch_size)

    # Log final memory
    log_memory_usage("After completing the preprocessing")


if __name__ == "__main__":
    set_up_logging()
    main()