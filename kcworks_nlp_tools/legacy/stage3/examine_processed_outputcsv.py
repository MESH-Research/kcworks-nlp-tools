import pandas as pd
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(filename="stage3deduplication.log", level=logging.INFO, filemode="w", format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def main():
    try:
        # Load CSV
        logging.info("Loading prcessed_output.csv.")
        load_dotenv()
        input_filepath = os.getenv("PREPROCESS_PATH", "processed_output.csv")
        output_filepath = os.getenv("DEDUPLICATED_OUTPUT_PATH", "deduplicated_output.csv")
        processed_df = pd.read_csv(input_filepath)

        # Since there are NaN in DOI, I had to check and make sure they both exist
        required_columns = ['Record ID', 'DOI']
        for column in required_columns:
            if column not in processed_df.columns:
                logging.error(f"Column '{column}' is missing in the dataset.")
                raise ValueError(f"Column '{column} is missing in the dataset.")

        # Filter out rows with NaN in 'Record ID' or 'DOI' to avoid issues during deduplication
        logging.info("Filtering rows with NaN values in 'Record ID' or 'DOI'.")
        filtered_df = processed_df.dropna(subset=required_columns)
        logging.info(f"{len(processed_df) - len(filtered_df)} rows dropped due to NaN values.")

        # Identify duplicates based on both 'Record ID' and 'DOI'
        logging.info("Identifying duplicates based on 'Record ID' and 'DOI'.")
        duplicates = filtered_df[filtered_df.duplicated(subset=required_columns, keep=False)]
        logging.info(f"Found {len(duplicates)} duplicate rows.")

        # Show duplicates
        if not duplicates.empty:
            duplicates_output = "duplicates_review.csv"
            duplicates.to_csv(duplicates_output, index=False)
            logging.info(f"All duplicate rows have been saved to '{duplicates_output}' for review.")

        #TODO: thinking about deduplication
        #TODO: thinking about only taking the newest version of the deposit when taking data from Invenio in the first place.

        # Drop duplicates, keeping the first one
        logging.info("Dropping duplicate rows, keeping the first one.")
        deduplicated_df = filtered_df.drop_duplicates(subset=required_columns, keep='first')
        logging.info(f"Dataset after duplication: {len(deduplicated_df)} rows remain.")

        # Save the deduplicated DataFrame
        logging.info(f"Saving deduplicated dataset to {output_filepath}.")
        deduplicated_df.to_csv(output_filepath, index=False)
        logging.info("Deduplicated dataset saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
