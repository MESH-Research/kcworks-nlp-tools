import pandas as pd
import os
import sys
import csv
from dotenv import load_dotenv

# Ensure large CSV fields can be processed
csv.field_size_limit(sys.maxsize)

# cd path/to/your/folder
# Run this in terminal `python examine_output9.py`

def examine_csv(file_path, output_report):
    """
    Analyze a CSV file and generate a detailed report as another CSV file.

    Args:
        file_path(str): path to the csv file to examine.
        output_report(str): path to the report CSV file.
    """

    try:
        # Load the CSV
        print(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)

        # Create a report DataFrame
        report_data = []

        # Basic overview
        print(f"Analyzing basic statistics.")
        num_rows = len(df)
        column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing_values = df.isnull().sum().to_dict()

        # Content inspection
        print(f"Inspecting content.")
        df['Text Length'] = df['Extracted Text'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
        avg_length = df['Text Length'].mean()
        median_length = df['Text Length'].median()
        min_length = df ['Text Length'].min()
        max_length = df['Text Length'].max()

        # Language analysis
        print(f"Analyzing language.")
        language_counts = df['Languages'].value_counts().to_dict()

        # Failed documents analysis
        print("Analyzing 'Failed' column.")
        failed_counts = df['Failed'].value_counts().to_dict() if 'Failed' in df.columns else {}

        # Supported documents analysis
        print("Analyzing 'Supported' column.")
        supported_counts = df['Supported'].value_counts().to_dict() if 'Supported' in df.columns else {}

        # Problematic rows
        print("Identifying problematic rows.")
        empty_text_rows_count = len(df[df['Extracted Text'].isnull()])
        empty_language_rows_count = len(df[df['Languages'].isnull()])

        # Append basic statistics to report
        report_data.append(['Total Rows', num_rows])
        report_data.append(['Columns Info', column_info])
        report_data.append(['Missing Values', missing_values])

        # Append text length analysis to report
        report_data.append(['Average Text Length', avg_length])
        report_data.append(['Median Text Length', median_length])
        report_data.append(['Minimum Text Length', min_length])
        report_data.append(['Maximum Text Length', max_length])

        # Append language analysis to report
        for lang, count in language_counts.items():
            report_data.append([f"Language: {lang}", count])

        # Append failed documents analysis to report:
        for flag, count in failed_counts.items():
            report_data.append([f"Failed: {flag}", count])

        # Append supported documents analysis to report:
        for flag, count in supported_counts.items():
            report_data.append([f"Supported: {flag}", count])

        # Append problematic rows info
        report_data.append(['Empty Text Rows', empty_text_rows_count])
        report_data.append(['Empty Language Rows', empty_language_rows_count])

        # Save the report to a CSV File
        print(f"Writing analysis report to {output_report}")
        with open(output_report, 'w', newline='', encoding='utf-8') as report_file:
            writer = csv.writer(report_file)
            writer.writerow(['Metric', 'Value'])
            writer.writerows(report_data)

        print("Report generated successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Load from .env file
    load_dotenv()

    # Get CSV path from environment variables
    csv_file_path = os.getenv("OUTPUT_CSV")

    # Hard coded output path
    report_output_path = "stage2/output9_analysis_report.csv"
    examine_csv(csv_file_path, report_output_path)

if __name__ == "__main__":
    main()
