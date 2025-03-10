# KCWorks NLP Tools

v0.1.0-alpha1

`kcworks-nlp-tools` is a an experimental package of NLP applications for Knowledge Commons Works.

## License

`kcworks-nlp-tools` is released as free software under the MIT license.

## Configuration

Configuration variables are set in the `config.py` file. Available config values are:

| Variable name           | type | description                                                              |
|-------------------------|------|--------------------------------------------------------------------------|
| DOWNLOADED_FILES_PATH   | str  | Path where downloaded files will be stored                               |
| OUTPUT_FILES_PATH       | str  | Path where output files will be saved                                    |
| API_ENDPOINT            | str  | API endpoint for records ("records")                                     |
| API_URL                 | str  | Base URL for the KC Works API (defaults to "https://works.hcommons.org") |
| BATCH_SIZE              | int  | Number of records to process in each api request (defaults to 10)        |
| CORPUS_SIZE             | int  | Total number of records to process (defaults to 100)                     |
| CHUNK_SIZE              | int  | Size of text chunks for processing (defaults to 400)                     |
| EXTRACTED_TEXT_CSV_PATH | Path | Path to CSV file containing extracted text                               |
| KCWORKS_API_KEY         | str  | API key for KC Works authentication                                      |
| PREPROCESSED_PATH       | Path | Path to CSV file containing preprocessed text                            |
| TIKA_SERVER_ENDPOINT    | str  | URL for Tika server (defaults to "http://localhost:9998")                |


## Required environment variables

A few required environment variables must be provided in a `.env` file placed at the top level of this project folder (i.e., the same folder that contains this README file). These variables must include:

| Variable name   | Description                             |
|-----------------|-----------------------------------------|
| KCWORKS_API_KEY | A valid oauth token for the KCWorks api |

## Acknowledgements
Initial work on this package was done by Tianyi (Titi) Kou-Herrema as a graduate assistant for Knowledge Commons with supervision by Ian Scott and Stephanie E. Vasko.
