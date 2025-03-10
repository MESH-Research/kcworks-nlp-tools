# Tutorial 1: Text Extraction
This document demonstrates the process of selecting python libraries for text extraction.

## Introduction
With a variety of file types hosted on InvenioRDM, developing a robust strategy for text extraction is essential. This tutorial evaluates and demonstrates Python libraries for extracting text from the following common file types:
- PDF (Text-based and Image-based)
- DOCX
- PPTX
- EPUB
- RTF
- CSV

Each library is evaluated based on:
1. Document Quality
2. Ease of Use
3. Community Support (maintenance; up-to-date)
4. Performance Metrics

### 1. PDF
PDFs (Portable Document Format) can contain either text-based or image-based data. Each type requires a different approach for efficient text extraction.

#### 1.1 For Text-based PDF: PyMuPDF
For **Text-Based PDF**, several libraries are available:
- *PyPDF2* (simple)
- *PDFMiner* (deep learning curve)
- *PyMuPDF(fitz)*, selected for this workflow.
- *Textract*.

After comparing all results, **PyMuPDF(fitz)** was selected for its balance of performance, easy to use, and reliability. However, if things change in the future, if may be worth revisiting alternative libraries.

**Implementation: Extracting Text with PyMuPDF(fitz)**

*Remember to install packages first, `pip install PyMuPDF`.*

Here is a simple example to extract text from a PDF:
```python
import fitz
import timeit

# Replace the path below 'text4test/paper.pdf' with your own file path
def process_pdf():
    with fitz.open('text4test/paper.pdf') as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            print(page.get_text())

# You can do multiple times and then find average.
number_of_executions = 1
# Use timeit to measure the execution time
execution_time = timeit.timeit('process_pdf()', setup='from __main__ import process_pdf', number=number_of_executions)

print(f"Execution time: {execution_time} seconds")
```

#### Main Checkpoints:
- Footnotes: Included
- Forms/Tables: Embedded text is extracted; images of tables are not processed.
- Citations: Included

#### 1.2 For Image-based PDF: PyTesseract
For image-based PDFs, Optical Character Recognition (OCR) is required to extract text. This process involves two steps:
1. Using a PDF processing library like **PyMuPDF(fitz)** to extract images from the PDF.
2. Applying **PyTesseract** for OCR to convert the images into text. **PyTesseract** was chosen due to its active community support and reliable performance.

The following code demonstrates a workflow with key features:
1. A **progress bar** to monitor OCR progress.
2. An **error warning message** to handle pages or images that fail to process. E.g. "An error occured on page X."
3. **Page Indicator** such as "--- End of Page X ---" for clear segmentation of results.
4. Terminal will only show partial text if the file is large (e.g testing file is 11.5M).
5. An **output file** ('pdf-img-extext.txt') for storing extracted text.
6. **Time tracking** for the OCR process.

**Implementation: Extracting Text from Image-Based PDFs**

*Remember to install packages first, `pip install PyMuPDF Pytesseract Pillow`.*

Here’s the Python code for OCR processing:
```python
import fitz
import pytesseract
from PIL import Image
import io
import timeit

# Progress bar to monitor OCR status
def update_progress(progress):
    bar_length = 50
    block = int(round(bar_length * progress))
    text = "\rProgress: [{0}] {1}%".format("#" * block + "-" * (bar_length - block), round(progress * 100, 2))
    print(text, end='')

# OCR function for extracting text
def extract_text_ocr(pdf_path, output_path):
    """Extracts text from an image-based PDF using OCR."""
    text = ''
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            total_images = len(image_list)

            for image_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    text += pytesseract.image_to_string(image)
                except Exception as e:
                    print(f"\nAn error occurred on page {page_num + 1}, image {image_index + 1}: {e}")

                # Update progress bar
                current_progress = ((page_num * total_images + image_index + 1) / (total_pages * total_images))
                update_progress(current_progress)

            text += f"\n----- End of Page {page_num + 1} -----\n"

    # Write the extracted text to the output file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

# Main function for execution
def main():
    pdf_path = 'text4test/paper-img.pdf' # File path, replace it with your file path
    output_path = 'pdf-img-extext.txt' # Get a txt file as the output
    extract_text_ocr(pdf_path, output_path)

if __name__ == "__main__":
    # Time the execution of the main function, with the testing file, it took 54 seconds.
    execution_time = timeit.timeit('main()', setup='from __main__ import main', number=1)
    print(f"\nExecution time: {execution_time:.2f} seconds")
```

#### Main Checkpoints:
- Footnotes: Included
- Forms/Tables: The textual information of forms/tables can be extracted
- Citations: Included

### 2. DOCX
DOCX files are based on the Open XML format, which makes them more efficient, reliable, and less prone to corruption compared to older binary formats like DOC. The “X” in DOCX stands for XML (eXtensible Markup Language), which forms the backbone of this file type. *The file tested in this section contains the same content as the one used in the PDF testing section.

Several Python libraries are available for extracting text from DOCX files:
- *docx-simple*
- *docx2txt*
- *python-docx*, selected for this tutorial.
- *Mammoth*

After comparing all results, **python-docx** was selected for its simplicity. However, if things change in the future, if may be worth revisiting alternative libraries.

**Implementation: Extracting Text from DOCX Files**

#### 2.1 Python-docx Library
*Install packages first, `pip install python-docx`.*

Here’s the Python code for extracting text from a DOCX file using **python-docx**:
```Python
import docx
import timeit

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n" # Append each paragraph with a newline
    return text

if __name__ == "__main__":
    docx_file = "text4test/paper.docx"  # File path, replace it with your file path

    # Measure execution time
    execution_time = timeit.timeit(
        stmt=lambda: extract_text_from_docx(docx_file),
        number=1  # You can change the number of iterations if needed
    )

    extracted_text = extract_text_from_docx(docx_file)

    print("Extracted Text:") # (Optional) Display extracted text
    print(extracted_text)
    print(f"Execution Time: {execution_time:.4f} seconds")
```
#### Main checkpoints:
- Footnotes: Not included
- Forms/Tables: If the table is in embedded, than it's fine; image won't work
- Citations: Included

#### 2.2 Docx2txt Library
The docx2txt library is a lightweight tool for extracting text from DOCX files. While it doesn’t support footnotes, it offers the ability to extract text from embedded images.

*Install packages first, `pip install docx2txt`.*

```Python
import docx2txt
import timeit

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file."""
    text = docx2txt.process(docx_file)
    return text

if __name__ == "__main__":
    docx_file = "text4test/paper.docx"  # File path, replace it with your file path
    output_file = "docx-extext.txt"  # Get a txt file as the output

    # Measure execution time
    execution_time = timeit.timeit(
        stmt=lambda: extract_text_from_docx(docx_file),
        number=1 # Adjust as needed
    )

    extracted_text = extract_text_from_docx(docx_file)

    # Save extracted text to the output file
    with open(output_file, "w", encoding="utf-8") as output:
        output.write(extracted_text)

    print("Extracted Text:") #You don't necessarily need this
    print(extracted_text)
    print(f"Execution Time: {execution_time:.4f} seconds")
    print(f"Extracted text saved to {output_file}")
```
#### Main checkpoints:
- Footnotes: Not included
- Forms/Tables: Embedded tables are processed; images in tables are ignored. It can process images with text.
- Citations: Included

#### 2.3 Mammoth library
The **Mammoth** library is specifically designed to convert DOCX files into clean HTML or plain text while preserving basic formatting. It’s particularly effective for extracting footnotes and inline citations.

*Install packages first, `pip install mammoth beautifulsoup4`.*

```Python
import mammoth
from bs4 import BeautifulSoup
import timeit

def extract_text(docx_file):
    """Extracts text from a DOCX file and converts it to plain text."""
    with open(docx_file, "rb") as docx:
        result = mammoth.convert_to_html(docx)
        html_content = result.value  # Extracted HTML
        messages = result.messages # You can print any processing messages lateron

    soup = BeautifulSoup(html_content, "html.parser") # Use bs to parse HTML content
    text = soup.get_text(separator='\n', strip=True) # Stripping extra whitespace
    return text

if __name__ == "__main__":
    docx_file = "text4test/paper.docx"  # Replace with your file path

    # Measure execution time
    execution_time = timeit.timeit(
        stmt=lambda: extract_text(docx_file),
        number=1  # Number of iterations; adjust as needed
    )

    print("\nExtracted Text:")
    print(extract_text(docx_file))
    print(f"Execution time: {execution_time} seconds")
```

#### Main checkpoints:
- Footnotes: Included
- Forms/Tables: Embedded text is extracted; images in tables are ignored.
- Citations: Included

### 3. PPTX
PPTX (PowerPoint) files are widely used for presentations and often contain a mix of text, images, captions, speaker notes, tables, and references. Extracting meaningful content from these files requires handling their diverse elements effectively.

**Implementation: Extracting Text from PPTX Files**

*Install packages first, `pip install python-pptx tersseract`.*

```python
from pptx import Presentation
import timeit
import io
from PIL import Image
import pytesseract

def extract_content_from_pptx(pptx_file):
    """Extracts text, speaker notes, and image text from a PPTX file."""
    prs = Presentation(pptx_file)
    extracted_content = []
    image_texts = []

    for slide in prs.slides:
        # Extracting text from each shape on the slide
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    extracted_content.append(paragraph.text.strip())

            # Extracting text from images using OCR        
            elif shape.shape_type == 13:  # Shape type 13 corresponds to a picture
                try:
                    image = shape.image
                    image_bytes = io.BytesIO(image.blob)
                    image_text = pytesseract.image_to_string(Image.open(image_bytes))
                    image_texts.append(image_text.strip())
                except Exception as e:
                    print(f"Error processing image: {e}")

            # Extracting text from tables
            if shape.shape_type == 19:  # Shape type 19 corresponds to a table
                for row in shape.table.rows:
                    for cell in row.cells:
                        extracted_content.append(cell.text.strip())

        # Extracting speaker notes
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            if notes_slide.notes_text_frame:
                for paragraph in notes_slide.notes_text_frame.paragraphs:
                    extracted_content.append(paragraph.text.strip())

    return '\n'.join(extracted_content), '\n'.join(image_texts)

if __name__ == "__main__":
    pptx_file = "text4test/talk.pptx"  # Replace with your file path

    # Measure execution time
    execution_time = timeit.timeit(
        stmt=lambda: extract_content_from_pptx(pptx_file),
        number=1  # Number of iterations; adjust if needed
    )
    
    # Extract and print content
    extracted_content, image_texts = extract_content_from_pptx(pptx_file)
    
    print("\nExtracted Content:")
    print(extracted_content)
    print("\nText from Images:")
    print(image_texts)
    print(f"Execution time: {execution_time} seconds")
    # Using the testing file, around 8-9s.
```

#### Main checkpoints:
- Speakers note: Extracted and listed alongside the slide content.
- Images: Text within images is processed using PyTesseract..
- Tables: Extracted successfully, with all textual content preserved.
- Special characters: Captured accurately; equations in images may require further verification.
- References (last page): Clear and formatted correctly.

### 4. EPUB
EPUB, short for Electronic Publication, is a widely used e-book file format with the “.epub” extension. It is designed to store text and multimedia content in a compressed and standardized manner, making it suitable for a variety of e-reading platforms.

Among the available Python libraries for processing EPUB files, **EbookLib** stands out for its ability to handle both text and images efficiently.

**Implementation: Extracting Text from EPUB**
*Install packages first, `pip install ebooklib beautifulsoup4`.*

```Python
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def extracted_text_epub(file_path):
  book = epub.read_epub(file_path)
  test=''

  for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
      soup= BeautifulSoup(item.content,'html.praser')
      text+=soup.get_text()+'\n' # Extracting and formatting text

  return text

if __name__ == "__main__":
    epub_file_path = "text4test/paper.epub"  # Replace with your file path

    # Extract text from the EPUB file
    extracted_text = extract_text_from_epub(epub_file_path)

    # Print the extracted text
    print("Extracted Text:")
    print(extracted_text)
```
#### Main checkpoints:
- Footnotes: Included and rendered clearly.
- Forms/Tables: Extracts textual information from embedded tables; image-based tables or visualizations are ignored.
- Citations: Clear and good.

### 5. RTF
RTF (Rich Text Format), is a file format that facilitates the exchange of text documents between different word processors and operating systems. It supports a range of formatting options but can sometimes be challenging to process programmatically due to its complexity.

Two commonly used libraries for handling RTF files are **Pyth RTF** and **striprtf**. However, both libraries have significant limitations, such as outdated maintenance and difficulties in converting certain RTF features into readable formats. Attempts to convert .rtf files to .xml and extract text were unsuccessful due to the lack of support and compatibility issues.

To process RTF files effectively, the recommended approach is to first convert them to the DOCX format using LibreOffice. Once converted, the DOCX files can be processed using the methods outlined in the DOCX section.

Install LibreOffice and use the following command to convert an RTF file to DOCX format:
```console
libreoffice --convert-to docx Paperfortest1.rtf --headless
```

### 6. CSV
A CSV (Comma-Separated Values) file is a plain text file format that uses a specific structure to save data in a tabular form, with each line representing a data record and fields separated by commas. CSV, which stands for Comma-Separated Values, is widely used for datasets due to its simplicity and compatibility across platforms.

**Notes**:
Based on testing with various CSV files, it is recommended to handle CSV files separately. Unlike textual files, CSV files are often shared as datasets, sometimes accompanying published papers. The decision to include CSV files in a workflow should depend on the research goals and the nature of the project.

The Python library `pandas` provides robust tools for handling CSV files, making it an ideal choice for this purpose.

*Install packages first, `pip install pandas`.*

```Python
import pandas as pd

# mental_health.csv has a very simple data structure; instrument_export.csv has more columns. This comparison is to show each csv file is very different and should be handled with more human judgement.

file_path='text4test/mental_health.csv'
# Or file_path='text4test/instrument_export.csv' or your file

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

print("Column names",df.columns.tolist())

# Extract specific columns
columns_to_extract = ['selectedColumn1','selectedColumn2']
textual_data = df[columns_to_extract]

# Show the first few rows of the extracted data
print(textual_data.head())
```

## Conclusion
This tutorial tested and showcased the strengths and limitations of Python libraries for text extraction across various file formats. These libraries are tested, evaluated, and some of them are later used in stage 2 to perform data extraction.