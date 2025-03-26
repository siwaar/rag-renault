"""
    Extracts images and tables from PDF files in the specified local directory
    and saves the results to a designated output path.
"""

import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from config.settings import LOCAL_FILES, DATA_EXTRACTED_PATH

# [Optional] You may need these lines if you are using Windows
# os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Tesseract-OCR'
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def extract_images_and_tables(file_path: str, output_path: str) -> None:
    """
    Extract images and tables from a PDF using high-resolution strategy.

    Args:
        file_path (str): Path to the input PDF file.
        output_path (str): Directory where extracted images and tables will be saved.
    """
    partition_pdf(
        filename=file_path,
        infer_table_structure=True,  # extract tables
        strategy="hi_res",  # mandatory to extract tables
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_images_in_pdf=True,
        extract_image_block_output_dir=output_path,
    )
    return None


if __name__ == "__main__":
    for file_path in LOCAL_FILES:
        output_path = os.path.join(DATA_EXTRACTED_PATH, Path(file_path).stem)
        os.makedirs(output_path, exist_ok=True)
        extract_images_and_tables(file_path, output_path)
