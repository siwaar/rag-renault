"""
Utility functions for file operations and image processing.

"""

import base64
import io
import re
from typing import Any, Dict, Optional
from IPython.display import display
from PIL import Image
from config.logger import logger
from langchain.schema.document import Document



def save_doc_to_file(doc: Document, filename: str) -> None:
    """
    Save a single document to a text file.

    Args:
        doc: LangChain Document containing metadata and content.
        filename: Path to the output file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {doc.metadata.get('title', 'N/A')}\n")
        f.write(f"Source: {doc.metadata.get('source', 'N/A')}\n")
        f.write(f"Year: {doc.metadata.get('year', 'N/A')}\n")
        f.write("Content:\n")
        f.write(doc.page_content)
    logger.info(f"Saved transcript to file: {filename}")


def extract_year(text: str) -> Optional[int]:
    """
    Extract a 4-digit year from the end of a string separated by underscores.

    Args:
        text: The input text string.

    Returns:
        The extracted year as an integer, or None if not found.
    """
    parts = text.split("_")
    if parts and parts[-1].isdigit() and len(parts[-1]) == 4:
        return int(parts[-1])
    return None


def filter_none_metadata(metadata: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """
    Remove keys with None values from metadata.

    Args:
        metadata: Dictionary of metadata.

    Returns:
        Filtered metadata dictionary.
    """
    return {k: v for k, v in metadata.items() if v is not None}


def display_base64_image(base64_code: str) -> None:
    """
    Display an image from its base64 encoded string.

    Args:
        base64_code (str): Base64 encoded string of the image.
    """
    try:
        image_data = base64.b64decode(base64_code)
        display(Image(data=image_data))
    except Exception as e:
        logger.error(f"Error displaying image: {e}")


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False
    

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None
   

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []

    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)

    return {"images": b64_images, "texts": texts}


def parse_docs(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, str) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}