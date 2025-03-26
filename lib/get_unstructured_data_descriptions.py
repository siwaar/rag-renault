import os
import time
import base64
from typing import Dict, Tuple, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.cache_manager import load_from_cache, save_to_cache
from config.settings import vision_model
from config import logger


def encode_image(image_path: str) -> str:
    """
    Encode an image file to a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""


def encode_all_images(base_path: str) -> Dict[str, str]:
    """
    Encode all images in a directory whose filenames start with 'table'.

    Args:
        base_path (str): Path to the base directory.

    Returns:
        dict: Mapping of file paths to base64-encoded image strings.
    """
    encoded_images = {}
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().startswith("table"):
                file_path = os.path.join(root, file)
                encoded = encode_image(file_path)
                if encoded:
                    encoded_images[file_path] = encoded
    return encoded_images


def build_vision_chain():
    """
    Build and return a LangChain pipeline for describing images or tables.

    Returns:
        LangChain chain: A pipeline consisting of prompt, vision model, and output parser.
    """
    system_prompt = (
        "You are an assistant tasked with describing images or tables for retrieval. "
        "These descriptions will be embedded and used to retrieve the raw image or table. "
        "Give a concise description of the image or table with different information that are well optimized for retrieval."
    )

    messages = [
        (
            "user",
            [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt | vision_model | StrOutputParser()


def get_image_description_single(base64_image: str, chain) -> str:
    """
    Generate a single description for a base64-encoded image.

    Args:
        base64_image (str): Base64 string of the image.
        chain: LangChain chain to generate the description.

    Returns:
        str: Generated description.
    """
    return chain.invoke({"image": base64_image})


def generate_unstructured_data_descriptions(
    path: str,
    sleep_seconds: int = 2
) -> Tuple[Dict[str, str], List[str]]:
    """
    Generate descriptions and base64 strings for images in a folder, with optional caching.

    Args:
        path (str): Path to the folder containing images.
        sleep_seconds (int): Seconds to sleep between API calls (default: 2).

    Returns:
        tuple: (Dict of base64-encoded images, List of corresponding descriptions)
    """
    encoded_images = encode_all_images(path)
    cached_data = load_from_cache(path)

    if cached_data:
        logger.info(f"Loaded image descriptions from cache: {os.path.basename(path)}")
        return encoded_images, cached_data

    chain = build_vision_chain()
    descriptions = []

    for image_path, base64_image in encoded_images.items():
        logger.info(f"Describing image: {os.path.basename(image_path)}")
        description = get_image_description_single(base64_image, chain)
        descriptions.append(description)
        time.sleep(sleep_seconds)

    save_to_cache(path, descriptions)
    return encoded_images, descriptions
