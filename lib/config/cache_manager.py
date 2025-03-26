import os
import hashlib
import pickle
from typing import List, Optional, Any

from config.settings import CACHE_DIR
from config.logger import logger


def get_cache_path(url: str) -> str:
    """
    Generate a unique cache file path for a given URL.

    Args:
        url: The YouTube video URL.

    Returns:
        Path to the cache file.
    """
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + ".pickle")


def load_from_cache(url: str) -> Optional[List[Any]]:
    """
    Load transcript data from cache if available.

    Args:
        url: The YouTube video URL.

    Returns:
        Cached list of documents, or None if cache does not exist.
    """
    cache_path = get_cache_path(url)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            logger.info(f"Loading from cache: {url}")
            return pickle.load(f)
    return None


def save_to_cache(url: str, data: List[Any]) -> None:
    """
    Save transcript data to cache.

    Args:
        url: The YouTube video URL.
        data: List of LangChain Documents.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(url)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Saved to cache: {url}")