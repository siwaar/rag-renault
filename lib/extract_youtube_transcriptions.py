"""
    Loads and processes YouTube transcripts using LangChain's YoutubeLoader.
"""

import os

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple
from langchain_community.document_loaders import YoutubeLoader
from config.logger import logger
from config.settings import TRANSCRIPTS_DIR, YOUTUBE_URLS
from langchain_core.documents import Document

from config.cache_manager import load_from_cache, save_to_cache
from utils import extract_year, save_doc_to_file, filter_none_metadata



class BaseLoader:
    """
    Base class for loading documents from various sources.
    """

    def __init__(self, urls: Dict[str, str]) -> None:
        self.urls = urls

    def load(self) -> List[Document]:
        """
        Load documents concurrently for all URLs.
        
        Returns:
            A list of LangChain Documents.
        """
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self._load_single, self.urls.items()))
        return [doc for result in results if result for doc in result]

    def _load_single(self, item: Tuple[str, str]) -> Optional[List[Document]]:
        """
        Abstract method to load a single document. Should be implemented by subclasses.
        """
        raise NotImplementedError


class CustomYouTubeLoader(BaseLoader):
    """
    Custom loader for YouTube transcripts using LangChain's YoutubeLoader.
    """

    def _load_single(self, item: Tuple[str, str]) -> Optional[List[Document]]:
        """
        Load a transcript from YouTube or cache.
        
        Args:
            item: Tuple containing title and URL.
        
        Returns:
            A list of LangChain Documents, or None if loading fails.
        """
        title, url = item
        cached_data = load_from_cache(url)
        if cached_data:
            logger.info(f"Loaded YouTube transcript from cache: {title}")
            return cached_data

        try:
            logger.info(f"Fetching YouTube transcript: {title}")
            loader = YoutubeLoader.from_youtube_url(
                youtube_url=url,
                language=["fr", "en"],
                translation="fr"
            )
            docs = loader.load()
            year = extract_year(title) 
            for doc in docs:
                doc.metadata.update(filter_none_metadata({
                    "source": url,
                    "title": title,
                    "year": year
                }))
            logger.info(f"Successfully loaded transcript for: {title}")
            save_to_cache(url, docs)
            return docs
        except Exception as e:
            logger.error(f"Failed to load transcript for '{title}': {e}", exc_info=True)
            return None


def save_transcripts(all_docs: List[Document]) -> None:
    """
    Save all documents to text files in the specified transcripts directory.
    
    Args:
        all_docs: List of LangChain Documents.
    """
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    for doc in all_docs:
        title = doc.metadata.get('title', 'untitled')
        filename = f"{title}.txt"
        file_path = os.path.join(TRANSCRIPTS_DIR, filename)
        save_doc_to_file(doc, file_path)


if __name__ == "__main__":
    logger.info("Starting YouTube transcript loading process")

    # Load YouTube transcripts
    youtube_loader = CustomYouTubeLoader(YOUTUBE_URLS)
    all_docs = youtube_loader.load()

    # Save each transcript to a separate file
    logger.info("Saving transcripts to individual files")
    save_transcripts(all_docs)

    logger.info("YouTube transcript loading process completed")