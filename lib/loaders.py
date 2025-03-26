"""
    Loaders for processing local PDF, TXT, and YouTube transcript data.

"""

from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import YoutubeLoader as LCYoutubeLoader, PyPDFLoader, TextLoader
from utils import extract_year
from config.cache_manager import load_from_cache, save_to_cache
import os


class BaseLoader:
    """
    Abstract base loader that defines a common interface for batch loading documents.
    """
    def __init__(self, urls):
        self.urls = urls

    def load(self):
        """
        Load all documents concurrently using ThreadPoolExecutor.

        Returns:
            list: List of loaded document objects.
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._load_single, self.urls))
        return [doc for result in results if result for doc in result]

    def _load_single(self, url):
        raise NotImplementedError


class LocalPDFLoader(BaseLoader):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def _load_single(self, file_path):
        cached_data = load_from_cache(file_path)
        if cached_data:
            print(f"Loaded local PDF from cache: {os.path.basename(file_path)}")
            return cached_data

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            year = extract_year(file_path)
            for doc in docs:
                doc.metadata.update({
                    "source": file_path,
                    "title": os.path.basename(file_path),
                    "year": year
                })
            print(f"Loaded local PDF: {os.path.basename(file_path)}")
            save_to_cache(file_path, docs)
            return docs
        except Exception as e:
            print(f" Failed to load local PDF '{file_path}': {e}")
            return None


class LocalTextLoader(BaseLoader):
    def __init__(self, file_paths):
        super().__init__(file_paths)

    def _load_single(self, file_path):
        cached_data = load_from_cache(file_path)
        if cached_data:
            print(f"Loaded local TXT from cache: {os.path.basename(file_path)}")
            return cached_data

        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            # Extract metadata from the file content
            lines = docs[0].page_content.strip().split("\n")[:3]
            title = os.path.basename(file_path)
            source = None
            year = None
            for line in lines[:3]:  # Check the first 3 lines for metadata
                if line.startswith("Title: "):
                    title = line.replace("Title: ", "").strip()
                elif line.startswith("Source: "):
                    source = line.replace("Source: ", "").strip()
                elif line.startswith("Year: "):
                    year = line.replace("Year: ", "").strip()
            for doc in docs:
                doc.metadata.update({
                    "source": source,
                    "title": title ,
                    "year": year
                })
            print(f"Loaded local TXT: {os.path.basename(file_path)}")
            save_to_cache(file_path, docs)
            return docs
        except Exception as e:
            print(f"Failed to load local TXT '{file_path}': {e}")
            return None


class YouTubeLoader(BaseLoader):
    def _load_single(self, item):
        title, url = item
        cached_data = load_from_cache(url)
        if cached_data:
            print(f"Loaded YouTube transcript from cache: {title}")
            return cached_data

        try:
            loader = LCYoutubeLoader.from_youtube_url(
                youtube_url=url,
                language=["fr", "en"],
                translation="fr"
            )
            docs = loader.load()
            year = extract_year(title) 
            for doc in docs:
                doc.metadata.update({
                    "source": url,
                    "title": title,
                    "year": year
                })
            print(f"Loaded transcript for: {title}")
            save_to_cache(url, docs)
            return docs
        except Exception as e:
            print(f"Failed to load transcript for '{title}': {e}")
            return None
