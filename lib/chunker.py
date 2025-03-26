"""
    Module for text chunking
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import ID_KEY


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the text splitter with specified chunk size and overlap.
        
        Args:
            chunk_size (int): Size of each text chunk.
            chunk_overlap (int): Overlap between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: List[Document], doc_ids: List[str]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks and tags each chunk with the corresponding document ID.

        Args:
            documents (List[Document]): A list of LangChain Document objects to split.
            doc_ids (List[str]): A list of unique identifiers corresponding to each document.

        Returns:
            List[Document]: A list of smaller Document chunks with metadata including the original document ID.
        """
        chunks = []
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i]
            sub_docs = self.text_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                sub_doc.metadata[ID_KEY] = doc_id
            chunks.extend(sub_docs)
        return chunks