""" 
    Script to launch to update vectorstore and docstore
"""

import uuid
from typing import List

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import (
    COLLECTION_NAME,
    CONNECTION_STRING,
    DATA_EXTRACTED_PATH,
    ID_KEY,
    LOCAL_FILES,
    YOUTUBE_URLS,
)
from store import PostgresByteStore
from chunker import TextChunker
from get_unstructured_data_descriptions import generate_unstructured_data_descriptions
from extract_youtube_transcriptions import CustomYouTubeLoader
from loaders import LocalPDFLoader
from config.logger import logger


def get_retriever() -> MultiVectorRetriever:
    """
    Initialize and return a MultiVectorRetriever.

    Returns:
        MultiVectorRetriever: An instance of MultiVectorRetriever configured with
        PGVector for vector storage and PostgresByteStore for document storage.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Initializing MultiVectorRetriever")
    embeddings = embedding_model
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    store = PostgresByteStore(CONNECTION_STRING, COLLECTION_NAME)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=ID_KEY,
    )
    logger.info("MultiVectorRetriever initialized successfully")
    return retriever


def load_all_documents() -> List[Document]:
    """
    Load documents from local PDF files and YouTube transcripts.

    Returns:
        List[Document]: A list of Document objects containing the loaded content.
    """
    docs = []
    logger.info("Starting to load documents")
    docs.extend(LocalPDFLoader(LOCAL_FILES[3:4]).load())
    docs.extend(CustomYouTubeLoader(YOUTUBE_URLS).load())
    return docs


def process_documents(docs: List[Document], retriever: MultiVectorRetriever) -> None:
    """
    Process a list of documents by splitting them into chunks and adding them to the retriever.

    Args:
        docs (List[Document]): The list of documents to process.
        retriever (MultiVectorRetriever): The retriever to add the processed documents to.
    """
    logger.info(f"Processing {len(docs)} documents")
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    # Split text into chunks
    splitter = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(docs, doc_ids)
    logger.info(f"Split documents into {len(chunks)} chunks")

    logger.info("Adding chunks to vectorstore")
    retriever.vectorstore.add_documents(chunks)
    
    logger.info("Updating docstore")
    items = [
        (doc_id, doc, doc.metadata.get("source", "unknown_file"))
        for doc_id, doc in zip(doc_ids, docs)
    ]
    retriever.docstore.mset(items)
    logger.info("Document processing completed")


def process_images(retriever: MultiVectorRetriever) -> None:
    """
    Process images by generating descriptions and adding them to the retriever.

    Args:
        retriever (MultiVectorRetriever): The retriever to add the processed images to.
    """
    logger.info("Starting image processing")
    encoded_images, img_descriptions = generate_unstructured_data_descriptions(DATA_EXTRACTED_PATH)
    logger.info(f"Generated descriptions for {len(encoded_images)} images")

    img_ids = [str(uuid.uuid4()) for _ in encoded_images]

    logger.info("Adding image summaries to vectorstore")
    summary_img = [
        Document(page_content=summary, metadata={ID_KEY: img_id})
        for img_id, summary in zip(img_ids, img_descriptions)
    ]
    retriever.vectorstore.add_documents(summary_img)

    logger.info("Adding images to docstore")
    items = [
        (img_id, img, filename)
        for img_id, (filename, img) in zip(img_ids, encoded_images.items())
    ]
    retriever.docstore.mset(items)
    logger.info("Image processing completed")


def main():
    """
    Main function to orchestrate the document and image processing workflow.
    """
    logger.info("Starting main workflow")
    retriever = get_retriever()
    docs = load_all_documents()
    process_documents(docs, retriever)
    process_images(retriever)
    logger.info("Main workflow completed")


if __name__ == "__main__":
    main()
    logger.info("Script completed successfully")