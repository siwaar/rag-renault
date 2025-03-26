# RAG on Renault Data
The goal is to develop an LLM agent(s) that can respond to queries using structured and
unstructured multi-modal data from popular websites and APIs. The agent's knowledge base
will include information on Renault's Renaulution strategy plan, based on our CEO Luca di
Meo's talks, Renault's recent annual reports, Renault's stock prices for the current year, and
the overall performance of the CAC40.


## Why PostgreSQL?

Using PostgreSQL with pgvectorscale as your vector database offers several key advantages over dedicated vector databases:

- PostgreSQL is a robust, open-source database with a rich ecosystem of tools, drivers, and connectors. This ensures transparency, community support, and continuous improvements.

- By using PostgreSQL, you can manage both your relational and vector data within a single database. This reduces operational complexity, as there's no need to maintain and synchronize multiple databases.

- Pgvectorscale enhances pgvector with faster search capabilities, higher recall, and efficient time-based filtering. It leverages advanced indexing techniques, such as the DiskANN-inspired index, to significantly speed up Approximate Nearest Neighbor (ANN) searches.

Pgvectorscale Vector builds on top of [pgvector](https://github.com/pgvector/pgvector), offering improved performance and additional features, making PostgreSQL a powerful and versatile choice for AI applications.

## Prerequisites
- Poetry
- Docker
- Python 3.7+
- PostgreSQL GUI client

## Steps

1. Set up Docker environment
2. Connect to the database using a PostgreSQL GUI client
3. Create a Python script to insert document chunks as vectors
4. Create a Python function to perform similarity search

## Detailed Instructions


!sudo apt-get install poppler-utils
!sudo apt-get install libleptonica-dev tesseract-ocr libtesseract-dev python3-pil tesseract-ocr-eng tesseract-ocr-script-latn



### Set up Docker environment

Using the`docker-compose.yml` file, run the Docker container:

```
docker-compose -f docker-compose.db.yml up -d

```
Verify the Container ankane/pgvector image is Running , :
```
docker ps
```
To manage and inspect the content of your PostgreSQL database tables effectively,
installing a graphical database management tool like pgAdmin is recommended.

