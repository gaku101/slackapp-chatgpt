import logging
import os
import sys
import time

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Pinecone

load_dotenv()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_vectorstore():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-west-2")

    # check for and delete index if already exists
    index_name = os.environ["PINECONE_INDEX"]
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # create a new index
    pc.create_index(index_name, dimension=1536, metric="dotproduct", spec=spec)  # dimensionality of text-embedding-ada-002

    # wait for index to be initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)


if __name__ == "__main__":
    file_path = sys.argv[1]
    loader = UnstructuredPDFLoader(file_path)
    raw_docs = loader.load()
    logger.info("Loaded %d documents", len(raw_docs))

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_docs)
    logger.info("Split %d documents", len(docs))

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)
