import os
import chromadb
from dotenv import load_dotenv
from constants import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.llms.groq import Groq
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# Load env
load_dotenv()

# Initialize llm
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
)

# Load embedding model
embedding = HuggingFaceEmbedding()

# create the persistent client for db
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# Load collection
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)

# connect to vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embedding
)

# create the query engine
query_engine = index.as_query_engine(
    similarity_top_k=3
)

# Send user query and display response
u_query = "What does document say about deductive reasoning?"
response = query_engine.query(u_query)
print(response.response)
