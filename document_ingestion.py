import nltk
import chromadb
from constants import DOC_DIR_PATH, VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

nltk.download("punkt_tab")
nltk.download("stopwords")

# Initailize directory loader
loader = SimpleDirectoryReader(
    input_dir=DOC_DIR_PATH
)

# load documents
documents = loader.load_data()

# Initialize node parser
parser = SimpleNodeParser.from_defaults(
    chunk_size=1024,
    chunk_overlap=50
)

# convert documents into nodes
nodes = parser.get_nodes_from_documents(
    documents=documents
)

# define persistant db location
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# create or retrieve the vector collection
chroma_collection = db.get_or_create_collection(
    name=COLLECTION_NAME
)

# Load embedding model
embedding = HuggingFaceEmbedding()

# create vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)

# create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create vector store index
index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embedding,
    storage_context=storage_context,
    vector_store=vector_store
)

print("Vector database created!")
