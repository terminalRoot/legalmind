import os
os.environ['TRANSFORMERS_CACHE'] = './huggingface'
os.environ['HF_HOME'] = './huggingface'

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

model_id = "sentence-transformers/msmarco-MiniLM-L-12-v3"
print(f'loading model {model_id}')
embeddings = SentenceTransformerEmbeddings(model_name=model_id)

print(f"preparing directory documents")
persist_directory = "vector_data"   
loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

print(f"splitting documents to text")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
# sentence-transformers/msmarco-MiniLM-L-12-v3
print(f"saving documents to {persist_directory}")
db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

