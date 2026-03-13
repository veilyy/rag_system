from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/bulgakov.pdf")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "])

all_splits = text_splitter.split_documents(docs)

print(f'Total splits: {len(all_splits)}')