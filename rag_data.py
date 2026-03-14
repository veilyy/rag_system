from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Загрузка 
loader = PyPDFLoader("data/Hameleon.pdf")
docs = loader.load()
print(f"Загружено страниц: {len(docs)}")

# 2. Разбивка на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900, 
    chunk_overlap=150, 
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " "]
)
all_splits = text_splitter.split_documents(docs)
print(f' чанков: {len(all_splits)}')

# Эмбеддинг
passage_embeddings = HuggingFaceEmbeddings(
    model_name="deepvk/USER2-small",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={
        'normalize_embeddings': True,
        'prompt_name': 'search_document'
    }
)

# Создание вб
vector_store = Chroma(
    embedding_function=passage_embeddings,
    collection_name='book',
    persist_directory='data/chroma')

ids = vector_store.add_documents(all_splits)
print(f" Документы добавлены в базу: {len(ids)}")