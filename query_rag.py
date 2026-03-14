from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

embeddings = HuggingFaceEmbeddings(
    model_name="deepvk/USER2-small",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={
        'normalize_embeddings': True,
    }
)

vector_store = Chroma(
    embedding_function= embeddings,
    collection_name= 'book',
    persist_directory= 'data/chroma')

prompt = ChatPromptTemplate.from_template("""Ты — эксперт по Рассказу Чехова Хамелеон.
Если ответа нет в контексте — так и скажи: В предоставленных отрывках этого нет. 
Вопрос: {question}
Контекст: {context}
Ответ:""")

llm = Ollama(model="modelscope.cn/qwen/Qwen2.5-7B-Instruct-GGUF:latest")

question = "Кто автор данного рассказа?"

retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = '\n'.join([doc.page_content for doc in retrieved_docs])

message = prompt.invoke({'question': question, 'context': docs_content})

answer = llm.invoke(message)

print(answer)