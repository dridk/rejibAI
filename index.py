from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableParallel
from langchain.schema.runnable.config import RunnableConfig
from operator import itemgetter

# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
import chromadb
from glob import glob 

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

for file in glob("docs/*.pdf"):
    print(file)
    loader = PyPDFLoader(file)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(pages)
    db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")


# db = Chroma(persist_directory="./chroma_db5", embedding_function=embeddings)
# retriever = db.as_retriever(search_kwargs = {"k": 20})

# for d in retriever.invoke("Comment coder un sepsis  ? "):

#     print(d.page_content)
