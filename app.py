from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter



import chainlit as cl
import os 

os.environ["OPENAI_API_KEY"]= "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"

@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(model="mistral")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    db = Chroma(persist_directory="./chroma_db4", embedding_function=embeddings)
    retriever = db.as_retriever(k=2)

    template = """Répond à la question en français et en utilisant le contexte suivant:
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)    
    
    chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model 
    | StrOutputParser()
    )

   
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):


    chain = cl.user_session.get("chain")

    answer = cl.Message(content="")
    # config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

    print(message.content)
    async for chunk in chain.astream(message.content):
        await answer.stream_token(chunk)        


    await answer.send()
