from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableParallel
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


@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(model="mistral")
    
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

    
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs = {"k": 3})

    template = """Répond à la question en français et en utilisant uniquement le contexte. Si tu ne trouve rien dans le contexte, répond : 'je ne sais pas':
    Contexte: {context}
    Question: [/INST] {question} [INST]
    """
    
    prompt = ChatPromptTemplate.from_template(template)    
    
    # chain = (
    # {"context": retriever, "question": RunnablePassthrough()}
    # | prompt
    # | model 
    # )

    rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: x["context"]))
    | prompt
    | model
    | StrOutputParser()
    )

    chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
   
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):


    chain = cl.user_session.get("chain")

    answer = cl.Message(content="")
    # config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

    sources = set()
    async for chunk in chain.astream(message.content):

        if "context" in chunk:
            for doc in chunk["context"]:
                sources.add(doc.metadata["source"])
                                
        if "answer" in chunk:
            await answer.stream_token(chunk["answer"])


    answer.elements = [cl.Pdf(name="pdf1", display="inline", path=s) for s in sources]                

    
    await answer.send()
