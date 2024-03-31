from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI

import chainlit as cl
import os 

os.environ["OPENAI_API_KEY"]= "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"

@cl.on_chat_start
async def on_chat_start():
    model = ChatOllama(model="mistral")
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Tu es un assistant qui parle français"),
            ("human","{question}")
        ]
    )
    
    llm = prompt | model | StrOutputParser()
    cl.user_session.set("llm", llm)


@cl.on_message
async def on_message(message: cl.Message):

    
    llm = cl.user_session.get("llm")

    answer = cl.Message(content="")

    config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
    
    async for chunk in llm.astream({"question": message.content},config=config):
        await answer.stream_token(chunk)        


    await answer.send()
