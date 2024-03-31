from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(model="mistral")


prompt = ChatPromptTemplate.from_messages([
                                              ("system", "tu es un documentaliste qui répond en francais"),
                                              ("user", "{input}")
                                          ])


print(prompt.messages)


# res = llm.invoke("quel est la capital de France ? ")

# print(res)
