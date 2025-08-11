import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

def generate_essay(topic:str, word_limit:int)->str:
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.3,api_key=os.getenv("OPENAI_API_KEY")
    )
    prompt = PromptTemplate(
        input_variables=["topic", "word limit"],
        template="Write a well-structured essay of {word_limit}words and on the topic {topic}"
    )
    chain = prompt | llm
    response= chain.invoke({"topic":topic, "word_limit":word_limit})
    return response.content

    