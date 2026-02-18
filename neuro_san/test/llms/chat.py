# from anthropic import Anthropic

# client = Anthropic()

# response = client.messages.create(
#     model="claude-3-sonnet-20240229",
#     max_tokens=10,
#     messages=[{"role": "user", "content": "hello"}],
# )

# print(response)





# from langchain_anthropic import ChatAnthropic
# from langchain.schema import HumanMessage
# from langchain.callbacks.base import BaseCallbackHandler


# class StreamHandler(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs):
#         # Neuro-SAN expects streaming tokens
#         print(token, end="", flush=True)


# def get_llm():
#     return ChatAnthropic(
#         model="claude-3-sonnet-20240229",
#         temperature=0,
#         streaming=True,
#         callbacks=[StreamHandler()],
#     )


# def run_llm(user_input: str) -> str:
#     llm = get_llm()
#     response = llm.invoke([HumanMessage(content=user_input)])
#     return response.content

# run_llm("What is the capital city of France")




# from langchain_aws import ChatBedrock
# from langchain_core.messages import HumanMessage
# from langchain.callbacks.base import BaseCallbackHandler


# class StreamHandler(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs):
#         print(token, end="", flush=True)


# def get_llm():
#     return ChatBedrock(
#         model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#         streaming=True,
#         callbacks=[StreamHandler()],
#         region_name="us-east-1",
#     )


# def run_llm(user_input: str) -> str:
#     llm = get_llm()
#     response = llm.invoke([HumanMessage(content=user_input)])
#     return response.content

# print(run_llm("What is the capital city of France?"))


# from langchain_aws import ChatBedrock
# from langchain_core.messages import HumanMessage
# import boto3


# bedrock_client = boto3.client(
#     service_name="bedrock-runtime",
#     region_name="us-east-1" 
# )

# llm = ChatBedrock(
#     client=bedrock_client,
#     model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
#     model_kwargs={
#         "temperature": 0.7,
#         "max_tokens": 512,
#     },
# )


# response = llm.invoke(
#     [
#         HumanMessage(content="Explain LangChain in simple terms")
#     ]
# )

# print(response.content)



# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# llm = ChatAnthropic(
#     model="us.anthropic.claude-sonnet-4-20250514-v1:0",
#     temperature=0.2
# )

# prompt = ChatPromptTemplate.from_template("{input}")

# chain = prompt | llm | StrOutputParser()
# print("[DEBUG] chain : ",chain)

# def run_chain(user_input: str) -> str:
#     return chain.invoke({"input": user_input})

# run_chain("Im from france")

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import boto3

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

llm = ChatBedrock(
    client=bedrock_client,
    model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 256,
    },
)

prompt = ChatPromptTemplate.from_template("{input}")

chain = prompt | llm | StrOutputParser()

def run_chain(user_input: str) -> str:
    return chain.invoke({"input": user_input})

print(run_chain("I'm from France"))
