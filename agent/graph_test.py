import os
from dotenv import load_dotenv
from pydantic import SecretStr

import asyncio

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from load_env import *

# load_dotenv('../../../.llm_env')
# API_KEY : str= os.getenv("MY_OPENAI_API_KEY") or ""
# BASE_URL = os.getenv("MY_OPENAI_API_BASE")

llm = ChatOpenAI(api_key=SecretStr(API_KEY),  base_url=BASE_URL,  model="gpt-4.1",temperature=0)

class State(TypedDict):
    # messages have the type "list".
    # The add_messages function appends messages to the list, rather than overwriting them
    messages: Annotated[list, add_messages]
graph_builder = StateGraph(State)

# Set entry and finish points
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# node
async def chatbot(state: State):
    async for chunk in llm.astream(state["messages"]):
        yield {"messages": [chunk]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used.’’’
graph_builder.add_node("chatbot", chatbot)

graph = graph_builder.compile()

# Run the chatbot
# while True:
#     user_input = input("User: ")
#     if user_input.lower() in ["quit", "exit", "q"]:
#         print("Goodbye!")
#         break
#     for message_chunk, metadata in graph.stream(
#         {"messages": [("user", user_input)]},
#         stream_mode="messages"
#     ):
#
#         if message_chunk:
#             print(message_chunk)
#         # for value in event.values():
#         #     print("Assistant:", value["messages"][-1].content)

async def main():
    while True:
        user_input = input("\nUser> ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        async for message_chunk, metadata in graph.astream(
            {"messages": [("user", user_input)]},
            stream_mode="messages",
        ):
            if message_chunk.content:
                print(message_chunk.content, end='')
                # print(message_chunk.content, end="", flush=True)

asyncio.run(main())
