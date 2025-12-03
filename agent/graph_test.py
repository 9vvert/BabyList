import os
from dotenv import load_dotenv
from pydantic import SecretStr
from IPython.display import Image, display

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv('../../.llm_env')
API_KEY : str= os.getenv("MY_OPENAI_API_KEY") or ""
BASE_URL = os.getenv("MY_OPENAI_API_BASE")

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
def chatbot(state: State):
    print(state)
    return {"messages": [llm.invoke(state["messages"])]}
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever the node is used.’’’
graph_builder.add_node("chatbot", chatbot)

graph = graph_builder.compile()
try:
    with open("struct.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
except Exception:
    print('error')

# Run the chatbot
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
