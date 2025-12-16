import os
import asyncio
from typing import Annotated, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from load_env import *

# Configure the base chat model. Temperature kept low for determinism.
BASE_MODEL = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="gpt-5.1",
    temperature=0.2,
)


@tool("list_dir")
def list_dir(path: str = ".") -> str:
    """List the files and folders inside a directory."""
    target = os.path.abspath(path)
    if not os.path.exists(target):
        return f"Path not found: {target}"
    if not os.path.isdir(target):
        return f"Not a directory: {target}"
    try:
        entries = sorted(os.listdir(target))
    except OSError as exc:
        return f"Error reading directory {target}: {exc}"
    return "\n".join(entries)


@tool("read_file")
def read_file(path: str) -> str:
    """Read a text file and return its contents (truncated to 4000 characters)."""
    target = os.path.abspath(path)
    if not os.path.exists(target):
        return f"File not found: {target}"
    if not os.path.isfile(target):
        return f"Not a file: {target}"
    try:
        with open(target, "r", encoding="utf-8", errors="ignore") as file:
            data = file.read(4000)
    except OSError as exc:
        return f"Error reading file {target}: {exc}"
    return data


tools = [list_dir, read_file]
tool_node = ToolNode(tools)
model_with_tools = BASE_MODEL.bind_tools(tools)


SYSTEM_PROMPT = (
    "You are Analyzer Bot. Use the tools list_dir and read_file when they help. "
    "Show a concise visible thinking trace using the format 'Thinking: ...' followed "
    "by 'Answer: ...'. Keep thinking compact but real. Be direct and avoid fluff."
)


class State(dict):
    messages: Annotated[List, add_messages]


graph_builder = StateGraph(State)

graph_builder.add_node(
    "agent",
    lambda state: {
        "messages": [model_with_tools.invoke(state["messages"])]
    },
)

# Node that executes whichever tool was requested.
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")
# Route to tools when the model requests them, otherwise end.
graph_builder.add_conditional_edges("agent", tools_condition)
# After a tool runs, return to the agent for follow-up.
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()


def format_chunk_content(chunk_content) -> str:
    if isinstance(chunk_content, list):
        # Some providers return a list of parts; we only need the text portions here.
        return "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in chunk_content])
    return str(chunk_content)


def print_tool_message(tool_message: ToolMessage) -> None:
    tool_output = format_chunk_content(tool_message.content)
    print(f"\n[tool:{tool_message.name}]\n{tool_output}\n")


async def chat_loop() -> None:
    # Conversation memory persists across turns in this list.
    history: List = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("\nUser> ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        history.append(HumanMessage(content=user_input))

        assembled_reply = ""
        pending_tool_messages: List[ToolMessage] = []

        async for message, _metadata in graph.astream(
            {"messages": history},
            stream_mode="messages",
        ):
            if message is None:
                continue

            if isinstance(message, ToolMessage):
                pending_tool_messages.append(message)
                print_tool_message(message)
                continue
            else:
                if message.content:
                    chunk_text = format_chunk_content(message.content)
                    assembled_reply += chunk_text
                    print(chunk_text, end="", flush=True)

        # Persist any tool outputs and the assistant reply for memory.
        history.extend(pending_tool_messages)
        if assembled_reply:
            history.append(AIMessage(content=assembled_reply))


if __name__ == "__main__":
    asyncio.run(chat_loop())
