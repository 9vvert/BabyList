#!/usr/bin/env python3
"""
交互式文件分析聊天系统（流式输出）
特点：
- 流式输出 LLM 回复
- 工具调用边执行边显示
- 不使用 checkpoint
"""

import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain.tools import tool
from pydantic import SecretStr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableLambda

from load_env import API_KEY, BASE_URL

# -------------------- 初始化 LLM -------------------- #
llm = ChatOpenAI(
    streaming=True,  # 开启流式输出
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="gpt-5.1",
    temperature=0
)

# -------------------- 工具定义 -------------------- #
@tool
def list_directory(directory_path: str = ".") -> str:
    try:
        if not os.path.isabs(directory_path):
            directory_path = os.path.abspath(directory_path)
        if not os.path.exists(directory_path):
            return f"Error: dir '{directory_path}' doesn't exist"
        if not os.path.isdir(directory_path):
            return f"Error:'{directory_path}' is not a directory"
        items = []
        for item in sorted(os.listdir(directory_path)):
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                items.append(f"[dir] {item}/")
            else:
                size = os.path.getsize(item_path)
                items.append(f"[file] {item} ({size} bytes)")
        return f"content of dir '{directory_path}':\n" + "\n".join(items)
    except Exception as e:
        return f"错误：列出目录时出现问题 - {str(e)}"

@tool
def read_file_tool(file_path: str) -> str:
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' doesn't exist"
        if not os.path.isfile(file_path):
            return f"Error: file '{file_path}' is not a file"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"file: '{file_path}', length: {len(content)}, content:\n\n{content}"
        except UnicodeDecodeError:
            with open(file_path, 'rb') as f:
                content = f.read()
            return f"file '{file_path}' is binary, length {len(content)}, cannot display as text."
    except Exception as e:
        return f"Error in reading file - {str(e)}"

tools = [list_directory, read_file_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)

# -------------------- 状态类型 -------------------- #
class State(TypedDict):
    messages: list[AnyMessage]

# -------------------- 节点定义 -------------------- #
def llm_call(state: State):
    system_prompt = """You are a professional file-analyzing assistant. You can use the following tools:
1. list_directory: list dir content
2. read_file_tool: read file content
Decide if tool usage is needed according to user's request.
"""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return {
        "messages": [RunnableLambda(lambda _: model_with_tools.invoke(messages))]
    }

def tool_node(state: State):
    last_message = state["messages"][-1]
    result = []

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            try:
                observation = tool.invoke(tool_call["args"])
            except Exception as e:
                observation = f"错误：执行工具时出现问题 - {str(e)}"
            result.append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))
    return {"messages": result}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    return END

# -------------------- 构建图 -------------------- #
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", END: END}
    )
    graph_builder.add_edge("tool_node", "llm_call")
    return graph_builder.compile()  # 不使用 checkpoint

# -------------------- 主循环（重写流式部分） -------------------- #
def main():
    print("="*60)
    print("File Helper (streaming mode)")
    print("="*60)
    print("\nCurrent working directory:", os.getcwd())

    graph = build_graph()

    while True:
        user_input = input("\nUser> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break

        # 用户输入封装
        inputs = {"messages": [HumanMessage(content=user_input)]}

        print("\nAssistant: ", end="", flush=True)

        try:
            # 这里直接使用 stream() 进行增量输出
            for event in graph.stream(inputs, stream_mode="events"):
                evt_type = event.get("event")
                if evt_type == "on_llm_stream":
                    # 增量 token 输出
                    chunk = event.get("chunk", "")
                    print(chunk, end="", flush=True)
                elif evt_type == "on_llm_end":
                    # LLM 输出结束
                    print()  # 换行
                elif evt_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    print(f"\n[Invoking tool: {tool_name}]")
                elif evt_type == "on_tool_end":
                    print("\n[Tool finished]")

        except Exception as e:
            print(f"\n错误：{e}")

if __name__ == "__main__":
    main()

