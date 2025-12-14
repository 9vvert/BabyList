#!/usr/bin/env python3
"""
交互式文件分析聊天系统（流式输出版本）
支持：
1. 工具调用（读取文件夹和文件内容，由LLM决定）
2. 流式输出
3. 记忆功能（支持追问）
"""

import os
import operator
from typing import Annotated, Literal
from langchain_core.runnables import RunnableLambda
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain.tools import tool
from pydantic import SecretStr
from langgraph.checkpoint.memory import MemorySaver

from load_env import API_KEY, BASE_URL

# 初始化 LLM（开启流式输出）
llm = ChatOpenAI(
    streaming=True,
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="gpt-4.1",
    temperature=0
)


# 定义工具
@tool
def list_directory(directory_path: str = ".") -> str:
    """list file under specific directory
    
    Args:
        directory_path: dir path to be listed, default is current dir: '.'
    
    Returns:
        content list of a directory
    """
    try:
        if not os.path.isabs(directory_path):
            # 如果是相对路径，基于当前工作目录
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
        
        result = f"content of dir '{directory_path}':\n" + "\n".join(items)
        return result
    except Exception as e:
        return f"Error: failed to list directory - {str(e)}"


@tool
def read_file_tool(file_path: str) -> str:
    """read content of certain file
    
    Args:
        file_path: path of target file (absolute or relative path)
    
    Returns:
        file content. if doesn't exist, it is error message.
    """
    try:
        if not os.path.isabs(file_path):
            # 如果是相对路径，基于当前工作目录
            file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' doesn't exist"
        
        if not os.path.isfile(file_path):
            return f"Error: file '{file_path}' is not a file"
        
        # 尝试读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"file: '{file_path}' , length: {len(content)}, content:\n\n{content}"
        except UnicodeDecodeError:
            # 如果是二进制文件，尝试以二进制模式读取
            with open(file_path, 'rb') as f:
                content = f.read()
            return f"file '{file_path}' is a binary file with length {len(content)}, cannot displayed as text."
    except Exception as e:
        return f"Error in reading file - {str(e)}"


# 绑定工具到 LLM
tools = [list_directory, read_file_tool]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """对话状态，包含消息历史"""
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: State):
    """LLM 节点，决定是否调用工具"""
    system_prompt = """You are a professional file-analyzing assistant. You can use the following tools to help your user:
1. list_directory: list the content of a dir
2. read_file_tool: read the content of a (text) file

Decide if there is a need to use tool according to user's need. If user request to analyze file/dir, 
or you think the analyzed file need the content of another file, you should invoke according tools.
"""
    
    # messages = [SystemMessage(content=system_prompt)]
    # messages.extend(state["messages"])
    
    # # 调用 LLM（流式输出在 main 函数中通过 graph.stream() 处理）
    # response = model_with_tools.invoke(messages)
    
    # return {"messages": [response]}

    system_prompt = """You are a professional file-analyzing assistant. You can use the following tools..."""
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    # 注意这里使用 stream 而不是 invoke
    return {"messages": [RunnableLambda(lambda _: model_with_tools.stream(messages))]}

def tool_node(state: State):
    """工具节点，执行工具调用"""
    result = []
    last_message = state["messages"][-1]
    
    # 检查是否有工具调用
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        try:
            observation = tool.invoke(tool_call["args"])
        except Exception as e:
            observation = f"Error: failed to execute tool - {str(e)}"
        
        result.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": result}


def should_continue(state: State) -> Literal["tool_node", END]:
    """判断是否继续调用工具"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果 LLM 调用了工具，则执行工具节点
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    
    # 否则结束，返回给用户
    return END


def build_graph():
    """构建对话图"""
    graph_builder = StateGraph(State)
    
    # 添加节点
    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)
    
    # 设置入口
    graph_builder.add_edge(START, "llm_call")
    
    # 条件边：根据是否调用工具决定下一步
    graph_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END
        }
    )
    
    # 工具执行后返回 LLM
    graph_builder.add_edge("tool_node", "llm_call")
    
    # 使用 checkpoint 支持记忆功能
    checkpointer = MemorySaver()
    
    return graph_builder.compile(checkpointer=checkpointer)


def main():
    """主函数"""
    print("=" * 60)
    print("File Helper (Streaming)")
    print("=" * 60)
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("You can ask me to list directories or read files.\n")
    
    # 构建图
    graph = build_graph()
    
    # 使用 checkpoint 管理对话状态
    thread_id = "file-helper-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 交互式循环
    while True:
        try:
            user_input = input("User> ").strip()
            
            if not user_input:
                continue
            
            # 准备输入
            inputs = {
                "messages": [HumanMessage(content=user_input)]
            }
            
            # 流式处理
            print("\nAssistant: ", end="", flush=True)
            
            # 使用 stream_mode="events" 来捕获细粒度事件
            for event in graph.stream(
                inputs,
                config=config,
                stream_mode="events"
            ):
                print(event)
                event_type = event.get("event")
                
                # 捕获 LLM 的 token 流式输出
                if event_type == "on_llm_stream":
                    chunk = event.get("chunk")
                    if chunk:
                        # chunk 可能是字符串、AIMessageChunk 或其他对象
                        if isinstance(chunk, str):
                            print(chunk, end="", flush=True)
                        elif hasattr(chunk, 'content'):
                            content = chunk.content
                            if content:
                                print(content, end="", flush=True)
                        else:
                            # 尝试直接打印
                            print(str(chunk), end="", flush=True)
                
                # 捕获工具调用开始
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    print(f"\n[invoking tool: {tool_name}]")
                
                # 捕获工具调用结束
                elif event_type == "on_tool_end":
                    print("[tool finished]")
            
            print()  # 换行
            
        except KeyboardInterrupt:
            print("\n\nBye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

