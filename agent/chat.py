#!/usr/bin/env python3
"""
交互式文件分析聊天系统
支持：
1. 可以调用工具读取文件夹和文件内容（由LLM决定）
2. 记忆功能（保存对话历史）
3. 人工追问和干预
"""

import os
import sys
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

from langgraph.checkpoint.memory import MemorySaver # checkpoint

from load_env import API_KEY, BASE_URL

# 初始化 LLM
llm = ChatOpenAI(
    streaming=True,     # [1] 开启流式输出
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="gpt-5.1",
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
        return f"错误：列出目录时出现问题 - {str(e)}"


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
    """LLM node, decide if there is a need to use tool"""
    system_prompt = """You a are professional file-analyzing assistant. You can use the following tools to help your user:
1. list_directory: list the content of a dir
2. read_file_tool: read the content of a (text) file

Decide if there is a need to use tool according to user's need. If user request to analyze file/dir, 
or you think the analyzed file need the content of another file, you should invoke according tools.
"""
    
    messages = [SystemMessage(content=system_prompt)]
    # messages.extend(state["messages"])
    
    # NOTE: 改为stream后，这里如果还用invoke，仍然不会获得event
    # 使用 invoke 返回完整响应（节点函数需要返回完整结果）
    # 流式输出在 main 函数中通过 graph.stream() 处理
    # response = model_with_tools.invoke(messages)
    
    # return {"messages": [response]}

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # 关键：直接调用 runnable，而不是 invoke
    # return {
    #     "messages": model_with_tools
    # }


    #  这里必须用 [] 包裹，因为会对state进行 "add"，要求是list
    return { 
        "messages": [RunnableLambda(lambda _: model_with_tools.invoke(messages)) ]
    }



def tool_node(state: State):
    """Tool node"""
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
            observation = f"错误：执行工具时出现问题 - {str(e)}"
        
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

    # 增加check point
    checkpointer = MemorySaver()
    
    return graph_builder.compile(checkpointer=checkpointer)


def main():
    """主函数"""
    print("=" * 60)
    print("File Helper")
    print("=" * 60)
    print("\nCurrent working directory:", os.getcwd())
    print("\nprompt>")
    print("  - 'quit', 'exit', 'q' to exit")
    print("  - 'clear' to clear history")
    print("  - 'show' to show chat history")
    print("-" * 60)
    
    # 构建图
    graph = build_graph()
    
    # 不再使用current_state（隐藏细节）
    # 使用check point
    session_id = 1
    thread_id = f"file-helper-session-{session_id}"
    
    # 交互式循环
    while True:
        user_input = input("\nUser> ").strip()
        
        if not user_input:
            continue
        
        # 处理退出命令
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break
        
        # 处理清空历史命令
        if user_input.lower() == "clear":
            # 使用checkpoint后，不再手动管理State状态
            session_id += 1
            thread_id = f"file-helper-session-{session_id}"
            print("clear the chat history")
            continue
        
        # 处理查看历史命令
        if user_input.lower() == "show":
            print("\n" + "=" * 60)
            print("chat history:")
            print("=" * 60)

            state = graph.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            # 手动获得state，然后再打印
            messages = state.values["messages"]
            for i, msg in enumerate(messages, 1):
                if isinstance(msg, HumanMessage):
                    print(f"\n[{i}] User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"\n[{i}] Assistant: {msg.content}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"      [invoking: {[tc['name'] for tc in msg.tool_calls]}]")
                elif isinstance(msg, ToolMessage):
                    print(f"\n[{i}] [result] {msg.content[:200]}...")
            print("=" * 60)
            continue
        
        # 处理用户输入
        try:
            # 添加用户消息到状态
            # 使用check point后，不再需要手动new state
            
            # 使用 stream 进行流式处理
            print("\nAssistant: ", end="", flush=True)

            inputs = {
                "messages": [HumanMessage(content=user_input)]
            }


            
            # 流式处理图的执行
            for event in graph.stream(
                inputs,
                stream_mode="events",
                config={"configurable": {"thread_id": thread_id}}   # 指定thread id，可供checkpoint使用
            ):     # [2] stream node
                print('------------------')
                print(event)
                # [3] 监听token事件
                if event["event"] == "on_llm_stream":
                    chunk = event["chunk"]
                    print(chunk, end="", flush=True)

                elif event["event"] == "on_tool_start":
                    print(f"\n[invoking: {event['name']}]")

                elif event["event"] == "on_tool_end":
                    print("\n[tool finished]")
            
        except Exception as e:
            print(f"\n错误：处理请求时出现问题 - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

