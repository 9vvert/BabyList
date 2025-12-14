#!/usr/bin/env python3
"""
交互式文件分析聊天系统
支持：
1. 读取文件作为分析任务
2. 记忆功能（保存对话历史）
3. 人工追问和干预
"""

import os
import sys
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import SecretStr

from load_env import API_KEY, BASE_URL

# 初始化 LLM
llm = ChatOpenAI(
    api_key=SecretStr(API_KEY),
    base_url=BASE_URL,
    model="gpt-4.1",
    temperature=0
)


class State(TypedDict):
    """对话状态，包含消息历史和文件内容"""
    messages: Annotated[list, add_messages]
    file_content: str  # 保存读取的文件内容


def read_file(file_path: str) -> str:
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        sys.exit(1)
    except Exception as e:
        print(f"错误：读取文件时出现问题 - {e}")
        sys.exit(1)


def chatbot(state: State):
    """聊天机器人节点，处理用户输入并生成回复"""
    # 构建系统提示词，包含文件内容
    system_prompt = """你是一个专业的文件分析助手。用户会提供一个文件内容，你需要仔细分析它。

文件内容：
{file_content}

请根据用户的问题，对文件内容进行深入分析。如果用户没有提供具体问题，你可以主动提供文件的关键信息摘要。
""".format(file_content=state.get("file_content", ""))

    # 构建消息列表：系统消息 + 历史消息
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    
    # 调用 LLM
    response = llm.invoke(messages)
    
    return {"messages": [response]}


def build_graph():
    """构建对话图"""
    graph_builder = StateGraph(State)
    
    # 设置入口和出口
    graph_builder.set_entry_point("chatbot")
    graph_builder.set_finish_point("chatbot")
    
    # 添加节点
    graph_builder.add_node("chatbot", chatbot)
    
    return graph_builder.compile()


def main():
    """主函数"""
    print("=" * 60)
    print("文件分析聊天系统")
    print("=" * 60)
    
    # 获取文件路径
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("请输入要分析的文件路径: ").strip()
    
    if not file_path:
        print("错误：未提供文件路径")
        sys.exit(1)
    
    # 读取文件
    print(f"\n正在读取文件: {file_path}")
    file_content = read_file(file_path)
    print(f"文件读取成功，共 {len(file_content)} 个字符\n")
    
    # 构建图
    graph = build_graph()
    
    # 初始化状态
    initial_state = {
        "messages": [],
        "file_content": file_content
    }
    
    # 发送初始分析请求
    print("正在生成初始分析...")
    initial_message = "请分析这个文件，提供关键信息摘要。"
    result = graph.invoke({
        **initial_state,
        "messages": [HumanMessage(content=initial_message)]
    })
    
    # 显示初始分析结果
    if result["messages"]:
        print("\n" + "=" * 60)
        print("初始分析结果：")
        print("=" * 60)
        print(result["messages"][-1].content)
        print("=" * 60 + "\n")
    
    # 更新状态，保存对话历史
    current_state = result
    
    # 交互式循环
    print("提示：")
    print("  - 输入你的问题或指令进行追问")
    print("  - 输入 'quit'、'exit' 或 'q' 退出")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'show' 查看当前对话历史")
    print("-" * 60)
    
    while True:
        user_input = input("\n你: ").strip()
        
        if not user_input:
            continue
        
        # 处理退出命令
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
        
        # 处理清空历史命令
        if user_input.lower() == "clear":
            current_state = {
                "messages": [],
                "file_content": file_content
            }
            print("对话历史已清空")
            continue
        
        # 处理查看历史命令
        if user_input.lower() == "show":
            print("\n" + "=" * 60)
            print("对话历史：")
            print("=" * 60)
            for i, msg in enumerate(current_state["messages"], 1):
                if isinstance(msg, HumanMessage):
                    print(f"\n[{i}] 用户: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"\n[{i}] 助手: {msg.content}")
            print("=" * 60)
            continue
        
        # 处理用户输入
        try:
            # 添加用户消息到状态
            new_state = {
                **current_state,
                "messages": [HumanMessage(content=user_input)]
            }
            
            # 调用图处理
            result = graph.invoke(new_state)
            
            # 显示回复
            if result["messages"]:
                print("\n助手:", result["messages"][-1].content)
            
            # 更新当前状态，保留所有历史消息
            current_state = result
            
        except Exception as e:
            print(f"错误：处理请求时出现问题 - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

