#!/usr/bin/env python3
"""
智能文件分析助手（支持思考、计划、决策、执行）
包含完整的 Agent 架构：
- thinking/plan: 思考并制定计划
- decide: 决定下一步行动
- act: 执行工具调用
- observe: 观察工具执行结果
"""

import os
import operator
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain.tools import tool
from pydantic import SecretStr
from langgraph.checkpoint.memory import MemorySaver

from load_env import API_KEY, BASE_URL

# 初始化 LLM（非流式输出）
llm = ChatOpenAI(
    streaming=False,
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
            file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            return f"Error: file '{file_path}' doesn't exist"
        
        if not os.path.isfile(file_path):
            return f"Error: file '{file_path}' is not a file"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"file: '{file_path}' , length: {len(content)}, content:\n\n{content}"
        except UnicodeDecodeError:
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
    """Agent 状态"""
    messages: Annotated[list[AnyMessage], operator.add]
    thinking: str  # 思考过程
    plan: str  # 计划
    action: Optional[str]  # 当前行动
    needs_more_info: bool  # 是否需要更多信息


def thinking_node(state: State):
    """思考节点：分析用户需求并制定计划"""
    system_prompt = """You are a professional file-analyzing assistant. You need to:
1. Understand the user's request
2. Think about what information you need
3. Make a plan to solve the problem

You have access to these tools:
- list_directory: list the content of a directory
- read_file_tool: read the content of a file

Think step by step about:
- What does the user want?
- What information do I need?
- What tools should I use?
- What is my plan?

Format your thinking and plan clearly."""
    
    # 获取用户最后一条消息
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    last_user_message = user_messages[-1] if user_messages else None
    
    if not last_user_message:
        return {
            "thinking": "No user message found.",
            "plan": "Wait for user input.",
            "action": "wait"
        }
    
    # 构建思考提示词
    thinking_prompt = f"""User request: {last_user_message.content}

Previous context:
{chr(10).join([f"- {msg.content[:100]}" for msg in state["messages"][-5:-1] if hasattr(msg, 'content')])}

Think about:
1. What is the user asking for?
2. Do I have enough information to answer?
3. What tools do I need to use?
4. What is my step-by-step plan?

Provide your thinking process and plan:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=thinking_prompt)
    ]
    
    response = llm.invoke(messages)
    thinking_content = response.content
    
    # 提取思考和计划
    thinking = thinking_content
    plan = thinking_content  # 可以进一步解析，这里简化处理
    
    print("\n" + "="*60)
    print("🤔 THINKING:")
    print("="*60)
    print(thinking_content)
    print("="*60 + "\n")
    
    return {
        "thinking": thinking_content,
        "plan": plan,
        "action": "decide"
    }


def decide_node(state: State):
    """决策节点：决定下一步行动"""
    system_prompt = """Based on your thinking and plan, decide what to do next.

Options:
1. "use_tool" - If you need to call a tool (list_directory or read_file_tool)
2. "ask_user" - If you need more information from the user
3. "respond" - If you have enough information to answer the user

Respond with ONLY one word: "use_tool", "ask_user", or "respond"."""
    
    thinking = state.get("thinking", "")
    plan = state.get("plan", "")
    
    decision_prompt = f"""Thinking: {thinking}

Plan: {plan}

What should I do next? Choose one:
- "use_tool" if I need to use a tool
- "ask_user" if I need more information
- "respond" if I can answer now

Decision:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=decision_prompt)
    ]
    
    response = llm.invoke(messages)
    decision = response.content.strip().lower()
    
    # 清理决策文本，提取关键词
    if "use_tool" in decision or "tool" in decision:
        decision = "use_tool"
    elif "ask_user" in decision or "ask" in decision or "more" in decision:
        decision = "ask_user"
    else:
        decision = "respond"
    
    print(f"\n📋 DECISION: {decision.upper()}\n")
    
    return {
        "action": decision
    }


def act_node(state: State):
    """执行节点：调用工具"""
    system_prompt = """You are a professional file-analyzing assistant. You can use the following tools:
1. list_directory: list the content of a dir
2. read_file_tool: read the content of a (text) file

Based on your plan, decide which tool to use and call it."""
    
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    
    # 添加思考过程到上下文
    if state.get("thinking"):
        messages.append(SystemMessage(
            content=f"Your thinking: {state['thinking']}\nYour plan: {state.get('plan', '')}"
        ))
    
    print("\n🔧 ACTING: Calling tool...\n")
    
    response = model_with_tools.invoke(messages)
    
    # 检查是否调用了工具
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc['name'] for tc in response.tool_calls]
        print(f"📌 Invoking tools: {', '.join(tool_names)}\n")
    
    return {"messages": [response]}


def observe_node(state: State):
    """观察节点：执行工具并观察结果"""
    last_message = state["messages"][-1]
    
    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        return {"messages": []}
    
    print("\n👀 OBSERVING: Executing tools...\n")
    
    result = []
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        print(f"  → Executing {tool_call['name']} with args: {tool_call['args']}")
        
        try:
            observation = tool.invoke(tool_call["args"])
            print(f"  ✓ Tool executed successfully\n")
        except Exception as e:
            observation = f"Error: failed to execute tool - {str(e)}"
            print(f"  ✗ Tool execution failed: {e}\n")
        
        result.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": result}


def respond_node(state: State):
    """响应节点：生成最终回复"""
    system_prompt = """You are a professional file-analyzing assistant. Based on your thinking, plan, and the information you've gathered, provide a clear and helpful answer to the user."""
    
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    
    # 添加思考过程
    if state.get("thinking"):
        messages.append(SystemMessage(
            content=f"Your thinking process: {state['thinking']}"
        ))
    
    print("\n💬 RESPONDING: Generating answer...\n")
    
    response = llm.invoke(messages)
    
    return {"messages": [response]}


def ask_user_node(state: State):
    """询问用户节点：向用户请求更多信息"""
    system_prompt = """You need more information from the user to complete the task. Ask a clear and specific question."""
    
    thinking = state.get("thinking", "")
    plan = state.get("plan", "")
    
    ask_prompt = f"""Based on your thinking and plan, what information do you need from the user?

Thinking: {thinking}
Plan: {plan}

Ask the user a clear question:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ask_prompt)
    ]
    
    response = llm.invoke(messages)
    
    print("\n❓ ASKING USER:\n")
    print(response.content)
    print()
    
    return {
        "messages": [response],
        "needs_more_info": True
    }


def should_continue(state: State) -> Literal["act", "respond", "ask_user", END]:
    """路由函数：根据决策决定下一步"""
    action = state.get("action", "respond")
    
    if action == "use_tool":
        return "act"  # 需要调用工具
    
    elif action == "ask_user":
        return "ask_user"
    
    elif action == "respond":
        return "respond"
    
    else:
        return END


def should_loop_after_observe(state: State) -> Literal["thinking", END]:
    """观察工具结果后，判断是否需要重新思考"""
    # 工具执行完成后，总是重新思考下一步
    return "thinking"


def build_graph():
    """构建 Agent 图"""
    graph_builder = StateGraph(State)
    
    # 添加节点
    graph_builder.add_node("thinking", thinking_node)
    graph_builder.add_node("decide", decide_node)
    graph_builder.add_node("act", act_node)
    graph_builder.add_node("observe", observe_node)
    graph_builder.add_node("respond", respond_node)
    graph_builder.add_node("ask_user", ask_user_node)
    
    # 设置入口
    graph_builder.add_edge(START, "thinking")
    
    # 思考后进入决策
    graph_builder.add_edge("thinking", "decide")
    
    # 决策后根据结果路由
    graph_builder.add_conditional_edges(
        "decide",
        should_continue,
        {
            "act": "act",
            "observe": "observe",
            "respond": "respond",
            "ask_user": "ask_user",
            END: END
        }
    )
    
    # 执行工具后观察结果
    graph_builder.add_edge("act", "observe")
    
    # 观察后重新思考下一步
    graph_builder.add_conditional_edges(
        "observe",
        should_loop_after_observe,
        {
            "thinking": "thinking",  # 重新思考
            END: END
        }
    )
    
    # 响应和询问用户后结束
    graph_builder.add_edge("respond", END)
    graph_builder.add_edge("ask_user", END)
    
    # 使用 checkpoint 支持记忆
    checkpointer = MemorySaver()
    
    return graph_builder.compile(checkpointer=checkpointer)


def main():
    """主函数"""
    print("=" * 60)
    print("🤖 Smart File Helper (with Thinking & Planning)")
    print("=" * 60)
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("You can ask me to analyze files, list directories, etc.\n")
    print("Type 'quit' or 'exit' to quit.\n")
    
    # 构建图
    graph = build_graph()
    
    # 使用 checkpoint 管理对话状态
    thread_id = "chat-boy-session"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 交互式循环
    while True:
        try:
            user_input = input("User> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nBye! 👋\n")
                break
            
            # 准备输入（只传入新消息，checkpoint 会自动维护历史状态）
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "thinking": "",
                "plan": "",
                "action": None,
                "needs_more_info": False
            }
            
            # 执行图（checkpoint 会自动保存和恢复状态）
            result = graph.invoke(inputs, config=config)
            
            # 显示最终回复
            if result.get("messages"):
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    print("\n" + "="*60)
                    print("🤖 ASSISTANT:")
                    print("="*60)
                    print(last_message.content)
                    print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nBye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

