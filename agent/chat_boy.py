#!/usr/bin/env python3
"""
智能文件分析助手（修正版）
根据 info_chatboy_ver1.md 的建议重构：
- thinking 只发生一次，使用结构化输出
- decide 是纯程序逻辑，不调用 LLM
- act 只执行工具，不调用 LLM
- observe 后条件判断，不无条件重新思考
"""

import os
import operator
import json
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from langchain.tools import tool
from pydantic import SecretStr, BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

from load_env import API_KEY, BASE_URL

# 初始化 LLM（非流式输出）
llm = ChatOpenAI(
    streaming=False,
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


# 定义结构化输出模型
class ThinkingOutput(BaseModel):
    """思考输出结构"""
    thinking: str = Field(description="Your thinking process about the user's request")
    plan: str = Field(description="Your step-by-step plan to solve the problem")
    next_action: Literal["use_tool", "ask_user", "respond"] = Field(
        description="What to do next: use_tool if need to call a tool, ask_user if need more info, respond if can answer now"
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="If next_action is 'use_tool', specify which tool: 'list_directory' or 'read_file_tool'"
    )
    tool_args: Optional[dict] = Field(
        default=None,
        description="If next_action is 'use_tool', specify the tool arguments"
    )
    question: Optional[str] = Field(
        default=None,
        description="If next_action is 'ask_user', specify the question to ask"
    )


# 使用结构化输出的 LLM
structured_llm = llm.with_structured_output(ThinkingOutput)


class State(TypedDict):
    """Agent 状态"""
    messages: Annotated[list[AnyMessage], operator.add]
    thinking: str  # 思考过程（用于显示，不注入到 prompt）
    plan: str  # 计划（用于显示，不注入到 prompt）
    next_action: Optional[str]  # 下一步行动
    tool_name: Optional[str]  # 要调用的工具名
    tool_args: Optional[dict]  # 工具参数
    question: Optional[str]  # 要问用户的问题
    task_complete: bool  # 任务是否完成


def thinking_node(state: State):
    """思考节点：一次性完成思考、计划、决策（只调用一次 LLM）"""
    system_prompt = """You are a professional file-analyzing assistant. 

You have access to these tools:
- list_directory: list the content of a directory
- read_file_tool: read the content of a file

Analyze the user's request, think about what you need, make a plan, and decide what to do next.

Think step by step, then output your plan and next action."""
    
    # 获取最近的对话上下文（只取最后几条消息，避免上下文过长）
    recent_messages = state["messages"][-10:]  # 只取最近10条消息
    
    # 构建消息列表
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(recent_messages)

    # print(messages)
    
    print("\n" + "="*60)
    print("🤔 THINKING:")
    print("="*60)
    
    # 调用结构化 LLM（只调用一次）
    try:
        print(messages)
        result = structured_llm.invoke('hello, please think')
        print('*****************')
        result = structured_llm.invoke(messages)
        # result = llm.invoke(messages)
        # print(result)
        
        thinking_content = result.thinking
        plan_content = result.plan
        next_action = result.next_action
        
        print(thinking_content)
        print("\n" + "-"*60)
        print("📋 PLAN:")
        print("-"*60)
        print(plan_content)
        print("="*60 + "\n")
        
        # 返回结构化结果
        return {
            "thinking": thinking_content,
            "plan": plan_content,
            "next_action": next_action,
            "tool_name": result.tool_name,
            "tool_args": result.tool_args,
            "question": result.question,
            "task_complete": False
        }
    except Exception as e:
        print(f"Error in thinking: {e}")
        # 如果结构化输出失败，回退到 respond
        return {
            "thinking": f"Error in thinking: {e}",
            "plan": "Unable to create plan",
            "next_action": "respond",
            "task_complete": False
        }


def decide_node(state: State):
    """决策节点：纯程序逻辑，不调用 LLM（只用于显示）"""
    next_action = state.get("next_action", "respond")
    print(f"\n📋 DECISION: {next_action.upper()}\n")
    return {}  # 不修改状态，只用于显示


def route_after_decide(state: State) -> Literal["act", "ask_user", "respond", END]:
    """路由函数：根据决策结果路由"""
    next_action = state.get("next_action", "respond")
    
    if next_action == "use_tool":
        return "act"
    elif next_action == "ask_user":
        return "ask_user"
    elif next_action == "respond":
        return "respond"
    else:
        return END


def act_node(state: State):
    """执行节点：只执行工具，不调用 LLM"""
    tool_name = state.get("tool_name")
    tool_args = state.get("tool_args", {})
    
    if not tool_name or tool_name not in tools_by_name:
        print(f"❌ Error: Invalid tool name '{tool_name}'")
        return {
            "messages": [AIMessage(content=f"Error: Invalid tool name '{tool_name}'")],
            "task_complete": True
        }
    
    tool = tools_by_name[tool_name]
    
    print(f"\n🔧 ACTING: Calling {tool_name}")
    print(f"   Args: {tool_args}\n")
    
    try:
        # 只执行工具，不调用 LLM
        observation = tool(**tool_args)
        print(f"   ✓ Tool executed successfully\n")
        
        # 创建工具消息（注意：这里需要模拟 tool_call_id，实际应该从 thinking 阶段获取）
        # 为了简化，我们直接创建 ToolMessage
        tool_message = ToolMessage(
            content=str(observation),
            tool_call_id=f"call_{tool_name}_{hash(str(tool_args))}"
        )
        
        return {"messages": [tool_message]}
    except Exception as e:
        error_msg = f"Error executing tool {tool_name}: {str(e)}"
        print(f"   ✗ {error_msg}\n")
        return {
            "messages": [ToolMessage(content=error_msg, tool_call_id="error")],
            "task_complete": True
        }


def observe_node(state: State):
    """观察节点：处理工具执行结果"""
    last_message = state["messages"][-1] if state["messages"] else None
    
    if isinstance(last_message, ToolMessage):
        print("\n👀 OBSERVING: Tool result received\n")
        # 工具结果已收到，继续流程
        return {}
    
    return {}


def should_continue_after_observe(state: State) -> Literal["thinking", "respond", END]:
    """观察后判断：是否需要重新思考或直接回复"""
    # 检查是否有足够的工具结果来回答问题
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    
    # 如果有工具结果，检查是否还需要更多信息
    if tool_messages:
        # 检查任务是否可能完成（这里简化处理，实际可以更智能）
        # 如果工具调用成功，通常可以尝试回答
        last_tool_msg = tool_messages[-1]
        if "Error" in last_tool_msg.content:
            # 工具执行失败，可能需要重新思考或询问用户
            return "thinking"
        else:
            # 工具执行成功，可以尝试回答
            return "respond"
    
    # 没有工具结果，直接回复
    return "respond"


def respond_node(state: State):
    """响应节点：生成最终回复"""
    system_prompt = """You are a professional file-analyzing assistant. Based on the conversation history and any tool results, provide a clear and helpful answer to the user.

Do NOT repeat your thinking process or plan in the response. Just provide the answer directly."""
    
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    
    # 注意：不注入 thinking/plan 文本到 prompt（避免反模式）
    # thinking/plan 只用于显示和程序逻辑
    
    print("\n💬 RESPONDING: Generating answer...\n")
    
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "task_complete": True
    }


def ask_user_node(state: State):
    """询问用户节点：向用户请求更多信息"""
    question = state.get("question", "I need more information to help you. Could you please provide more details?")
    
    print("\n❓ ASKING USER:\n")
    print(question)
    print()
    
    # 创建询问消息
    ask_message = AIMessage(content=question)
    
    return {
        "messages": [ask_message],
        "task_complete": False
    }


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
    
    # 思考后进入决策（纯程序逻辑）
    graph_builder.add_edge("thinking", "decide")
    
    # 决策后根据结果路由
    graph_builder.add_conditional_edges(
        "decide",
        route_after_decide,  # 使用路由函数
        {
            "act": "act",
            "ask_user": "ask_user",
            "respond": "respond",
            END: END
        }
    )
    
    # 执行工具后观察结果
    graph_builder.add_edge("act", "observe")
    
    # 观察后条件判断
    graph_builder.add_conditional_edges(
        "observe",
        should_continue_after_observe,
        {
            "thinking": "thinking",  # 需要重新思考
            "respond": "respond",  # 可以直接回复
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
    print("🤖 Smart File Helper (Corrected Version)")
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
                "next_action": None,
                "tool_name": None,
                "tool_args": None,
                "question": None,
                "task_complete": False
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
