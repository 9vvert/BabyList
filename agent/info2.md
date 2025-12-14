1️⃣ 为什么用 RunnableLambda / Runnable？

在 LangGraph 或 LangChain 里，Runnable 是一个“延迟执行、可组合”的对象，主要用途是：

延迟执行 LLM 调用

你不希望在节点定义时就调用 LLM，而是在 Graph 执行时才调用。

RunnableLambda(lambda _: llm.invoke(...)) 就是一个延迟执行的方式。

支持流式事件

Graph 能捕捉 runnable 执行过程中的 token / 工具调用事件 (on_llm_stream / on_tool_start / on_tool_end)。

如果直接调用 llm.invoke()，整个输出一次性返回，不会触发流式事件。

可组合 / 可封装

Runnable 可以和其他 Runnable 组合（Sequence、Map、Lambda），实现复杂流水线。

Graph 内部统一处理 Runnable 返回的对象或事件。