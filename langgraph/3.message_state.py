"""
LangGraph MessagesState 源码机制演示
===================================

本文件重点回答 6 个问题：

1. MessagesState 到底是什么？
2. 为什么节点返回一条消息后，旧消息不会被整个覆盖？
3. add_messages 如何追加新消息？
4. add_messages 如何根据 message.id 更新已有消息？
5. RemoveMessage 如何删除已有消息？
6. MessagesState 是否天然支持多轮会话记忆？

安装：
    uv add langgraph langchain

运行：
    uv run python messages_state_demo.py

说明：
    本示例不调用真实大模型，不需要 API Key。
"""

from typing import Any

from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


# ============================================================================
# 一、MessagesState 的本质
# ============================================================================
#
# MessagesState 并不是一个“自动调用大模型的聊天状态”，也不是一个普通 list。
# 它本质上是 LangGraph 预定义的 TypedDict，概念上近似：
#
#     class MessagesState(TypedDict):
#         messages: Annotated[list[AnyMessage], add_messages]
#
# 其中最关键的不是字段名 messages，而是它绑定的 reducer：add_messages。
#
# 当多个节点返回：
#
#     return {"messages": [new_message]}
#
# LangGraph 不会直接执行：
#
#     state["messages"] = [new_message]
#
# 而是近似执行：
#
#     state["messages"] = add_messages(
#         state["messages"],
#         [new_message],
#     )
#
# 所以消息默认会被合并，而不是把整个历史列表覆盖掉。


class ChatState(MessagesState):
    """
    MessagesState 可以被继承，并添加普通业务字段。

    messages：使用 add_messages reducer 合并。
    topic：普通状态字段，节点返回新值时直接覆盖旧值。
    step：普通状态字段，节点返回新值时直接覆盖旧值。
    """

    topic: str
    step: int


def print_messages(title: str, state: ChatState) -> None:
    """打印当前完整消息状态，方便观察每一步的变化。"""

    print(f"\n{'=' * 24} {title} {'=' * 24}")
    print(f"topic = {state.get('topic')!r}")
    print(f"step  = {state.get('step')!r}")

    for index, message in enumerate(state["messages"], start=1):
        print(
            f"[{index}] "
            f"type={message.type!r}, "
            f"class={type(message).__name__}, "
            f"id={message.id!r}, "
            f"content={message.content!r}"
        )


# ============================================================================
# 二、节点 1：用相同 ID 更新用户消息
# ============================================================================


def normalize_user_message(state: ChatState) -> dict[str, Any]:
    """
    清理用户输入前后的空格。

    初始输入中的用户消息 ID 是 user-1。
    当前节点返回的 HumanMessage 仍然使用 user-1。

    add_messages 发现新旧消息 ID 相同后，不会追加第二条用户消息，
    而是用新消息替换旧消息。
    """

    print_messages("进入 normalize_user_message", state)

    user_message = state["messages"][-1]

    # 使用相同 ID：更新已有消息。
    normalized_message = HumanMessage(
        content=str(user_message.content).strip(),
        id=user_message.id,
    )

    return {
        "messages": [normalized_message],
        "step": 1,
    }


# ============================================================================
# 三、节点 2：使用新 ID 追加 AI 消息
# ============================================================================


def append_ai_message(state: ChatState) -> dict[str, Any]:
    """
    返回一条 ID 从未出现过的 AIMessage。

    add_messages 找不到相同 ID，因此把它追加到消息列表尾部。
    """

    print_messages("进入 append_ai_message", state)

    return {
        "messages": [
            AIMessage(
                content="MessagesState 不是普通列表，它的 messages 字段绑定了 add_messages reducer。",
                id="assistant-1",
            )
        ],
        "step": 2,
    }


# ============================================================================
# 四、节点 3：使用相同 ID 修订 AI 消息
# ============================================================================


def revise_ai_message(state: ChatState) -> dict[str, Any]:
    """
    再次返回 assistant-1。

    因为 ID 与上一节点生成的 AI 消息相同，所以这是“修改”，不是“追加”。
    最终状态中只会保留一个 assistant-1。
    """

    print_messages("进入 revise_ai_message", state)

    return {
        "messages": [
            AIMessage(
                content=(
                    "MessagesState 的核心是 messages Channel 使用 add_messages reducer："
                    "新 ID 会追加，相同 ID 会更新。"
                ),
                id="assistant-1",
            )
        ],
        "step": 3,
    }


# ============================================================================
# 五、节点 4：追加一条临时消息
# ============================================================================


def append_temporary_message(state: ChatState) -> dict[str, Any]:
    """添加一条稍后会被删除的消息。"""

    print_messages("进入 append_temporary_message", state)

    return {
        "messages": [
            AIMessage(
                content="这是一条临时消息，下一节点会删除它。",
                id="temporary-1",
            )
        ],
        "step": 4,
    }


# ============================================================================
# 六、节点 5：通过 RemoveMessage 删除消息
# ============================================================================


def remove_temporary_message(state: ChatState) -> dict[str, Any]:
    """
    RemoveMessage 不是一条普通聊天消息，而是一条状态更新指令。

    add_messages 收到 RemoveMessage(id="temporary-1") 后，
    会从已有消息列表中删除对应 ID 的消息。
    """

    print_messages("进入 remove_temporary_message", state)

    return {
        "messages": [RemoveMessage(id="temporary-1")],
        "step": 5,
    }


# ============================================================================
# 七、构建状态图
# ============================================================================


def build_graph(*, with_memory: bool = False):
    builder = StateGraph(ChatState)

    builder.add_node("normalize_user_message", normalize_user_message)
    builder.add_node("append_ai_message", append_ai_message)
    builder.add_node("revise_ai_message", revise_ai_message)
    builder.add_node("append_temporary_message", append_temporary_message)
    builder.add_node("remove_temporary_message", remove_temporary_message)

    builder.add_edge(START, "normalize_user_message")
    builder.add_edge("normalize_user_message", "append_ai_message")
    builder.add_edge("append_ai_message", "revise_ai_message")
    builder.add_edge("revise_ai_message", "append_temporary_message")
    builder.add_edge("append_temporary_message", "remove_temporary_message")
    builder.add_edge("remove_temporary_message", END)

    # MessagesState 只负责定义消息字段及其合并规则。
    # 跨多次 invoke() 保存状态，需要额外配置 checkpointer。
    checkpointer = InMemorySaver() if with_memory else None
    return builder.compile(checkpointer=checkpointer)


# ============================================================================
# 八、演示 1：单次 invoke 中的追加、更新和删除
# ============================================================================


def demo_message_reducer() -> None:
    print("\n\n########## 演示 1：add_messages 的合并规则 ##########")

    graph = build_graph()

    result = graph.invoke(
        {
            # 这里故意使用字典，而不是手动创建 HumanMessage。
            # add_messages 会把兼容的消息字典反序列化成 LangChain Message 对象。
            "messages": [
                {
                    "type": "human",
                    "content": "  LangGraph 的 MessagesState 是普通列表吗？  ",
                    "id": "user-1",
                }
            ],
            "topic": "MessagesState",
            "step": 0,
        }
    )

    print_messages("演示 1 最终状态", result)

    print("\n最终应当只剩两条消息：")
    print("1. user-1：空格已经被清理，但没有重复追加。")
    print("2. assistant-1：内容已经被修订，但没有重复追加。")
    print("3. temporary-1：已经被 RemoveMessage 删除。")


# ============================================================================
# 九、演示 2：MessagesState 本身不等于跨调用记忆
# ============================================================================


def demo_without_checkpointer() -> None:
    print("\n\n########## 演示 2：没有 checkpointer ##########")

    graph = build_graph()

    first_result = graph.invoke(
        {
            "messages": [HumanMessage(content="第一次调用", id="call-1")],
            "topic": "无持久化",
            "step": 0,
        }
    )

    second_result = graph.invoke(
        {
            "messages": [HumanMessage(content="第二次调用", id="call-2")],
            "topic": "无持久化",
            "step": 0,
        }
    )

    print("\n第一次调用最终消息 ID：")
    print([message.id for message in first_result["messages"]])

    print("第二次调用最终消息 ID：")
    print([message.id for message in second_result["messages"]])

    print(
        "\n两个 invoke() 相互独立。MessagesState 定义的是状态结构和合并规则，"
        "并不会自动保存上一次调用的状态。"
    )


# ============================================================================
# 十、演示 3：checkpointer + thread_id 才能形成线程级短期记忆
# ============================================================================


def demo_with_checkpointer() -> None:
    print("\n\n########## 演示 3：使用 InMemorySaver ##########")

    graph = build_graph(with_memory=True)

    # 相同 thread_id 表示属于同一个会话线程。
    config = {"configurable": {"thread_id": "messages-state-demo"}}

    first_result = graph.invoke(
        {
            "messages": [HumanMessage(content="第一轮消息", id="round-1")],
            "topic": "有持久化",
            "step": 0,
        },
        config=config,
    )

    print("\n第一轮调用后的消息 ID：")
    print([message.id for message in first_result["messages"]])

    second_result = graph.invoke(
        {
            "messages": [HumanMessage(content="第二轮消息", id="round-2")],
            "topic": "有持久化",
            "step": 0,
        },
        config=config,
    )

    print("\n第二轮调用后的消息 ID：")
    print([message.id for message in second_result["messages"]])

    print(
        "\n第二轮状态中可以看到第一轮消息。原因不是 MessagesState 单独完成了记忆，"
        "而是 checkpointer 根据相同 thread_id 恢复旧状态，随后 add_messages 再合并新消息。"
    )


# ============================================================================
# 十一、程序入口
# ============================================================================


if __name__ == "__main__":
    demo_message_reducer()
    demo_without_checkpointer()
    demo_with_checkpointer()


"""
运行完本文件后，应掌握以下结论：

一、MessagesState 的定义

    MessagesState 预定义了 messages 字段，并为它绑定 add_messages reducer。

二、节点如何更新 messages

    节点应该返回增量：

        return {"messages": [new_message]}

    不需要手动把旧消息列表复制一遍。

三、add_messages 的主要规则

    1. 新消息没有重复 ID：追加。
    2. 新消息 ID 已存在：替换对应旧消息。
    3. 收到 RemoveMessage：删除对应 ID 的旧消息。
    4. 收到兼容的消息字典：转换成 LangChain Message 对象。

四、普通字段与 messages 字段的差异

    topic、step 等普通字段没有显式 reducer，通常采用覆盖语义。
    messages 绑定 add_messages，因此采用消息合并语义。

五、MessagesState 与记忆的关系

    MessagesState：定义当前状态中如何保存、合并消息。
    Checkpointer：负责跨多次图调用保存和恢复线程状态。
    thread_id：标识使用哪一份会话线程状态。

    所以：

        MessagesState != 自动跨轮次记忆

    更准确的关系是：

        MessagesState
        + Checkpointer
        + 相同 thread_id
        = 可以跨 invoke() 延续的线程级短期记忆
"""