from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# =========================
# 1. 定义不同层次的状态
# =========================

class InputState(TypedDict):
  """调用 graph.invoke() 时允许传入的字段。"""
  user_input: str


class OutputState(TypedDict):
  """graph.invoke() 最终返回的字段。"""
  graph_output: str


class OverallState(TypedDict, total=False):
  """
  图的全局状态。

  total=False 表示从 Python 类型角度看，
  节点可以只返回其中一部分字段。
  """
  user_input: str
  normalized_name: str
  graph_output: str


class PrivateState(TypedDict):
  """
  图内部使用的私有状态。

  greeting 不需要暴露给图的调用者，
  只在 node_2 和 node_3 之间传递。
  """
  greeting: str


# =========================
# 2. 定义节点
# =========================

def node_1(state: InputState) -> dict:
  """
  输入只能读取 InputState 中的 user_input。

  但是节点可以向已经登记过的其他 Channel 写入数据，
  因此可以返回 normalized_name。
  """
  print("node_1 接收到：", state)

  return {
    "normalized_name": state["user_input"].strip()
  }


def node_2(state: OverallState) -> PrivateState:
  """
  读取全局状态 normalized_name，
  写入私有状态 greeting。
  """
  print("node_2 接收到：", state)

  return {
    "greeting": f"你好，{state['normalized_name']}",
  }


def node_3(state: PrivateState) -> OutputState:
  """
  这里非常关键：

  node_3 第一个参数被声明为 PrivateState。
  add_node() 会推断出这个输入 Schema，
  然后调用 _add_schema(PrivateState)。

  因此 greeting 会成为图中的一个 Channel。
  """
  print("node_3 接收到：", state)

  return {
    "graph_output": f"{state['greeting']}，欢迎学习 LangGraph！"
  }


# =========================
# 3. 创建状态图
# =========================

builder = StateGraph(
  OverallState,
  input_schema=InputState,
  output_schema=OutputState,
)

# 创建 StateGraph 时：
# OverallState、InputState、OutputState 已经被 _add_schema() 处理
print("创建 StateGraph 后：")
print(list(builder.channels.keys()))
print()

# =========================
# 4. 逐个添加节点，观察 Channel
# =========================

builder.add_node("node_1", node_1)

print("添加 node_1 后：")
print(list(builder.channels.keys()))
print()

builder.add_node("node_2", node_2)

print("添加 node_2 后：")
print(list(builder.channels.keys()))
print()

builder.add_node("node_3", node_3)

print("添加 node_3 后：")
print(list(builder.channels.keys()))
print()

# =========================
# 5. 添加边
# =========================

builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

# =========================
# 6. 编译并执行
# =========================

graph = builder.compile()

result = graph.invoke({
  "user_input": "  leonyangdev  "
})

print("最终返回结果：")
print(result)
