
from typing import Annotated, Literal
from typing_extensions import TypedDict
import operator

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

# =========================
# MODEL
# =========================

model = init_chat_model(
    "openai:gpt-4.1-nano",
    temperature=0
)


# =========================
# TOOLS
# =========================

@tool
def addition(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


@tool
def subtraction(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b


tools = [addition, subtraction]

model_with_tools = model.bind_tools(tools)

tools_by_name = {
    tool.name: tool
    for tool in tools
}


# =========================
# STATE
# =========================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# =========================
# NODES
# =========================

def llm_call(state: AgentState):

    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant "
                    "that uses tools for arithmetic."
                )
            )
        ]
        + state["messages"]
    )

    return {
        "messages": [response]
    }


def tool_node(state: AgentState):

    result = []

    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:

        tool = tools_by_name[tool_call["name"]]

        observation = tool.invoke(
            tool_call["args"]
        )

        result.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": result
    }


# =========================
# ROUTER
# =========================

def should_continue(
    state: AgentState,
) -> Literal["tool_node", END]:

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END


# =========================
# GRAPH
# =========================

builder = StateGraph(AgentState)

builder.add_node(
    "llm_call",
    llm_call
)

builder.add_node(
    "tool_node",
    tool_node
)

builder.add_edge(
    START,
    "llm_call"
)

builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)

builder.add_edge(
    "tool_node",
    "llm_call"
)

agent = builder.compile()


# =========================
# TEST
# =========================

result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Add 3 and 4"
            )
        ]
    }
)

for message in result["messages"]:
    message.pretty_print()
