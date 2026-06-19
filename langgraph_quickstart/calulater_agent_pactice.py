from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langchain.messages import HumanMessage, SystemMessage


model=init_chat_model("openai:gpt-4.1-nano",temperature=0)


@tool
def addition(a:float,b:float)->float:
    """to add the value a and b"""

    return a+b;

@tool
def substraction(a:float,b:float)->float:
    """to subtract the variable a from b"""
    return a-b;



model_with_tools=model.bind_tools([addition,substraction])
model_tool_names={tool.name:tool for tool in [addition,substraction]}


def structured_response(TypedDict):
    """to define the structured response format"""
    operation:str
    result:float

def tools_called(state:dict):

    if state.tool_calls:

        result=[]
        for tool_calls in state.tool_calls:
            if tool_calls["name"]=="addition":
                result.append(addition.invoke(tool_calls["args"]))
            elif tool_calls["name"]=="substraction":
                result.append(substraction.invoke(tool_calls["args"]))
        
        
def llm_call(state:dict):
    """to perform the llm call and return the response"""

    if state and state["tool_calls"]:
        result = model_with_tools.invoke([ SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]+ [HumanMessage(content=state["messages"])])
        return result
    elif state["result"]:
        model.invoke([SystemMessage(
                        content="You are a helpful assistant, for the arthemetic operations, the result is this {state['result']}, reply the user politely")])
    return "There's no message or state in it, please try again."


def should_continue(state:dict)->Literal["tool_calls","llm_calls",END]:

    if state["tool_calls"]:
        return "tool_calls"
    elif state["result"]:
        return "llm_calls"
    return END


agent=StateGraph();

agent.add_node("llm_calls",llm_call)
agent.add_node("tool_calls",tools_called)

agent.add_edge(START,"llm_calls")
agent.add_conditional_edges("llm_calls",should_continue,["tool_calls","llm_calls",END])
agent.add_edge("tool_calls","llm_calls")

