from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY




class Answer(BaseModel):
    summary:str
    confidence: float

@tool
def search(query: str)->str:
    """search for information"""
    return f"Results for {query}"

agent = create_agent("openai:gpt-4.1-nano", tools=[search],
                     system_prompt="You are a helpful assistant. Be concise and accurate.",
                     name="research_assistant",
                     response_format=Answer,
                     checkpointer=InMemorySaver(),
                     )

config = {"configurable": {"thread_id": str(uuid7())}}


result=agent.invoke({"messages":[{"role":"user","content":"what is apple?"}]},config=config)

print(result["structured_response"])

result=agent.invoke({"messages":[{"role":"user","content":"which topic you were writing before?"}]},config=config)
print(result["structured_response"])
