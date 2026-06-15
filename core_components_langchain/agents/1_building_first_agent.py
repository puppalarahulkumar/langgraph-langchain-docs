from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
from pydantic import BaseModel


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
                     response_format=Answer
                     )
result=agent.invoke({"messages":[{"role":"user","content":"summarize the ai trends"}]})
print(result)

#till name completed