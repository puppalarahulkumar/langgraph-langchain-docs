from langchain.agents import create_agent
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


def get_weather(city:str)-> str:
    ''' Get the weather of the city'''
    return f"IT's always sunny in{city}"

agent=create_agent(
    model="gpt-4.1-nano",
    tools=[get_weather],
    system_prompt="You are a helpful assitant"
)

print(agent.invoke({"messages":[{"role":"user","content":"what is the weather in gunupur?"}]}))