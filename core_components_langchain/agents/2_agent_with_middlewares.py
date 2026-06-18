
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, TodoListMiddleware, ToolRetryMiddleware

from langchain.tools import tool


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

backend=StateBackend()

agent = create_agent(
    model="gpt-4.1-nano",
    tools=[get_weather_for_location],
    middleware=[ FilesystemMiddleware(backend=backend),
                ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
                TodoListMiddleware(), 
                SubAgentMiddleware(backend=backend,
                                   subagents={"name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "model": "gpt-4.1-nano", "tools": [get_weather_for_location]}
                                   ), 
                ])
