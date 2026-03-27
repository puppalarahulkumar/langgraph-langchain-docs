from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

model = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to French.",
    ),
    ("human", "I love programming."),
]
result=model.invoke(messages)
print(result)