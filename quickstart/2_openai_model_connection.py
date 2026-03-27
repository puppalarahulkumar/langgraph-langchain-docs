import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

model=init_chat_model("gpt-4.1-nano",max_tokens=25)

response = model.invoke("Why do parrots talk?" )
print(response.content)
