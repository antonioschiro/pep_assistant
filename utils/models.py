from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
base_url = os.getenv("BASE_URL")
model_name = os.getenv("DOCKER_MODEL")

model_config = {"model": model_name,
                "temperature": 0.,
                "base_url": base_url
}

code_llm = ChatOpenAI(**model_config) # This generate the code.
evaluator_llm = ChatOpenAI(**model_config) # This evaluate docs and responses.