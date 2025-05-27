from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOllama

code_llm = OllamaLLM(model="qwen3:4b", temperature=0.) # This is used for generating code.
evaluator_llm = ChatOllama(model="llama3.2:latest", temperature=0.) # This is used for evaluating either retrieved docs and generated responses.