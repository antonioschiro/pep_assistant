
from unittest.mock import Base
from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
import uvicorn
from utils.graph import compiled_graph, AnswerState
# Langsmith configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY") 

# Defining an agent class
class MultiAgentGraph:
    def __init__(self, compiled_graph = compiled_graph, max_retry:int = 3):
        self.graph = compiled_graph
        self.max_retry = max_retry

    async def ainvoke(self, input_code: str) -> str:
        """
        Invoke the graph with the provided input code.
        
        Args:
            input_code (str): The input data to be processed by the graph.
        
        Returns:
            str: The output code processed by the graph.
        """
        try:
            response = await self.graph.ainvoke({"code_question": input_code,
                                                "iterations": 0,
                                                "max_iterations": self.max_retry,
                                                "answer_state": AnswerState.NOT_GENERATED
                                                })
            answer = response.get("generated_response")
            if answer is None:
                answer = input_code
            return answer
        except Exception as e:
            print(f"An exception occurred:\n{e}")

python_agent = MultiAgentGraph(max_retry=0)

# API section
backend = FastAPI()

class InputModel(BaseModel):
    input_code: Annotated[str, Field(title= "The code to be reviewed.")]

@backend.post("/process", status_code = status.HTTP_200_OK)
async def pythonize(input_model: InputModel) -> dict:
    try:
        input_code = input_model.input_code
        response = await python_agent.ainvoke(input_code = input_code)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code = status.HTTP_500_INTERNAL_SERVER_ERROR, detail = str(e))