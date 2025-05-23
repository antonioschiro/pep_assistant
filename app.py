
from dotenv import load_dotenv
load_dotenv()
import os
from groq import RateLimitError
from utils.graph import compiled_graph, AnswerState, llm
# Langsmith configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY") 

def pythonize_code(code:str) -> dict:
    """
    This function makes the code compliant to PEP standards.
        Input parameters:
        code (str): The code snippet to be edited.
        
        Returns a dict containing the fixed code. 
    """
    try:
        response = compiled_graph.invoke({"code_question": code, "iterations": 0, "answer_state": AnswerState.NOT_GENERATED})
        print(response) 
        answer = response.get("generated_response")
        return {"answer": answer}
    except KeyError:
        print(f"Answer not generated. More info:\n{e}")
    except RateLimitError as ratelimit_except:
        print(f"Requests rate limit reached for this model {llm.model_name}:\nError details:\n{ratelimit_except}")
    except Exception as e:
        print(f"An unknown exception occurred:\n{e}")

if __name__ == "__main__":
    input_code = input("Insert your code here: ")
    pythonize_code(input_code)