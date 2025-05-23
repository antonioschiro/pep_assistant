from dotenv import load_dotenv
load_dotenv()
import os
import re
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langsmith import traceable
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from datasets import dataset_name
from utils.graph import compiled_graph, llm, AnswerState
from groq import RateLimitError
from enum import Enum
# Langsmith configurations
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Defining prediction function
@traceable
def predict_rag_answer(example:dict) -> dict:
    try:
        response = compiled_graph.invoke({"code_question": example["question"], "iterations": 0, "answer_state": AnswerState.NOT_GENERATED})
        answer= response.get("generated_response")
        # Looking for llm markdown
        match = re.search(r"```(?:python)?\n(.+?)\n```", answer, re.DOTALL)
        answer = match.group(1).strip() if match else answer.strip()
        return {"answer": answer}
    except RateLimitError as ratelimit_except:
        print(f"Requests rate limit reached for this model {llm.model_name}:\nError details:\n{ratelimit_except}")
    except Exception as e:
        print(f"An unknown exception occurred:\n{e}")

@traceable
def predict_rag_context_response(example:dict) -> dict:
    try:
        response = compiled_graph.invoke({"code_question": example["question"], "iterations": 0, "answer_state": AnswerState.NOT_GENERATED})
        docs = response.get("retrieved_documents")
        context = '\n\n'.join(doc.page_content for doc in docs)
        answer = response.get("generated_response")
        # Looking for llm markdown
        match = re.search(r"```(?:python)?\n(.+?)\n```", answer, re.DOTALL)
        answer = match.group(1).strip() if match else answer.strip()
        return {"answer": answer, "context": context}
    except RateLimitError as ratelimit_except:
        print(f"Requests rate limit reached for this model {llm.model_name}:\nError details:\n{ratelimit_except}")
    except Exception as e:
        print(f"An unknown exception occurred:\n{e}")


# Evaluators
from langsmith.schemas import Run, Example
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Annotated
class GroundnessScore(BaseModel):
    score: Annotated[int, "Groundness score from 0 to 10"]
    reasoning: Annotated[str, "Reasoning behind the score."]

class AdherenceScore(BaseModel):
    score: Annotated[int, "Adherence score from 0 to 10"]
    reasoning: Annotated[str, "Reasoning behind the score."]

class StyleScore(BaseModel):
    score: Annotated[int, "Style score from 0 to 10"]
    reasoning: Annotated[str, "Reasoning behind the score."]

def adherence_evaluator(run:Run, example:Example) -> dict:
    '''
    This function is a custom evaluator for the adherence of the generated code.
    Input parameters:
    run: Run object containing the generated code and context.
    example: Example object containing the reference code and question.

    Returns:
    dict: a dictionary containing the adherence score, the reasoning and the key.
    '''
    prediction = run.outputs["answer"],
    reference = example.outputs["answer"]

    adherence_test_template = '''
        You are a strict and intelligent code evaluator. Your task is to assess the adherence of the predicted code with respect to the reference code.

        Definition of "adherence":

        - the predicted code has the same structure of the reference code;
        - the predicted code implements the same logical steps or algorythm of the reference code;
        - the predicted code produces the same output of the reference code given the same input.

        What must *not* influence the adherence:
        - variable names, even if they affect the code output;
        - minor differences in the structure that doesn't change the meaning (e.g. using a loop instead of a list comprehension)
        
        Rate adherence on a scale from 0 to 10:
            0: Incorrect logic or failure to implement the intended functionality.
            6: Mostly equivalent but with minor logical inconsistencies.
            10: Fully equivalent in logic and behaviour.
        Return the score calculated (0-10).


        Follow these steps:
        1. Compare the predicted code and the context line-by-line or block-by-block.
        2. Identify which parts of the code are aligned with the reference code.
        3. Reason explicitly about whether the predicted code is adherent to reference code.

        Here is the predicted code:
        {prediction}

        Here is the reference code:
        {reference}

        Return your analysis in the following **structured JSON format**:

        {{
        "score": <number from 0 to 10>,
        "reasoning": "<clear explanation of why this score was given, pointing to specific parts of the code and reference>"
        }}
        '''
    
    adherence_test_prompt = PromptTemplate.from_template(adherence_test_template)
    formatted_output_llm = llm.with_structured_output(AdherenceScore, method="json_mode")
    # Create the evaluation chain
    adherence_test_chain = (adherence_test_prompt | formatted_output_llm)
    response = adherence_test_chain.invoke({"prediction": prediction, "reference": reference})

    return {"score": response.score, "key": adherence_evaluator.__name__, "reasoning": response.reasoning}

# TO BE UPDATED
style_evaluator = LangChainStringEvaluator(
        "labeled_criteria",
        prepare_data= lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference": example.outputs["answer"],
            "input": example.inputs["question"],
        },
        config={
        "llm": llm,
        "criteria": {
            "style_score": '''
            Does the predicted code follow Python best practices and PEP standards?
            Rate the stile on a scale from 0 to 10:
             0 = the predicted code is not adherent to best practices and PEP standards at all
             6 = some best practices are adopted in the code but there are still some improvable code lines
             10 = the predicted code is fully adherent to Python best practices and PEP standards formatting.
            Return the score calculated (0-10).
            ''',
        },
        }
    )
    
def groundness_evaluator(run:Run, example:Example) -> dict:
    '''
    This function is a custom evaluator for the groundness of the generated code.
    Input parameters:
    run: Run object containing the generated code and context.
    example: Example object containing the reference code and question.

    Returns:
    dict: a dictionary containing the groundness score, the reasoning and the key.
    '''
    prediction = run.outputs["answer"]
    reference = run.outputs["context"]

    groundness_test_template = '''
        You are a strict and intelligent code evaluator. Your task is to assess how well the predicted code is grounded in the provided context.

        Definition:
        "Grounded" means the predicted code is inspired by, aligned with, or supported by any part of the context. The stronger the connection, the higher the score.

        Scoring scale:
         0: The code is completely ungrounded — no parts of it are supported or inspired by the context.
         6: The code is partially grounded — some parts match the context, but important suggestions or relevant improvements are missing.
         10: The code is fully grounded — the entire structure is consistent with, or clearly inspired by, the context.

        Follow these steps:
        1. Compare the predicted code and the context line-by-line or block-by-block.
        2. Identify which parts of the code are aligned with the context.
        3. Reason explicitly about whether the predicted code makes use of context effectively.

        Here is the predicted code:
        {prediction}

        Here is the context:
        {reference}

        Return your analysis in the following **structured JSON format**:

        {{
        "score": <number from 0 to 10>,
        "reasoning": "<clear explanation of why this score was given, pointing to specific parts of the code and context>"
        }}
        '''
    
    groundness_test_prompt = PromptTemplate.from_template(groundness_test_template)
    formatted_output_llm = llm.with_structured_output(GroundnessScore, method="json_mode")
    # Create the evaluation chain
    groundness_test_chain = (groundness_test_prompt | formatted_output_llm)
    response = groundness_test_chain.invoke({"prediction": prediction, "reference": reference})

    return {"score": response.score, "key": groundness_evaluator.__name__, "reasoning": response.reasoning}


# Run tests
# Defining an enum class to manage the tests type you want to execute
class TestType(Enum):
    CORRECTNESS = "Evaluate the correctness of the answers."
    GROUNDNESS = "Evaluate the groundness of the answers."
    
def execute_test_suite(test_type:list[TestType], dataset_name = dataset_name) -> None:
    try:
        for test in test_type:       
            if test == TestType.CORRECTNESS:
                print("Correctness tests launch started.")
                evaluate(
                    predict_rag_answer,
                    data = dataset_name,
                    evaluators = [adherence_evaluator],
                    experiment_prefix= "correctness_test",
                )
                print("Correctness tests terminated.")

            if test == TestType.GROUNDNESS:
                print("Groundness tests launch started.")
                evaluate(
                    predict_rag_context_response,
                    data = dataset_name,
                    evaluators = [groundness_evaluator],
                    experiment_prefix= "hallucination_test"
                )
                print("Groundness tests terminated.")
    except Exception as e:
        print(f"An unknown exception occurred:\n{e}")

execute_test_suite([TestType.CORRECTNESS])
