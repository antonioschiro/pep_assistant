from dotenv import load_dotenv
load_dotenv()
import os
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
        return {"answer": answer, "context": context}
    except RateLimitError as ratelimit_except:
        print(f"Requests rate limit reached for this model {llm.model_name}:\nError details:\n{ratelimit_except}")
    except Exception as e:
        print(f"An unknown exception occurred:\n{e}")

# Using two evaluators to estimate the response correctness 
correctness_evaluator = LangChainStringEvaluator(
        "labeled_criteria",
        prepare_data= lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference": example.outputs["answer"],
            "input": example.inputs["question"],
        },
        config={
        "llm": llm,
        "criteria": {
            "correctness": '''Is the predicted code equivalent to the reference provided? 
            Definition of equivalence:
            - the predicted code has the same structure of the reference code;
            - the predicted code implements the same logical steps or algorythm of the reference code;
            - the predicted code produces the same output of the reference code given the same input.

            What must *not* influence the correctness:
            - variable names, even if they affect the code output;
            - minor differences in the structure that doesn't change the meaning (e.g. using a loop instead of a list comprehension)
            
            Rate correctness on a scale from 0 to 10:
             0 = incorrect logic or failure to implement the intended functionality
             6 = Mostly equivalent but with minor logical inconsistencies
             10 = Fully equivalent in logic and behaviour
            Return the score calculated (0-10).
            ''',
        },
        }
    )

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

# Evaluator for hallucinations
hallucinations_evaluator = LangChainStringEvaluator(
        "labeled_criteria",
        prepare_data= lambda run, example: {
            "prediction": run.outputs["answer"],
            "reference": run.outputs["context"],
            "input": example.inputs["question"],
        },
        config={
        "llm": llm,
        "criteria": { 
            "grounded_score": '''
            Does the predicted code is supported by reference docs?
            Definition of grounding:
            - the code is grounding if it is consistent with, inspired by, or reasonably aligned with any part of the context — even if not a perfect match.
            Rate the grounding in reference docs from 0 to 10:
             0 = the predicted code is not grounded in reference docs at all.(e.g. all the changes in predicted code are not coming from reference docs);
             6 = the predicted code is partially grounded in reference docs;
             10 = the predicted code is fully grounded in reference docs.
            Return the score calculated (0-10).
            '''
        },
        }
)

# Run tests
# Defining an enum class to manage the tests type you want to execute
class TestType(Enum):
    CORRECTNESS = "Evaluate the correctness of the answers."
    GROUNDNESS = "Evaluate the groundness of the answers."
    
def execute_test_suite(test_type:list[TestType]) -> None:

    for test in test_type:       
        if test == TestType.CORRECTNESS:
            print("Correctness tests launch started.")
            correctness_experiment_results = evaluate(
                predict_rag_answer,
                data = dataset_name,
                evaluators = [correctness_evaluator, style_evaluator],
                experiment_prefix= "correctness_test",
            )
            print("Correctness tests terminated.")
        
        if test == TestType.GROUNDNESS:
            print("Groundness tests launch started.")
            hallucinations_experiment_results = evaluate(
                predict_rag_context_response,
                data = dataset_name,
                evaluators = [hallucinations_evaluator],
                experiment_prefix= "hallucination_test"
            )
            print("Groundness tests terminated.")