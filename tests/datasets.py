from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from data_collection import inputs, reference_responses

# Dataset info
dataset_name = "RAG_full_test"
dataset_id = "a219af8e-4db8-4276-a1c9-10bcf2d7caf0"
client = Client()

# Manage dataset functions
def create_dataset(dataset_name:str, dataset_description:str):
    dataset = client.create_dataset(
    dataset_name= dataset_name, 
    description= dataset_description
    )
    return dataset

def update_dataset(inputs:list, outputs:list, dataset_name:str = dataset_name, dataset_id:str = dataset_id): 
    try:
        dataset = client.read_dataset(dataset_id= dataset_id)
        # Check for duplicates
        # Update dataset
        client.create_examples(
            inputs = [{"question": q} for q in inputs],
            outputs = [{"answer": a} for a in outputs],
            )
    except LangSmithNotFoundError:
        print("No dataset found.")