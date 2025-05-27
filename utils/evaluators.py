from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain.prompts import PromptTemplate
from utils.prompts.templates import (retrieved_document_template, 
                            generate_response_template, 
                            hallucinations_template, 
                            completeness_template,
                            retry_generate_template,
                            )
from utils.models import evaluator_llm, code_llm

# Defining evaluators format outputs for Self-RAG
class RetrievedDocumentEvaluator(BaseModel):
    """Evaluates if a retrieved document is relevant for improving code."""
    score: bool = Field(description= "Relevance of document to the code. True or False.")

class HallucinationEvaluator(BaseModel):
    """Evaluates if response generated is affected by hallucinations or none."""
    score: bool = Field(description= '''Hallucination score. 
                        True if LLM is hallucinating, False if the answer is grounded.''')

class CompletenessEvaluator(BaseModel):
    score: bool = Field(description = '''Completeness score.
                        True if the response fully answers to the question. False otherwise.''')

# Defining parsers for structured outputs
document_evaluator_parser = PydanticOutputParser(pydantic_object=RetrievedDocumentEvaluator)
hallucination_parser = PydanticOutputParser(pydantic_object=HallucinationEvaluator)
completeness_parser = PydanticOutputParser(pydantic_object=CompletenessEvaluator)

# Creating prompts from templates
# Docs evaluation:
evaluation_doc_prompt = PromptTemplate.from_template(retrieved_document_template).partial(format_instructions=document_evaluator_parser.get_format_instructions())

# Response generation:
generation_prompt = PromptTemplate.from_template(generate_response_template)
retry_generation_prompt = PromptTemplate.from_template(generate_response_template + retry_generate_template)

# Response evaluation:
hallucination_prompt = PromptTemplate.from_template(hallucinations_template).partial(format_instructions=hallucination_parser.get_format_instructions())
completeness_prompt = PromptTemplate.from_template(completeness_template).partial(format_instructions=completeness_parser.get_format_instructions())


# Defining the evaluators
# Document evaluation chain
documents_evaluator_chain = (evaluation_doc_prompt
                            | evaluator_llm
                            | document_evaluator_parser
                            )

# Response generation chains
rag_pipeline = (generation_prompt 
                | code_llm
                | StrOutputParser()
                )
retry_rag_pipeline = (retry_generation_prompt # this chain is used in case of retry
                      | code_llm
                      | StrOutputParser()
                      )
# Response evaluation chains
# Hallucination chain
hallucinations_evaluator_chain = (hallucination_prompt
                                | evaluator_llm
                                | hallucination_parser
                                )
# Completeness chain
completeness_evaluator_chain = ( completeness_prompt
                                | evaluator_llm
                                | completeness_parser
                                )