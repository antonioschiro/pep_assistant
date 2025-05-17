from langgraph.graph import END, StateGraph, START
from utils.nodes import *
# Initialize graph
graph = StateGraph(State)
# Adding nodes
graph.add_node("retrieve", retrieve)
graph.add_node("evaluate_docs", evaluate)
graph.add_node("generate", generate)
graph.add_node("check_response", check_response)
# Adding edges
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "evaluate_docs")
graph.add_conditional_edges("evaluate_docs", check_relevance,
                        {"generate": "generate",
                         "END": END})
graph.add_edge("generate", "check_response"),
graph.add_conditional_edges("check_response", route_answer,
                            {"generate": "generate",
                            "END": END,
                            "retrieve": "retrieve"})
# Compiling graph
compiled_graph = graph.compile()