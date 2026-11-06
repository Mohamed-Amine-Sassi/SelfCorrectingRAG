from langgraph.graph import StateGraph, START, END
from Correction.RAGState import RAGState
MAX_RETRIES = 2
MAX_GENERATIONS = 3

def route_after_grading(state: RAGState):
    if state["contexts"]:
        return "generate"
    if state["retries"] >= MAX_RETRIES:
        return "no_answer"          # don't generate from nothing
    return "rewrite_query"

def route_after_answer(state: RAGState):
    if state["retries"] >= MAX_RETRIES or state["generations"] >= MAX_GENERATIONS:
        return END
    if not state["grounded"]:
        return "generate"
    if not state["useful"]:
        return "rewrite_query"
    return END