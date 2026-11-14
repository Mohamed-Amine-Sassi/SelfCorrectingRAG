from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from Correction.RAGState import RAGState
from Correction.RetreiveNode import RetrieveNode
from Correction.GradeDocuments import GradeDocuments
from Correction.DecisionNodes import RouteAfterGrading ,RouteAfterAnswer 
from Correction.RewriteQuery import Rewrite
from Correction.GenerateNode import Generate
from Correction.GradeAnswer import GradeAnswer
from Correction.GenerateNode import NoAnswer

def BuildGraph(vectorstore) -> StateGraph[RAGState]:
    graph=StateGraph(RAGState)
    graph.add_node("Retrieve",RetrieveNode(vectorstore))  # vectorstore will be set later
    graph.add_node("GradeDocuments",GradeDocuments)
    graph.add_node("GradeAnswer",GradeAnswer)
    graph.add_node("RewriteQuery",Rewrite)
    graph.add_node("Generate",Generate)
    graph.add_node("NoAnswer", NoAnswer)

    graph.add_edge(START,"Retrieve")
    graph.add_edge("Retrieve","GradeDocuments")
    graph.add_conditional_edges("GradeDocuments",RouteAfterGrading,
        {"generate":"Generate","rewrite_query":"RewriteQuery","no_answer":"NoAnswer"} #missing no_answer
    )
    graph.add_edge("RewriteQuery","Retrieve")
    graph.add_edge("Generate","GradeAnswer")
    graph.add_conditional_edges("GradeAnswer",RouteAfterAnswer,{
        "generate":"Generate","rewrite_query":"RewriteQuery",END:END
    })
    graph.add_edge("NoAnswer", END)

    return graph.compile()

