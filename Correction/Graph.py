from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from Correction.RAGState import RAGState
from RAGBase.Retrieve import Retrieve





graph=StateGraph(RAGState)
graph.add_node("Add-Op1",add1)
graph.add_node("Sub-Op1",Sub1)
graph.add_node("Router1",Sub1)

graph.add_edge(START,"Router1")
graph.add_conditional_edges("Router1",Decide1,
    {"Addition opperation1":"Add-Op1","Substruction opperation1":"Sub-Op1"}
)

graph.add_node("Add-Op2",add2)
graph.add_node("Sub-Op2",Sub2)
graph.add_node("Router2",Sub2)
graph.add_edge("Add-Op1","Router2")
graph.add_edge("Sub-Op1","Router2")
graph.add_conditional_edges("Router2",Decide2,
    {"Addition opperation2":"Add-Op2","Substruction opperation2":"Sub-Op2"}
)
graph.add_edge("Add-Op2",END)
graph.add_edge("Sub-Op2",END)

app=graph.compile()