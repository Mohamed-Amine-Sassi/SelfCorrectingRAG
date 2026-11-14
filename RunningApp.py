from RagCore import build_vectorstore, run_with_trace
from Correction.Graph import BuildGraph
vs = build_vectorstore("files/AIBook2.pdf")
graph = BuildGraph(vs)
for event in run_with_trace(graph, "What is Beta-VAE?"):
    print(event)