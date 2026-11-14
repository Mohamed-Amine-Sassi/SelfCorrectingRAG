import hashlib
import os
import tempfile

import streamlit as st

from Correction.Graph import BuildGraph
from RagCore import build_vectorstore, run_baseline, run_with_trace

st.set_page_config(page_title="Self-Correcting RAG", layout="wide")
st.title("Self-Correcting RAG")

MAX_MB = 20


@st.cache_resource(show_spinner="Indexing the document...")
def load(pdf_bytes: bytes):
    digest = hashlib.md5(pdf_bytes).hexdigest()[:10]
    path = os.path.join(tempfile.gettempdir(), f"{digest}.pdf")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(pdf_bytes)
    vs = build_vectorstore(path)
    return vs, BuildGraph(vs)


# --- sidebar ---
st.sidebar.header("Document")
up = st.sidebar.file_uploader("Upload a PDF", type="pdf")
mode = st.sidebar.radio("Mode", ["Self-correcting", "Baseline","Compare"])

if up is None:
    st.info("Upload a PDF in the sidebar to start.")
    st.stop()
if up.size > MAX_MB * 1024 * 1024:
    st.error(f"PDF too large (max {MAX_MB} MB).")
    st.stop()

vs, graph = load(up.getvalue())

# --- main ---
question = st.text_input("Ask a question about the document")

if st.button("Ask") and question:
    answer, contexts = None, []
    try:
        if mode == "Compare":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline")
                with st.spinner("Running baseline..."):
                    b = run_baseline(vs, question)
                st.write(b["answer"])
                with st.expander("Sources"):
                    for c in b["contexts"]:
                        st.markdown(f"> {c}")
            with col2:
                st.subheader("Self-correcting")
                answer, contexts = None, []
                with st.status("Working...", expanded=True) as status:
                    for ev in run_with_trace(graph, question):
                        if ev["node"] == "done":
                            answer, contexts = ev["answer"], ev["contexts"]
                        else:
                            st.write(f"**{ev['node']}**: {ev.get('text', 'step finished')}")
                    status.update(label="Done", state="complete")
                st.write(answer)
                with st.expander("Sources"):
                    for c in contexts:
                        st.markdown(f"> {c}")
            st.stop()
        if mode == "Baseline":
            out = run_baseline(vs, question)
            answer, contexts = out["answer"], out["contexts"]
        
        else:
            with st.status("Working...", expanded=True) as status:
                for ev in run_with_trace(graph, question):
                    if ev["node"] == "done":
                        answer, contexts = ev["answer"], ev["contexts"]
                    else:
                        st.write(f"**{ev['node']}**: {ev.get('text', 'step finished')}")
                status.update(label="Done", state="complete")
    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}: {e}")
        st.stop()

    if answer is None:
        st.error("The pipeline returned nothing. Check run_with_trace in RagCore.py.")
        st.stop()

    st.subheader("Answer")
    st.write(answer)
    with st.expander("Sources"):
        for c in contexts:
            st.markdown(f"> {c}")