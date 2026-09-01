"""Streamlit UI for the local Qwen3 research assistant (vLLM optimized)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agent.graph import (
    collect_tool_labels,
    last_ai_text,
    reset_graph,
    run_agent,
    run_agent_stream,
    to_langchain_messages,
)
from config.settings import settings
from loaders.document_loader import LoaderError
from rag.ingestion import clear_ingested_index, ingest_file, list_ingested_files
from rag.vector_store import VectorStoreError, reset_vector_store
from utils.errors import user_message
from utils.ollama import check_ollama


st.set_page_config(page_title="Local AI Research Assistant", page_icon="🔬", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("status", "")


def save_upload(uploaded) -> Path:
    destination = settings.upload_dir / uploaded.name
    destination.write_bytes(uploaded.getvalue())
    return destination


def sidebar() -> None:
    with st.sidebar:
        st.header("Local AI Research Assistant")
        
        # Show backend status
        if settings.llm_backend == "vllm":
            st.success(f"vLLM: {settings.vllm_model} (AWQ 4-bit)")
        else:
            ok, ollama_msg = check_ollama()
            if ok:
                st.success(ollama_msg)
            else:
                st.error(ollama_msg)

        st.caption(f"Backend: `{settings.llm_backend}`")
        if settings.llm_backend == "vllm":
            st.caption(f"Model: `{settings.vllm_model}`")
            st.caption(f"Quantization: `{settings.vllm_quantization}`")
            st.caption(f"Prefix Caching: `{settings.vllm_enable_prefix_caching}`")
        else:
            st.caption(f"Model: `{settings.ollama_model}`")
        st.caption(f"Embeddings: `{settings.embedding_model}` (ONNX: {settings.embedding_use_onnx})")
        st.caption(f"Search: `{settings.search_provider}`")
        st.caption(
            f"Chunks: size {settings.chunk_size}, overlap {settings.chunk_overlap}, k={settings.retrieval_k}"
        )

        st.subheader("Upload documents")
        files = st.file_uploader(
            "PDF / DOCX / TXT / HTML",
            type=["pdf", "docx", "txt", "html", "htm"],
            accept_multiple_files=True,
        )
        if files:
            for uploaded in files:
                if uploaded.size == 0:
                    st.warning(f"`{uploaded.name}` is empty.")
                    continue
                with st.status(f"Processing `{uploaded.name}`…", expanded=True) as status:
                    try:
                        path = save_upload(uploaded)
                        st.write("Extracting, splitting, embedding, and loading into Chroma…")
                        result = ingest_file(path)
                        status.update(label=result["message"], state="complete")
                    except (LoaderError, VectorStoreError) as exc:
                        status.update(label=user_message(exc), state="error")
                    except Exception as exc:  # noqa: BLE001
                        status.update(label=user_message(exc), state="error")

        ingested = list_ingested_files()
        st.subheader("Uploaded files")
        if ingested:
            for name in ingested:
                st.write(f"- {name}")
        else:
            st.caption("No documents in the vector store yet.")

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.status = ""
            st.rerun()

        if st.button("Clear documents"):
            reset_vector_store()
            clear_ingested_index()
            for leftover in settings.upload_dir.glob("*"):
                if leftover.is_file():
                    leftover.unlink()
            reset_graph()
            st.session_state.status = "Document store cleared."
            st.rerun()


def main() -> None:
    init_state()
    sidebar()
    st.title("Local AI Research Assistant")
    st.write(
        "Ask questions about uploaded documents, search the web, run calculations, "
        "or summarize papers. Qwen3 decides which tools to call."
    )

    if st.session_state.status:
        st.info(st.session_state.status)

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("tools"):
                st.caption("Used: " + ", ".join(item["tools"]))

    prompt = st.chat_input("Ask a question, request a summary, or search the web")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Qwen3 is thinking…", expanded=True) as status:
            try:
                history = to_langchain_messages(st.session_state.messages)
                
                # Use streaming for perceived latency improvement
                answer_placeholder = st.empty()
                full_answer = ""
                labels = []
                
                for chunk in run_agent_stream(history):
                    # chunk is a tuple of (node_name, node_output)
                    node_name, node_output = chunk
                    if node_name == "agent" and "messages" in node_output:
                        msg = node_output["messages"][-1]
                        if hasattr(msg, 'content') and msg.content:
                            full_answer += msg.content
                            answer_placeholder.markdown(full_answer + "▌")
                
                # Get final result
                result_messages = run_agent(history)
                labels = collect_tool_labels(result_messages)
                answer = last_ai_text(result_messages)
                
                if labels:
                    st.write("Tools: " + ", ".join(labels))
                    status.update(label="Used " + ", ".join(labels), state="running")
                status.update(label="Done", state="complete")
            except Exception as exc:  # noqa: BLE001
                answer = user_message(exc)
                labels = []
                status.update(label="Error", state="error")
        
        # Show final answer
        st.markdown(answer)
        if labels:
            st.caption("Used: " + ", ".join(labels))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": labels}
    )


main()