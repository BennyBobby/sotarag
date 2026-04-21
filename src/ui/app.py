import time
import requests
import streamlit as st
from src.config import API_URL

st.set_page_config(page_title="SotaRAG", layout="wide")

# Load persistent chat history from API on first load
if "chat_history" not in st.session_state:
    try:
        resp = requests.get(f"{API_URL}/chat/history", timeout=5)
        st.session_state.chat_history = resp.json() if resp.ok else []
    except Exception:
        st.session_state.chat_history = []

if "pending_tasks" not in st.session_state:
    st.session_state.pending_tasks = []


# --- Sidebar ---
with st.sidebar:
    st.title("Paper Library")
    with st.expander("Show Indexed Papers", expanded=False):
        try:
            papers = requests.get(f"{API_URL}/papers", timeout=5).json()
            if papers:
                for title, url in papers.items():
                    st.markdown(f"[{title[:40]}...]({url})")
            else:
                st.info("No papers indexed yet.")
        except Exception:
            st.warning("API unavailable.")

    st.divider()

    # Pending tasks progress tracker
    if st.session_state.pending_tasks:
        with st.status("Indexing papers...", expanded=True) as status_box:
            all_done = True
            for task in st.session_state.pending_tasks:
                try:
                    resp = requests.get(
                        f"{API_URL}/tasks/{task['task_id']}", timeout=5
                    ).json()
                    done = resp.get("status") == "done"
                except Exception:
                    done = False
                icon = "✅" if done else "⏳"
                st.write(f"{icon} {task['title'][:50]}")
                if not done:
                    all_done = False

            if all_done:
                st.session_state.pending_tasks = []
                status_box.update(label="All papers indexed!", state="complete")
                st.rerun()
            else:
                time.sleep(2)
                st.rerun()

    st.title("Find Research")
    theme = st.text_input("Topic", placeholder="e.g., Quantum Computing")
    max_docs = st.slider("Max articles", 1, 10, 5)

    if st.button("Search & Index Papers"):
        if theme:
            try:
                resp = requests.post(
                    f"{API_URL}/papers/search-and-ingest",
                    json={"topic": theme, "max_results": max_docs},
                    timeout=15,
                )
                if resp.status_code == 404:
                    st.warning("No articles found for this topic.")
                else:
                    st.session_state.pending_tasks = resp.json()["tasks"]
                    st.rerun()
            except Exception as e:
                st.error(f"API error: {e}")
        else:
            st.error("Please enter a topic.")

    st.divider()
    if st.button("Clear Chat History"):
        try:
            requests.delete(f"{API_URL}/chat/history", timeout=5)
            st.session_state.chat_history = []
            st.rerun()
        except Exception as e:
            st.error(f"Could not clear history: {e}")


# --- Chat ---
st.title("SotaRAG Chat")
st.markdown("Ask anything about the indexed papers.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do Transformers work?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:

            def stream_gen():
                with requests.post(
                    f"{API_URL}/chat/stream",
                    json={"question": prompt},
                    stream=True,
                    timeout=120,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(
                        chunk_size=None, decode_unicode=True
                    ):
                        if chunk:
                            yield chunk

            full_response = st.write_stream(stream_gen())
        except Exception as e:
            full_response = f"API error: {e}"
            st.markdown(full_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response}
    )
