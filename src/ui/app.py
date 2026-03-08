import streamlit as st
from src.crawler.arxiv_client import search_arxiv

st.set_page_config(page_title="Sotarag", layout="wide")

st.title("SotaRAG")
st.markdown("### Leveraging RAG to keep you at the State of the Art")

theme = st.text_input(
    "What topic would you like to explore?",
    placeholder="e.g., Computer Vision, RAG, LLMs...",
)

col1, col2 = st.columns([1, 3])

with col1:
    max_docs = st.slider("Number of articles", min_value=1, max_value=10, value=5)
    search_button = st.button("Search")

if search_button and theme:
    with st.spinner(f"Searching for the latest papers on '{theme}'..."):
        articles = search_arxiv(query=theme, max_results=max_docs)

        if articles:
            st.success(f"Found {len(articles)} recent articles!")
            for article in articles:
                with st.expander(f"{article['title']}"):
                    st.write(f"**Published on:** {article['published']}")
                    st.write(f"**Abstract:** {article['summary']}")
                    st.link_button("Read PDF", article["pdf_url"])
        else:
            st.warning("No articles found for this topic.")
