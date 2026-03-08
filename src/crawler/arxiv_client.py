import arxiv
import os


def search_arxiv(query: str, max_results: int = 5):
    """
    Search for papers on ArXiv and return metadata.

    Args:
        query (str): The search keywords or theme.
        max_results (int): Number of papers to retrieve.

    Returns:
        list: A list of dictionaries containing paper details.
    """
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in search.results():
        papers.append(
            {
                "id": result.entry_id,
                "title": result.title,
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "published": result.published,
            }
        )
    return papers


if __name__ == "__main__":
    test_results = search_arxiv("Computer Vision", max_results=2)
    for doc in test_results:
        print(f"Found: {doc['title']} / {doc['pdf_url']}")
