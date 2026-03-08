import requests


def get_embeddings(text_chunks: list, model: str = "mxbai-embed-large"):
    embeddings = []

    for chunk in text_chunks:
        try:
            response = requests.post(
                "http://ollama:11434/api/embed",
                json={"model": model, "input": chunk},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if "embeddings" in data:
                embeddings.append(data["embeddings"][0])

        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
            continue

    return embeddings


if __name__ == "__main__":
    test_text = ["Paris is the capital of France"]
    vectors = get_embeddings(test_text)
    if vectors:
        print(f"Success! Vector size: {len(vectors[0])}")
