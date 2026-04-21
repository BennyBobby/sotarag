from taskiq_redis import RedisAsyncResultBackend, ListQueueBroker
from src.config import REDIS_URL

broker = ListQueueBroker(url=REDIS_URL).with_result_backend(
    RedisAsyncResultBackend(redis_url=REDIS_URL)
)


@broker.task
def ingest_paper_task(url: str, title: str, authors: list = None, published: str = None, abstract: str = None) -> dict:
    from src.engine.ingest import ingest_paper
    ingest_paper(url, title, authors=authors, published=published, abstract=abstract)
    return {"title": title, "status": "done"}
