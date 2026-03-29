import numpy as np

from turborag.adapters.compat import ExistingRAGAdapter
from turborag.adapters.langchain import TurboVectorStore
from turborag.types import ChunkRecord


class FakeEmbeddings:
    def embed_query(self, text: str):
        return self._vectorize(text)

    def embed_documents(self, texts):
        return [self._vectorize(text) for text in texts]

    @staticmethod
    def _vectorize(text: str):
        words = text.lower().split()
        return np.array(
            [
                float(words.count("apple")),
                float(words.count("banana")),
                float(words.count("finance")),
                float(len(words)),
            ],
            dtype=np.float32,
        )


def _record_id(item):
    if hasattr(item, "chunk_id"):
        return item.chunk_id
    metadata = getattr(item, "metadata", {})
    return metadata.get("chunk_id")


def test_existing_rag_adapter_uses_external_record_store():
    embedder = FakeEmbeddings()
    records = {
        "a": ChunkRecord(chunk_id="a", text="apple apple finance"),
        "b": ChunkRecord(chunk_id="b", text="banana banana"),
        "c": ChunkRecord(chunk_id="c", text="finance banana"),
    }
    ids = list(records)
    embeddings = np.vstack([embedder.embed_query(records[chunk_id].text) for chunk_id in ids])

    adapter = ExistingRAGAdapter.from_embeddings(
        embeddings=embeddings,
        ids=ids,
        query_embedder=embedder,
        fetch_records=lambda requested_ids: [records[chunk_id] for chunk_id in requested_ids if chunk_id in records],
        bits=4,
    )

    results = adapter.query("apple finance", k=2)
    assert [result.chunk_id for result in results] == ["a", "c"]


def test_turbo_vector_store_exposes_familiar_similarity_methods():
    embedder = FakeEmbeddings()
    store = TurboVectorStore.from_texts(
        texts=["apple finance update", "banana inventory", "finance banana"],
        embedding=embedder,
        ids=["a", "b", "c"],
        bits=4,
    )

    hits = store.similarity_search("apple", k=2)
    first = hits[0]
    assert _record_id(first) == "a"

    scored = store.similarity_search_with_score("banana", k=1)
    assert scored[0][1] > 0


def test_turbo_vector_store_can_add_texts_in_managed_mode():
    embedder = FakeEmbeddings()
    store = TurboVectorStore.from_texts(
        texts=["banana inventory"],
        embedding=embedder,
        ids=["b"],
        bits=4,
    )

    store.add_texts(["apple apple apple"], ids=["a"])
    hits = store.similarity_search("apple", k=1)
    assert _record_id(hits[0]) == "a"
