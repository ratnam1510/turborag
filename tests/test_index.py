import numpy as np

from turborag import TurboIndex


def test_search_finds_self_match():
    rng = np.random.default_rng(21)
    vectors = rng.normal(size=(64, 8)).astype(np.float32)
    ids = [f"chunk-{i}" for i in range(len(vectors))]

    index = TurboIndex(dim=8, bits=4, seed=5)
    index.add(vectors, ids)

    results = index.search(vectors[0], k=5)
    assert results
    assert results[0][0] == "chunk-0"


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(8)
    vectors = rng.normal(size=(32, 6)).astype(np.float32)
    ids = [f"id-{i}" for i in range(len(vectors))]

    index = TurboIndex(dim=6, bits=3, shard_size=10, seed=17)
    index.add(vectors, ids)
    index.save(str(tmp_path / "index"))

    loaded = TurboIndex.open(str(tmp_path / "index"))
    assert len(loaded) == len(vectors)
    assert loaded.search(vectors[3], k=1)[0][0] == "id-3"
