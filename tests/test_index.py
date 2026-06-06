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


def test_in_place_save_after_delete_and_reload(tmp_path):
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(40, 6)).astype(np.float32)
    ids = [f"id-{i}" for i in range(len(vectors))]

    index = TurboIndex(dim=6, bits=3, shard_size=10, seed=3)
    index.add(vectors, ids)
    index.save(str(tmp_path / "index"))

    # Re-open (memmap-backed), delete some, and save in place over itself.
    reopened = TurboIndex.open(str(tmp_path / "index"))
    removed = reopened.delete(["id-5", "id-6", "id-7"])
    assert removed == 3
    reopened.save(str(tmp_path / "index"))  # in-place atomic swap + memmap reload

    # The live object keeps working after the in-place swap...
    assert len(reopened) == 37
    assert "id-5" not in reopened
    assert reopened.search(vectors[3], k=1)[0][0] == "id-3"

    # ...and a fresh open reflects the deletion with no stale shard files.
    again = TurboIndex.open(str(tmp_path / "index"))
    assert len(again) == 37
    assert "id-5" not in again
    # No staging/backup dirs left behind.
    assert not (tmp_path / "index.saving").exists()
    assert not (tmp_path / "index.bak").exists()


def test_load_detects_truncated_shard(tmp_path):
    import pytest
    from turborag.exceptions import IndexConfigError

    rng = np.random.default_rng(99)
    vectors = rng.normal(size=(16, 8)).astype(np.float32)
    ids = [f"id-{i}" for i in range(len(vectors))]
    index = TurboIndex(dim=8, bits=3, seed=4)
    index.add(vectors, ids)
    index.save(str(tmp_path / "index"))

    # Corrupt: truncate a shard .bin file by one byte.
    shard_bin = next((tmp_path / "index" / "shards").glob("*.bin"))
    data = shard_bin.read_bytes()
    shard_bin.write_bytes(data[:-1])

    with pytest.raises(IndexConfigError):
        TurboIndex.open(str(tmp_path / "index"))
