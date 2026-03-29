import numpy as np

from turborag.compress import compressed_dot, dequantize_qjl, generate_rotation, quantize_qjl


def test_generate_rotation_is_orthogonal():
    rotation = generate_rotation(dim=16, seed=13)
    identity = rotation @ rotation.T
    np.testing.assert_allclose(identity, np.eye(16, dtype=np.float32), atol=1e-5)


def test_quantize_round_trip_is_reasonable():
    rng = np.random.default_rng(7)
    vectors = rng.uniform(-1.0, 1.0, size=(8, 12)).astype(np.float32)
    packed = quantize_qjl(vectors, bits=3)
    restored = dequantize_qjl(packed, dim=12, bits=3)
    max_error = np.abs(vectors - restored).max()
    assert max_error <= 0.3


def test_compressed_dot_tracks_dense_similarity():
    rng = np.random.default_rng(11)
    query = rng.uniform(-1.0, 1.0, size=(1, 10)).astype(np.float32)
    database = rng.uniform(-1.0, 1.0, size=(6, 10)).astype(np.float32)

    packed_query = quantize_qjl(query, bits=4)
    packed_database = quantize_qjl(database, bits=4)

    approx = compressed_dot(packed_query, packed_database, dim=10, bits=4)
    exact = database @ query[0]

    np.testing.assert_allclose(approx, exact, atol=0.35)
