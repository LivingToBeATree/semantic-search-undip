import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend


class FakeModel:
    def encode(self, text):
        return np.array([1.0, 0.0])


class FakeCollection:
    def __init__(self, distances):
        self.distances = distances

    def query(self, query_embeddings, n_results, include):
        metas = [
            {
                "title": f"Judul {i}",
                "creators": "A, B",
                "description": f"Abstrak {i}",
                "url": f"http://x/{i}.pdf",
                "date": "2021",
            }
            for i in range(len(self.distances))
        ]
        return {"metadatas": [metas], "distances": [self.distances]}

    def count(self):
        return len(self.distances)


def make_client(distances, summarize=None):
    # TestClient dipakai TANPA `with`, jadi lifespan (model + ChromaDB asli)
    # tidak dijalankan; komponen diganti dengan yang palsu.
    backend.app.state.model = FakeModel()
    backend.app.state.collection = FakeCollection(distances)
    backend.app.state.summarize = summarize or (lambda q, r: "Ringkasan [1]")
    return TestClient(backend.app)


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "kemiskinan", "max_threshold": 2.5},
        {"query": "kemiskinan", "max_threshold": -0.1},
        {},
    ],
)
def test_input_tidak_valid_ditolak(body):
    r = make_client([0.1]).post("/search", json=body)
    assert r.status_code == 422


def test_tidak_ada_jurnal_lolos_memberi_arahan():
    r = make_client([0.9, 0.8]).post(
        "/search", json={"query": "kemiskinan", "max_threshold": 0.5}
    )
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "no_results"
    assert body["count"] == 0 and body["results"] == []
    assert body["arahan"] and body["ringkasan"] is None


def test_hasil_lengkap_dengan_ringkasan():
    r = make_client([0.1, 0.2, 0.9]).post(
        "/search", json={"query": "kemiskinan", "max_threshold": 0.5}
    )
    body = r.json()
    assert body["status"] == "full"
    assert body["count"] == 2
    assert body["ringkasan"] == "Ringkasan [1]"
    assert [x["peringkat"] for x in body["results"]] == [1, 2]


def test_llm_gagal_fallback_ke_daftar_jurnal():
    def broken(q, r):
        raise backend.LLMError("down")

    r = make_client([0.1, 0.2], summarize=broken).post(
        "/search", json={"query": "kemiskinan"}
    )
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "list_only"
    assert body["ringkasan"] is None
    assert body["count"] == 2


def test_jarak_sama_dengan_threshold_masih_lolos():
    r = make_client([0.5]).post(
        "/search", json={"query": "x", "max_threshold": 0.5}
    )
    assert r.json()["count"] == 1


def test_llm_hanya_menerima_top_k():
    received = {}

    def spy(query, results):
        received["n"] = len(results)
        return "ok"

    make_client([0.01 * i for i in range(20)], summarize=spy).post(
        "/search", json={"query": "x", "max_threshold": 0.5}
    )
    assert received["n"] == backend.SUMMARY_TOP_K


def test_health():
    r = make_client([0.1, 0.2, 0.3]).get("/health")
    assert r.json() == {"status": "ok", "documents": 3}


def test_startup_gagal_jelas_bila_koleksi_tidak_ada(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DB_PATH", str(tmp_path / "kosong"))
    with pytest.raises(RuntimeError, match="was not found"):
        with TestClient(backend.app):
            pass
