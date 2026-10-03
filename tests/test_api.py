"""API 层测试：健康检查、成功反演、422 字段错误、409 不可行、坏 JSON。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sample_payload():
    return {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -100, "max": 100},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 60, "max_elongation": 700},
            {"start_segment": 1, "end_segment": 1, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 2, "end_segment": 2, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 3, "end_segment": 3, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 4, "end_segment": 4, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 5, "end_segment": 5, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 6, "end_segment": 6, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 1, "end_segment": 3, "min_elongation": 0, "max_elongation": 2000},
        ],
    }


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_schema_hint_documents_fields():
    resp = client.get("/api/v1/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "window_checks" in body["success_response"]
    assert "INFEASIBLE" in body["errors"]


def test_invert_success_and_recompute():
    resp = client.post("/api/v1/invert", json=sample_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "OK"
    result = body["result"]
    assert len(result["strains"]) == 6
    assert all(-100 <= v <= 100 for v in result["strains"])
    # 全部回算和均可由提交数据复核且落在提交区间内。
    payload = sample_payload()
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        assert check["weighted_strain_sum"] == total
        assert win["min_elongation"] <= total <= win["max_elongation"]
        assert check["satisfied"] is True
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


def test_invert_invalid_fields_422():
    payload = sample_payload()
    payload["segment_lengths"] = [1, 2]  # 少于 6 段
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    assert any(f["field"] == "segment_lengths" for f in body["fields"])


def test_invert_infeasible_409():
    payload = sample_payload()
    payload["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_overlapping_exact_windows_parity_conflict_409():
    """重复的精确窗合法受理，但其整数奇偶矛盾必须返回 409 INFEASIBLE。"""
    full = {"start_segment": 1, "end_segment": 12,
            "min_elongation": 375, "max_elongation": 375}
    inside = {"start_segment": 5, "end_segment": 8,
              "min_elongation": 120, "max_elongation": 120}
    payload = {
        "segment_lengths": [2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2],
        "strain_bounds": {"min": 0, "max": 30},
        "windows": [dict(full) for _ in range(4)]
        + [dict(inside) for _ in range(4)],
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_nearby_feasible_exact_windows_200():
    """与矛盾仅差 1 的可行精确窗应正常求解，重复窗全部满足。"""
    full = {"start_segment": 1, "end_segment": 12,
            "min_elongation": 376, "max_elongation": 376}
    inside = {"start_segment": 5, "end_segment": 8,
              "min_elongation": 120, "max_elongation": 120}
    payload = {
        "segment_lengths": [2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2],
        "strain_bounds": {"min": 0, "max": 30},
        "windows": [dict(full) for _ in range(4)]
        + [dict(inside) for _ in range(4)],
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "OK"
    result = body["result"]
    assert len(result["window_checks"]) == 8
    assert all(c["satisfied"] for c in result["window_checks"])


def test_bad_json_400():
    resp = client.post(
        "/api/v1/invert",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_INPUT"
