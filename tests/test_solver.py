"""求解器测试：用全枚举暴力结果对照三级最优解，并覆盖不可行/大整数情形。"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import (
    InfeasibleError,
    ValidationErrors,
    _ceil_div,
    invert_payload,
    validate,
)


def brute_force_optimal(lengths, strain_min, strain_max, windows):
    """全枚举参考实现：直接在 x 空间枚举，按 (M, S, tuple(x)) 取最小。"""
    n = len(lengths)
    best = None
    for xs in itertools.product(range(strain_min, strain_max + 1), repeat=n):
        ok = True
        for s, e, lo, hi in windows:
            total = sum(lengths[i] * xs[i] for i in range(s, e + 1))
            if not (lo <= total <= hi):
                ok = False
                break
        if not ok:
            continue
        diffs = [xs[i] - xs[i - 1] for i in range(1, n)]
        m = max(abs(d) for d in diffs)
        s_abs = sum(abs(d) for d in diffs)
        key = (m, s_abs, xs)
        if best is None or key < best[0]:
            best = (key, list(xs), diffs)
    return best


def assert_result_matches(result, expected, lengths, windows_raw, strain_min, strain_max):
    _, strains, diffs = expected
    assert result["strains"] == strains
    assert result["adjacent_diffs"] == diffs
    # 每段应变必须在统一闭区间内。
    assert all(strain_min <= v <= strain_max for v in result["strains"])
    # 一级、二级指标必须能由相邻差直接复核。
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)
    # 每个回算和必须能由提交的长度与应变直接复核并落入提交区间。
    for check, (s, e, lo, hi) in zip(result["window_checks"], windows_raw):
        recomputed = sum(lengths[i] * result["strains"][i] for i in range(s, e + 1))
        assert check["weighted_strain_sum"] == recomputed
        assert check["total_length"] == sum(lengths[s : e + 1])
        assert check["satisfied"] is True
        assert lo <= recomputed <= hi


def test_ceil_div_negative_denominator():
    for a in range(-20, 21):
        for b in (-3, -1, 2, 5):
            assert _ceil_div(a, b) == -((-a) // b)


def test_handcrafted_two_options():
    # n=6，窗迫使序列在 0 附近；检查回算结构与指标自洽。
    payload = {
        "segment_lengths": [1, 1, 1, 1, 1, 1],
        "strain_bounds": {"min": -10, "max": 10},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 0, "max_elongation": 0},
        ]
        + [
            {"start_segment": i, "end_segment": i, "min_elongation": 0, "max_elongation": 0}
            for i in range(1, 6)
        ]
        + [
            {"start_segment": 6, "end_segment": 6, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 1, "end_segment": 3, "min_elongation": 0, "max_elongation": 0},
        ],
    }
    result = invert_payload(payload)
    assert result["strains"] == [0, 0, 0, 0, 0, 0]
    assert result["objectives"] == {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0}


@pytest.mark.parametrize("seed", range(12))
def test_random_small_instances_match_bruteforce(seed):
    rng = random.Random(seed)
    n = 6
    lengths = [rng.randint(1, 4) for _ in range(n)]
    strain_min, strain_max = -2, 2
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    windows_raw = []
    seen = set()
    while len(windows_raw) < 10:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        if (s, e) in seen:
            continue
        seen.add((s, e))
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        slack = rng.randint(0, 2)
        windows_raw.append((s, e, total - slack, total + slack))
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": strain_min, "max": strain_max},
        "windows": [
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": lo,
                "max_elongation": hi,
            }
            for s, e, lo, hi in windows_raw
        ],
    }
    result = invert_payload(payload)
    expected = brute_force_optimal(lengths, strain_min, strain_max, windows_raw)
    assert expected is not None
    assert_result_matches(
        result, expected, lengths, windows_raw, strain_min, strain_max
    )


def test_infeasible_conflicting_windows():
    payload = {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": -100, "max": 100},
        "windows": [
            {"start_segment": 1, "end_segment": 1, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 1, "end_segment": 2, "min_elongation": 5, "max_elongation": 5},
            {"start_segment": 2, "end_segment": 2, "min_elongation": 5, "max_elongation": 5},
            {"start_segment": 2, "end_segment": 3, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 3, "end_segment": 3, "min_elongation": -5, "max_elongation": -5},
            {"start_segment": 3, "end_segment": 4, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 4, "end_segment": 6, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 1, "end_segment": 6, "min_elongation": 100, "max_elongation": 200},
        ],
    }
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


# --- 重叠精确窗的整数（奇偶）矛盾：必须在 3 秒截止前判不可行 ---
CABLE_LENGTHS = [2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2]


def _cable_payload(outer: int, inner: int) -> dict:
    full = {"start_segment": 1, "end_segment": 12,
            "min_elongation": outer, "max_elongation": outer}
    inside = {"start_segment": 5, "end_segment": 8,
              "min_elongation": inner, "max_elongation": inner}
    return {
        "segment_lengths": CABLE_LENGTHS,
        "strain_bounds": {"min": 0, "max": 30},
        # 两组允许重复的精确窗，各 4 个。
        "windows": [dict(full) for _ in range(4)]
        + [dict(inside) for _ in range(4)],
    }


def test_overlapping_exact_windows_parity_contradiction_fast():
    # 全长 375 要求第 7 段应变为奇；内部 120 要求其为偶。
    payload = _cable_payload(375, 120)
    start = time.monotonic()
    with pytest.raises(InfeasibleError):
        invert_payload(payload)
    assert time.monotonic() - start < 3.0


@pytest.mark.parametrize("outer,inner", [(376, 120), (375, 121)])
def test_nearby_feasible_exact_windows_still_optimal(outer, inner):
    # 与矛盾仅差 1 的可行精确窗不得被误拒，且三级最优与逐窗回算都成立。
    payload = _cable_payload(outer, inner)
    start = time.monotonic()
    result = invert_payload(payload)
    assert time.monotonic() - start < 3.0
    strains = result["strains"]
    assert len(strains) == 12
    assert all(0 <= x <= 30 for x in strains)
    # 8 个重复窗全部保留并各自满足。
    assert len(result["window_checks"]) == 8
    for check in result["window_checks"]:
        assert check["satisfied"] is True
        assert check["weighted_strain_sum"] == check["min_elongation"]
    # 两个精确总量直接复核。
    assert sum(CABLE_LENGTHS[i] * strains[i] for i in range(0, 12)) == outer
    assert sum(CABLE_LENGTHS[i] * strains[i] for i in range(4, 8)) == inner



def test_infeasible_due_to_strain_bounds():
    payload = {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": 0, "max": 1},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 1, "end_segment": 1, "min_elongation": 1, "max_elongation": 1},
            {"start_segment": 2, "end_segment": 2, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 3, "end_segment": 3, "min_elongation": 1, "max_elongation": 1},
            {"start_segment": 4, "end_segment": 4, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 5, "end_segment": 5, "min_elongation": 1, "max_elongation": 1},
            {"start_segment": 6, "end_segment": 6, "min_elongation": 0, "max_elongation": 0},
            {"start_segment": 1, "end_segment": 3, "min_elongation": 1, "max_elongation": 2},
        ],
    }
    # 单段窗和 = 应变；窗口 6..6 要求 -1..-1 之类的直接越界。
    payload["windows"][5] = {
        "start_segment": 6,
        "end_segment": 6,
        "min_elongation": -1,
        "max_elongation": -1,
    }
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_validation_collects_field_errors():
    payload = {
        "segment_lengths": [1, 1, 1],  # 段数不足
        "strain_bounds": {"min": 5, "max": 1},  # 上下界颠倒
        "windows": [
            {  # 唯一一窗，数量不足且字段非法
                "start_segment": 0,
                "end_segment": 9,
                "min_elongation": 3,
                "max_elongation": 2,
            }
        ],
    }
    with pytest.raises(ValidationErrors) as exc:
        validate(payload)
    fields = {e["field"] for e in exc.value.fields}
    assert "segment_lengths" in fields
    assert "strain_bounds" in fields
    assert "windows" in fields
    assert any("start_segment" in f for f in fields)
    assert any("min_elongation" in f for f in fields)


def test_unknown_and_type_fields_rejected():
    payload = {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": 0, "max": 1, "extra": 1},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 0, "max_elongation": 1}
        ]
        * 7
        + [
            {
                "start_segment": 1,
                "end_segment": 6,
                "min_elongation": 0,
                "max_elongation": 1,
                "bogus": 0,
            }
        ],
        "top_bogus": 1,
    }
    with pytest.raises(ValidationErrors) as exc:
        validate(payload)
    fields = {e["field"] for e in exc.value.fields}
    assert "top_bogus" in fields
    assert any("strain_bounds" in f for f in fields)
    assert any("bogus" in f for f in fields)


def test_bool_and_float_not_accepted_as_integer():
    payload = {
        "segment_lengths": [1, True, 1, 1, 1, 1.0],
        "strain_bounds": {"min": 0, "max": 1},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 0, "max_elongation": 1}
        ]
        * 8,
    }
    with pytest.raises(ValidationErrors):
        validate(payload)


def test_exact_large_integers_preserved():
    """大整数不得被浮点污染：真值与回算必须逐位相等。"""
    big = 10**30
    lengths = [big, big + 1, big + 2, big + 3, big + 4, big + 5]
    truth = [big % 7 - 3 for _ in range(6)]
    windows = []
    idx = 0
    for s in range(6):
        for e in range(s, 6):
            if idx >= 8:
                break
            total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
            windows.append(
                {"start_segment": s + 1, "end_segment": e + 1,
                 "min_elongation": total, "max_elongation": total}
            )
            idx += 1
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10**12, "max": 10**12},
        "windows": windows,
    }
    result = invert_payload(payload)
    assert result["strains"] == truth
    for check in result["window_checks"]:
        assert check["weighted_strain_sum"] == check["min_elongation"]
        assert check["weighted_strain_sum"] == check["max_elongation"]


def test_max_size_instance_performance():
    rng = random.Random(42)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    truth = [rng.randint(-5000, 5000) for _ in range(n)]
    windows = []
    starts = list(range(12))
    rng.shuffle(starts)
    for s0 in starts:
        e0 = rng.randint(s0, n - 1)
        total = sum(lengths[i] * truth[i] for i in range(s0, e0 + 1))
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 100, "max_elongation": total + 100}
        )
    # 补到 20 个窗
    while len(windows) < 20:
        s0 = rng.randint(0, n - 1)
        e0 = rng.randint(s0, n - 1)
        total = sum(lengths[i] * truth[i] for i in range(s0, e0 + 1))
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 50, "max_elongation": total + 50}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows,
    }
    start = time.monotonic()
    result = invert_payload(payload)
    elapsed = time.monotonic() - start
    assert elapsed < 10.0
    assert all(c["satisfied"] for c in result["window_checks"])
    assert len(result["strains"]) == 12


def test_deterministic():
    rng = random.Random(7)
    n = 6
    lengths = [rng.randint(1, 5) for _ in range(n)]
    truth = [rng.randint(-3, 3) for _ in range(n)]
    windows = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:9]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows.append(
            {"start_segment": s + 1, "end_segment": e + 1,
             "min_elongation": total - 1, "max_elongation": total + 1}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -5, "max": 5},
        "windows": windows,
    }
    r1 = invert_payload(payload)
    r2 = invert_payload(payload)
    assert r1 == r2


def test_singleton_strain_interval():
    """min==max 时所有段应变必须相同，两级指标均为 0。"""
    payload = {
        "segment_lengths": [7, 3, 9, 2, 5, 8],
        "strain_bounds": {"min": 4, "max": 4},
        "windows": [
            {"start_segment": 1, "end_segment": 6,
             "min_elongation": 4 * 34, "max_elongation": 4 * 34},
        ]
        + [
            {"start_segment": i, "end_segment": i,
             "min_elongation": 4 * [7, 3, 9, 2, 5, 8][i - 1],
             "max_elongation": 4 * [7, 3, 9, 2, 5, 8][i - 1]}
            for i in range(1, 7)
        ]
        + [
            {"start_segment": 2, "end_segment": 5,
             "min_elongation": 4 * 19, "max_elongation": 4 * 19},
        ],
    }
    result = invert_payload(payload)
    assert result["strains"] == [4] * 6
    assert result["objectives"] == {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0}


def test_extreme_strain_bounds_and_12_segments_20_windows():
    """12 段、20 窗（允许重复窗）、±10^9 域与 10^18 量级伸长量。"""
    n = 12
    lengths = [10**6 + i for i in range(n)]
    truth = [(-1) ** i * (10**9 - i) for i in range(n)]
    windows = []
    k = 0
    while len(windows) < 20:
        s = k % n
        e = min(n - 1, s + (k % 4))
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows.append(
            {"start_segment": s + 1, "end_segment": e + 1,
             "min_elongation": total, "max_elongation": total}
        )
        k += 1
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -(10**9), "max": 10**9},
        "windows": windows,
    }
    result = invert_payload(payload)
    assert len(result["strains"]) == 12
    assert all(c["satisfied"] for c in result["window_checks"])
    for check in result["window_checks"]:
        assert check["weighted_strain_sum"] == check["min_elongation"]
