"""更强的随机对照：更宽应变域与 n=7，全部用全枚举暴力核验三级最优。"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import InfeasibleError, invert_payload
from tests.test_solver import brute_force_optimal


def make_payload(lengths, strain_min, strain_max, windows_raw):
    return {
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


@pytest.mark.parametrize("seed", range(8))
def test_wider_bounds_n6(seed):
    rng = random.Random(100 + seed)
    n = 6
    lengths = [rng.randint(1, 5) for _ in range(n)]
    strain_min, strain_max = -2, 2  # 5^6 = 15625 个枚举点
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    windows_raw = []
    for s, e in pairs[:10]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        slack = rng.randint(0, 3)
        windows_raw.append((s, e, total - slack, total + slack))
    result = invert_payload(make_payload(lengths, strain_min, strain_max, windows_raw))
    expected = brute_force_optimal(lengths, strain_min, strain_max, windows_raw)
    assert expected is not None
    assert result["strains"] == expected[1]
    assert result["objectives"]["max_adjacent_diff"] == expected[0][0]
    assert result["objectives"]["sum_adjacent_abs_diff"] == expected[0][1]
    for check, (s, e, lo, hi) in zip(result["window_checks"], windows_raw):
        total = sum(lengths[i] * result["strains"][i] for i in range(s, e + 1))
        assert check["weighted_strain_sum"] == total
        assert lo <= total <= hi


@pytest.mark.parametrize("seed", range(3))
def test_n7_bruteforce(seed):
    rng = random.Random(200 + seed)
    n = 7
    lengths = [rng.randint(1, 3) for _ in range(n)]
    strain_min, strain_max = -1, 1  # 3^7 = 2187
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    windows_raw = []
    for s, e in pairs[:12]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        slack = rng.randint(0, 1)
        windows_raw.append((s, e, total - slack, total + slack))
    result = invert_payload(make_payload(lengths, strain_min, strain_max, windows_raw))
    expected = brute_force_optimal(lengths, strain_min, strain_max, windows_raw)
    assert expected is not None
    assert result["strains"] == expected[1]


@pytest.mark.parametrize("seed", range(4))
def test_infeasible_randomized_n6(seed):
    """向随机实例注入一条与其余窗口冲突的硬窗，应判不可行。"""
    rng = random.Random(300 + seed)
    n = 6
    lengths = [rng.randint(1, 4) for _ in range(n)]
    strain_min, strain_max = -3, 3
    truth = [rng.randint(strain_min, strain_max) for _ in range(n)]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:9]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows_raw.append((s, e, total, total))
    # 全长窗固定为真值，再令首段窗取一个与全长等式及其他等式矛盾的值。
    full = sum(lengths[i] * truth[i] for i in range(n))
    windows_raw.append((0, n - 1, full, full))
    windows_raw.append((0, 0, truth[0] + 20, truth[0] + 20))
    # 对照暴力确认确实不可行
    assert brute_force_optimal(lengths, strain_min, strain_max, windows_raw) is None
    with pytest.raises(InfeasibleError):
        invert_payload(make_payload(lengths, strain_min, strain_max, windows_raw))


def test_infeasible_max_size_is_fast():
    rng = random.Random(99)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    windows = []
    for s in range(n):
        for e in range(s, n):
            if len(windows) >= 19:
                break
            windows.append(
                {
                    "start_segment": s + 1,
                    "end_segment": e + 1,
                    "min_elongation": 0,
                    "max_elongation": 0,
                }
            )
    # 单段窗 0..0=0 与全长窗要求巨量伸长相矛盾（应变界 [-1e5,1e5] 内也达不到）。
    windows.append(
        {
            "start_segment": 1,
            "end_segment": n,
            "min_elongation": 10**18,
            "max_elongation": 10**18 + 1,
        }
    )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows[:20],
    }
    start = time.monotonic()
    with pytest.raises(InfeasibleError):
        invert_payload(payload)
    assert time.monotonic() - start < 5.0
