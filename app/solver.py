"""海缆连续缆段整数微应变联合反演核心（仅使用 Python 标准库）。

输入
----
- 6..12 段按顺序排列的缆段长度（正整数）
- 统一应变闭区间 [strain_min, strain_max]（整数微应变）
- 8..20 个观测窗，每窗给出连续起止段（1 基，含端点）与累计伸长量闭区间

模型（所有比较与运算均为 Python 任意精度精确整数）
--------------------------------------------------
待求逐段应变为 x_0..x_{n-1}（整数微应变）。

观测窗 [s,e] 约束：Σ_{i=s..e} L_i·x_i ∈ [lo, hi]。
每段还须满足 strain_min ≤ x_i ≤ strain_max。

引入前缀和 P_0=0, P_i = Σ_{j<i} L_j·x_j，则
    P_{e+1} - P_s ∈ [lo, hi]
    P_{i+1} - P_i ∈ [L_i·strain_min, L_i·strain_max]
全部是 P 上的**差分约束**，其最长路闭包（Floyd-Warshall）：
- 出现正环 ⇒ 整体不可行（可靠的充分必要判定，针对松弛后的实值域；
  整数可行性仍以下层整数搜索为准）；
- 给出每个 P_i 的精确上下界，进而导出每段 x_i 的紧整数域：
      x_i ≥ ceil( D[i][i+1] / L_i )
      x_i ≤ floor( -D[i+1][i] / L_i )

优选准则（字典序三级，逐级不可放宽）
1. 最小化相邻段最大应变差 M = max_k |x_k - x_{k-1}|
   —— 在 x 空间就是差分约束 x_k - x_{k-1} ∈ [-M, M]，同样并入最长路闭包。
2. 在 1 的最优解中最小化 S = Σ_k |x_k - x_{k-1}|
   —— 辅助变量 e_k ≥ |Δ_k|，以 Σ e_k ≤ S 的线性松弛精确等价判定。
3. 在 1、2 的最优解中取应变序列 (x_0,...,x_{n-1}) 字典序最小者。

求解：
- 最长路闭包缩域（前缀和 P，以及并入 |Δ|≤M 的 x 闭包）；
- 通用整数界传播 **叠加 gcd/同余（CRT）传播**：后者负责界传播看不见的
  模矛盾（如加权和奇偶性互斥），从每条线性约束反解一元同余并合并，精确
  整数、只提前发现真实矛盾；
- MRV + 同余代表点上的折半回溯（决策变量优先于松弛变量）；
- 最小 M 以阈值二分得到；最小 S 与字典序最优序列以「现任最优解」做
  构造性下降（每次成功探测带回更优解，仅最后一次做不可行证明）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20


class ValidationErrors(ValueError):
    """输入字段错误（可一次性包含多个字段问题）。"""

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = fields
        super().__init__("; ".join(f"{f['field']}: {f['message']}" for f in fields))


class InfeasibleError(ValueError):
    """全部输入合法，但观测窗彼此冲突（含统一应变界），无可行解。"""


@dataclass(frozen=True)
class Window:
    start: int  # 0 基，含端点
    end: int  # 0 基，含端点
    lo: int  # 累计伸长量（长度加权应变和）闭区间下端
    hi: int  # 闭区间上端


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _err(fields: list[dict[str, str]], field: str, message: str) -> None:
    fields.append({"field": field, "message": message})


def _validate_lengths(payload: dict, fields: list[dict[str, str]]) -> list[int] | None:
    raw = payload.get("segment_lengths")
    if raw is None:
        _err(fields, "segment_lengths", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "segment_lengths", "must be an array of positive integers")
        return None
    lengths: list[int] = []
    for i, item in enumerate(raw):
        if not _is_int(item) or item < 1:
            _err(fields, f"segment_lengths[{i}]", "must be a positive integer")
        else:
            lengths.append(item)
    if not (MIN_SEGMENTS <= len(raw) <= MAX_SEGMENTS):
        _err(
            fields,
            "segment_lengths",
            f"must contain between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments"
            f" (got {len(raw)})",
        )
    return lengths if len(lengths) == len(raw) else None


def _validate_strain_bounds(
    payload: dict, fields: list[dict[str, str]]
) -> tuple[int, int] | None:
    raw = payload.get("strain_bounds")
    if raw is None:
        _err(fields, "strain_bounds", "field is required")
        return None
    if not isinstance(raw, dict):
        _err(fields, "strain_bounds", 'must be an object {"min": int, "max": int}')
        return None
    for key in raw:
        if key not in ("min", "max"):
            _err(fields, f"strain_bounds.{key}", "unknown field")
    lo = raw.get("min")
    hi = raw.get("max")
    ok = True
    if not _is_int(lo):
        _err(fields, "strain_bounds.min", "must be an integer")
        ok = False
    if not _is_int(hi):
        _err(fields, "strain_bounds.max", "must be an integer")
        ok = False
    if ok and lo > hi:
        _err(fields, "strain_bounds", "min must be less than or equal to max")
        return None
    return (lo, hi) if ok else None


def _validate_windows(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) ->list[Window] | None:
    raw = payload.get("windows")
    if raw is None:
        _err(fields, "windows", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "windows", "must be an array of observation windows")
        return None
    if not (MIN_WINDOWS <= len(raw) <= MAX_WINDOWS):
        _err(
            fields,
            "windows",
            f"must contain between {MIN_WINDOWS} and {MAX_WINDOWS} windows"
            f" (got {len(raw)})",
        )
    windows: list[Window] = []
    all_ok = True
    for i, item in enumerate(raw):
        prefix = f"windows[{i}]"
        if not isinstance(item, dict):
            _err(fields, prefix, "must be an object")
            all_ok = False
            continue
        ok = True
        for key in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
            if key not in item:
                _err(fields, f"{prefix}.{key}", "field is required")
                ok = False
            elif not _is_int(item[key]):
                _err(fields, f"{prefix}.{key}", "must be an integer")
                ok = False
        for key in item:
            if key not in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if not ok:
            all_ok = False
            continue
        s = item["start_segment"]
        e = item["end_segment"]
        lo = item["min_elongation"]
        hi = item["max_elongation"]
        if n is not None:
            if not (1 <= s <= n):
                _err(fields, f"{prefix}.start_segment", f"must be between 1 and {n}")
                ok = False
            if not (1 <= e <= n):
                _err(fields, f"{prefix}.end_segment", f"must be between 1 and {n}")
                ok = False
            if ok and s > e:
                _err(fields, f"{prefix}.start_segment", "must be <= end_segment")
                ok = False
        if lo > hi:
            _err(
                fields,
                f"{prefix}.min_elongation",
                "must be less than or equal to max_elongation",
            )
            ok = False
        if ok and n is not None:
            windows.append(Window(start=s - 1, end=e - 1, lo=lo, hi=hi))
        else:
            all_ok = False
    return windows if all_ok else None


_ALLOWED_TOP_LEVEL = {"segment_lengths", "strain_bounds", "windows"}


def validate(payload: object) -> tuple[list[int], int, int, list[Window]]:
    """校验并归一化输入；非法时抛 ValidationErrors。"""
    fields: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        raise ValidationErrors(
            [{"field": ".", "message": "request body must be a JSON object"}]
        )
    for key in payload:
        if key not in _ALLOWED_TOP_LEVEL:
            _err(fields, key, "unknown field")
    lengths = _validate_lengths(payload, fields)
    bounds = _validate_strain_bounds(payload, fields)
    n = len(lengths) if lengths is not None else None
    windows = _validate_windows(payload, n, fields)
    if fields:
        raise ValidationErrors(fields)
    assert lengths is not None and bounds is not None and windows is not None
    return lengths, bounds[0], bounds[1], windows


# --------------------------------------------------------------------------- #
# 差分约束最长路闭包
# --------------------------------------------------------------------------- #
def _longest_path_closure(
    node_count: int, edges: list[tuple[int, int, int]]
) -> list[list[int | None]] | None:
    """约束 P_v >= P_u + w 的最长路闭包；存在正环返回 None。

    D[i][j] = P_j - P_i 的最强（最大）下界；None 表示尚无路径（-∞）。
    D[i][i] > 0 即正环，矛盾。以 None 为哨兵可兼容任意精度输入。
    """
    d: list[list[int | None]] = [[None] * node_count for _ in range(node_count)]
    for i in range(node_count):
        d[i][i] = 0
    for u, v, w in edges:
        if d[u][v] is None or w > d[u][v]:
            d[u][v] = w
    for k in range(node_count):
        dk = d[k]
        for i in range(node_count):
            dik = d[i][k]
            if dik is None:
                continue
            di = d[i]
            for j in range(node_count):
                if dk[j] is None:
                    continue
                cand = dik + dk[j]
                if di[j] is None or cand > di[j]:
                    di[j] = cand
    for i in range(node_count):
        if d[i][i] is not None and d[i][i] > 0:
            return None
    return d


def _ceil_div(a: int, b: int) -> int:
    """数学上的 ceil(a / b)（b>0），纯整数。"""
    return -((-a) // b)


# --------------------------------------------------------------------------- #
# 模推理（gcd 消元 / 中国剩余同余合并）
# --------------------------------------------------------------------------- #
# 纯界传播只能看到一元闭区间 [lo,hi]；形如 2x_1+...+3x_7=375 与
# 2x_5+2x_6+3x_7+2x_8=120 的「奇偶性」矛盾（全缆要求 x_7 奇、内窗要求
# x_7 偶）不会缩窄任何一元界，回溯只能逐值枚举，无法在截止前裁决。
# 在界传播之外维护一元同余 x_j ≡ r (mod m)：把已钉死的变量并入常数、带
# 同余的变量写成 r+m·y，对剩余部分取 gcd，当约束区间内只剩唯一一个可达
# 余数目标时反解出 x_j 的同余，再用 CRT 合并。全部为精确整数推理，只可
# 能提前发现真实矛盾，绝不会误拒可行解（已用随机小实例全枚举对照验证）。
def _crt_merge(r1: int, m1: int, r2: int, m2: int) -> tuple[int, int] | None:
    """合并同余 x≡r1 (mod m1) 与 x≡r2 (mod m2)；不相容返回 None。

    返回 (r, lcm(m1,m2))，r 取模 lcm 的最小非负代表元。
    """
    g = math.gcd(m1, m2)
    if (r2 - r1) % g != 0:
        return None
    if g == m2:  # 第二个同余已被第一个蕴含
        return r1 % m1, m1
    if g == m1:  # 第一个同余已被第二个蕴含
        return r2 % m2, m2
    # x = r1 + m1·k，m1·k ≡ r2-r1 (mod m2)，且 g | (r2-r1)。
    k0 = (((r2 - r1) // g) * pow(m1 // g, -1, m2 // g)) % (m2 // g)
    lcm = m1 // g * m2
    return (r1 + m1 * k0) % lcm, lcm


def _modular_from_aggregate(
    j: int,
    wj: int,
    k_const: int,
    g_others: int,
    a: int,
    b: int,
    residues: list[tuple[int, int] | None],
    los: list[int],
    his: list[int],
) -> tuple | None:
    """由「除 x_j 外」聚合常数 K 与步长 gcd 判定 x_j 的同余。

    已钉死（lo==hi）的变量整段并入 K；已知同余 x_i=r_i+m_i·y 的变量并入
    代表元 r_i、步长 w_i·m_i 并入 gcd；其余自由变量以 w_i 入 gcd。
    令 g_o 为这些「其余项」步长的 gcd，则 Σw·x 关于 gcd(g_o,w_j) 只落在
    由 K 决定的余数上。返回：
    - None：无模信息；
    - ("conflict",)：[a,b] 内不存在可达余数目标（或细化余数无解），必矛盾；
    - ("residue", r, m)：x_j 必满足 x_j ≡ r (mod m)（可为对既有同余的细化）。
    """
    if wj == 0 or g_others == 0 or los[j] == his[j]:
        return None
    known = residues[j]
    if known is None:
        gw = math.gcd(g_others, abs(wj))
        first = a + ((k_const - a) % gw)
        if first > b:  # [a,b] 内没有 Σw·x 可取的余数 ⇒ 必矛盾
            return ("conflict",)
        if (b - first) // gw + 1 != 1:  # 区间含多个可达目标，无法钉死同余
            return None
        modulus = g_others // gw  # 与 wj/gw 互素
        if modulus == 1:
            return None
        q = (first - k_const) // gw
        r = (q * pow((wj // gw) % modulus, -1, modulus)) % modulus
        return ("residue", r, modulus)
    # 细化既有同余 x_j = r_j + m_j·y。
    r_j, m_j = known
    gw2 = math.gcd(g_others, abs(wj) * m_j)
    target = (k_const + wj * r_j) % gw2
    first = a + ((target - a) % gw2)
    if first > b:
        return ("conflict",)
    if (b - first) // gw2 + 1 != 1:
        return None
    rhs = first - k_const - wj * r_j
    if rhs % gw2 != 0:
        return ("conflict",)
    modulus = g_others // gw2  # 与 (wj·m_j/gw2) 互素
    if modulus == 1:
        return None
    coef = (wj * m_j // gw2) % modulus
    r_y = ((rhs // gw2) * pow(coef, -1, modulus)) % modulus
    new_modulus = m_j * modulus
    return ("residue", (r_j + m_j * r_y) % new_modulus, new_modulus)


def _prefix_closure(
    n: int, lengths: list[int], strain_min: int, strain_max: int,
    windows: list[Window],
) -> list[list[int]] | None:
    """前缀和 P 上的差分约束闭包（窗 + 每段应变界）。"""
    edges: list[tuple[int, int, int]] = []
    for i, length in enumerate(lengths):  # P_{i+1} - P_i ∈ [L·smin, L·smax]
        edges.append((i, i + 1, length * strain_min))
        edges.append((i + 1, i, -length * strain_max))
    for win in windows:  # P_{e+1} - P_s ∈ [lo, hi]
        edges.append((win.start, win.end + 1, win.lo))
        edges.append((win.end + 1, win.start, -win.hi))
    return _longest_path_closure(n + 1, edges)


def _x_bounds_from_prefix(
    n: int,
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    p_closure: list[list[int]],
) -> tuple[list[int], list[int]]:
    """由 P 的最强界导出每段 x_i 的紧整数域。"""
    los: list[int] = []
    his: list[int] = []
    for i, length in enumerate(lengths):
        forward = p_closure[i][i + 1]
        backward = p_closure[i + 1][i]
        # i -> i+1 与 i+1 -> i 均为直接边，闭包后必非 None。
        assert forward is not None and backward is not None
        lo = max(strain_min, _ceil_div(forward, length))
        hi = min(strain_max, (-backward) // length)
        los.append(lo)
        his.append(hi)
    return los, his


def _x_diff_closure(
    n: int, los: list[int], his: list[int], m: int
) -> tuple[list[int], list[int]] | None:
    """并入一元域与 |x_k-x_{k-1}|<=M 后的 x 最长路闭包（锚点节点 n）。"""
    anchor = n
    edges: list[tuple[int, int, int]] = []
    for i in range(n):  # x_i >= anchor + lo_i；anchor >= x_i - hi_i
        edges.append((anchor, i, los[i]))
        edges.append((i, anchor, -his[i]))
    for k in range(1, n):
        edges.append((k - 1, k, -m))  # x_k >= x_{k-1} - M
        edges.append((k, k - 1, -m))  # x_{k-1} >= x_k - M
    closure = _longest_path_closure(n + 1, edges)
    if closure is None:
        return None
    out_lo = [closure[anchor][i] if closure[anchor][i] is not None else los[i] for i in range(n)]
    out_hi = [-closure[i][anchor] if closure[i][anchor] is not None else his[i] for i in range(n)]
    for i in range(n):
        if out_lo[i] > out_hi[i]:
            return None
    return out_lo, out_hi


# --------------------------------------------------------------------------- #
# 通用整数界传播 + 回溯（x 空间）
# --------------------------------------------------------------------------- #
# 约束：sum_j w_j * var_j ∈ [lo, hi]，端点可为 None。
Constraint = tuple[tuple[tuple[int, int], ...], int | None, int | None]
# 预编译形式（一次顶层搜索只构建一次，递归节点共享，避免重复拆包/分配）：
# (变量下标元组, 系数元组, a, b)，a/b 仍可为 None。
CompiledConstraint = tuple[tuple[int, ...], tuple[int, ...], int | None, int | None]


def _compile_constraints(
    constraints: list[Constraint],
) -> list[CompiledConstraint]:
    return [
        (
            tuple(idx for idx, _ in terms),
            tuple(w for _, w in terms),
            a,
            b,
        )
        for terms, a, b in constraints
    ]


def _propagate(
    los: list[int],
    his: list[int],
    compiled: list[CompiledConstraint],
    residues: list[tuple[int, int] | None] | None = None,
) -> tuple[list[int], list[int], list[tuple[int, int] | None]] | None:
    """一元界传播 + gcd/同余传播到不动点；矛盾返回 None。

    residues[j] 为 x_j 已知的一元同余 (r, m)（x_j ≡ r mod m），可为 None。
    两类推理互相喂给：界传播钉死变量后，模消元的 gcd 更紧；同余反过来把
    域截到其可达代表点。约束数与项数都很小（≤20 窗、12 段加 11 个松弛
    变量），直接对全部约束做整轮到不动点，每条约束内部均为 O(k)。
    """
    los = list(los)
    his = list(his)
    residues = list(residues) if residues is not None else [None] * len(los)
    ceil_div = _ceil_div
    gcd = math.gcd
    aggregate = _modular_from_aggregate
    crt = _crt_merge

    while True:
        changed = False
        for idx_arr, w_arr, a, b in compiled:
            k = len(idx_arr)

            # ---- O(k) 一元界传播：先聚合整约束加权和的最小/最大值 ----
            sum_min = 0
            sum_max = 0
            tmin_arr = [0] * k
            tmax_arr = [0] * k
            for pos in range(k):
                j = idx_arr[pos]
                w = w_arr[pos]
                if w >= 0:
                    tmin = w * los[j]
                    tmax = w * his[j]
                else:
                    tmin = w * his[j]
                    tmax = w * los[j]
                tmin_arr[pos] = tmin
                tmax_arr[pos] = tmax
                sum_min += tmin
                sum_max += tmax
            for pos in range(k):
                j = idx_arr[pos]
                w = w_arr[pos]
                others_min = sum_min - tmin_arr[pos]
                others_max = sum_max - tmax_arr[pos]
                if b is not None:  # w*x_j <= b - others_min
                    c = b - others_min
                    if w > 0:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                    else:
                        v = ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                if a is not None:  # w*x_j >= a - others_max
                    c = a - others_max
                    if w > 0:
                        v = ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                    else:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                if los[j] > his[j]:
                    return None

            # ---- O(k) gcd/同余传播（仅双侧闭区间约束）----
            if a is not None and b is not None:
                total_const = 0
                const_arr = [0] * k
                step_arr = [0] * k
                for pos in range(k):
                    idx = idx_arr[pos]
                    w = w_arr[pos]
                    if los[idx] == his[idx]:
                        cval = w * los[idx]
                    else:
                        rem = residues[idx]
                        if rem is None:
                            cval = 0
                            step = abs(w)
                        else:
                            cval = w * rem[0]
                            step = abs(w * rem[1])
                        step_arr[pos] = step
                    const_arr[pos] = cval
                    total_const += cval
                # 即使整约束 gcd==1，剔除某一项后的 gcd 仍可能 >1
                # （目标场景：12 项权重 2 中夹一个 3，剔除 3 后 gcd=2），
                # 故逐变量用前后缀 gcd 判定，不能以整约束 gcd 提前跳过。
                pref = [0] * (k + 1)
                for pos in range(k):
                    pref[pos + 1] = gcd(pref[pos], step_arr[pos])
                suff = [0] * (k + 1)
                for pos in range(k - 1, -1, -1):
                    suff[pos] = gcd(suff[pos + 1], step_arr[pos])
                for pos in range(k):
                    g_others = gcd(pref[pos], suff[pos + 1])
                    if g_others == 1:
                        continue
                    j = idx_arr[pos]
                    info = aggregate(
                        j, w_arr[pos], total_const - const_arr[pos],
                        g_others, a, b, residues, los, his,
                    )
                    if info is None:
                        continue
                    if info[0] == "conflict":
                        return None
                    _, r, m = info
                    known = residues[j]
                    if known is None:
                        merged = (r, m)
                    else:
                        merged = crt(known[0], known[1], r, m)
                        if merged is None:  # 两条同余不可兼得 ⇒ 矛盾
                            return None
                    rr, mm = merged
                    first = los[j] + ((rr - los[j]) % mm)
                    if first > his[j]:
                        return None
                    last = his[j] - ((his[j] - rr) % mm)
                    if merged != known or first != los[j] or last != his[j]:
                        residues[j] = merged
                        los[j] = first
                        his[j] = last
                        changed = True

            # 界收紧后把两端夹到同余代表点上（域内不存在非代表点的解）。
            for pos in range(k):
                j = idx_arr[pos]
                rem = residues[j]
                if rem is not None:
                    r, m = rem
                    off_lo = (r - los[j]) % m
                    if off_lo:
                        new_lo = los[j] + off_lo
                        if new_lo > his[j]:
                            return None
                        los[j] = new_lo
                        changed = True
                    off_hi = (his[j] - r) % m
                    if off_hi:
                        his[j] -= off_hi
                        if los[j] > his[j]:
                            return None
                        changed = True
        if not changed:
            return los, his, residues


def _tight_counts(
    var_count: int, compiled: list[CompiledConstraint]
) -> list[int]:
    """每个变量出现在多少条精确等式（a==b）约束中。"""
    counts = [0] * var_count
    for idx_arr, _w_arr, a, b in compiled:
        if a is not None and a == b:
            for idx in idx_arr:
                counts[idx] += 1
    return counts


def _find_solution(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    decision_count: int | None = None,
) -> tuple[int, ...] | None:
    """对外入口：编译约束一次，然后进入共享该结构的回溯搜索。"""
    compiled = _compile_constraints(constraints)
    tight = _tight_counts(len(los), compiled)
    return _search(
        list(los), list(his), compiled, None,
        decision_count=decision_count, _tight=tight,
    )


def _search(
    los: list[int],
    his: list[int],
    compiled: list[CompiledConstraint],
    residues: list[tuple[int, int] | None] | None = None,
    *,
    decision_count: int | None = None,
    _tight: list[int] | None = None,
) -> tuple[int, ...] | None:
    """找一个可行赋值；无可行解返回 None（compiled 为全树共享的预编译约束）。

    变量序取 MRV（按同余代表点计数的最小域），但：
    - 优先在「真实决策变量」（下标 < decision_count）上分支——阶段 2 的松弛
      变量 e_k 随 x 钉死即可由传播确定，先分支只会放大搜索树；
    - MRV 同票时优先选择被更多精确等式夹住的变量，其取值后传播更强。
    取值在同余代表点序列上做域中点平衡二分，两个子域并集恰为整个剩余域，
    因此搜索完备且最坏分支深度为 O(log 域大小)；启发只影响速度，不影响可
    满足性判定与三级最优性。
    """
    if decision_count is None:
        decision_count = len(los)
    narrowed = _propagate(los, his, compiled, residues)
    if narrowed is None:
        return None
    los, his, residues = narrowed
    best = -1
    best_size = 0
    best_tight = -1
    # 真实决策变量优先；松弛变量只在所有决策变量钉死后才考虑。
    for priority in (True, False):
        for j in range(len(los)):
            if los[j] != his[j] and (j < decision_count) == priority:
                step = 1 if residues[j] is None else residues[j][1]
                size = (his[j] - los[j]) // step
                tight_count = _tight[j]
                if (
                    best == -1
                    or size < best_size
                    or (size == best_size and tight_count > best_tight)
                ):
                    best = j
                    best_size = size
                    best_tight = tight_count
        if best != -1:
            break
    if best == -1:
        return tuple(los)
    j = best
    step = 1 if residues[j] is None else residues[j][1]
    mid = (los[j] + his[j]) // 2

    def clamp_to_reps(value: int) -> int:
        if residues[j] is None:
            return max(los[j], min(his[j], value))
        r0 = residues[j][0]
        first = los[j] + ((r0 - los[j]) % step)
        last = his[j] - ((his[j] - r0) % step)
        if value <= first:
            return first
        if value >= last:
            return last
        return value - ((value - r0) % step)

    v = clamp_to_reps(mid)
    if v + step > his[j]:  # 只剩一个代表点可切：切到次顶代表点
        v = his[j] - step

    def recurse(new_lo: list[int], new_hi: list[int]) -> tuple[int, ...] | None:
        return _search(
            new_lo, new_hi, compiled, residues,
            decision_count=decision_count, _tight=_tight,
        )

    # 在同余代表点序列上做平衡二分：x_j<=v 与 x_j>=v+step。
    hi2 = list(his)
    hi2[j] = v
    result = recurse(los, hi2)
    if result is not None:
        return result
    lo3 = list(los)
    lo3[j] = v + step
    return recurse(lo3, his)


# --------------------------------------------------------------------------- #
# 反演
# --------------------------------------------------------------------------- #
def _window_constraints(n: int, lengths: list[int], windows: list[Window]) -> list[Constraint]:
    """构造窗约束；内容完全相同的重复窗只保留一条参与内部传播/搜索。

    重复观测窗是合法输入（结果回算仍逐窗列出），但两条一模一样的线性约束
    对可行性与最优性零增量，逐条处理只会重复计算，故在此去重。
    """
    cons: list[Constraint] = []
    seen: set[Constraint] = set()
    for win in windows:
        terms = tuple(
            (i, lengths[i]) for i in range(win.start, win.end + 1)
        )
        con = (terms, win.lo, win.hi)
        if con not in seen:
            seen.add(con)
            cons.append(con)
    return cons


def _diff_constraints(n: int, m: int) -> list[Constraint]:
    """|x_k - x_{k-1}| <= M（亦供通用传播，与差分闭包一致）。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _abs_sum_constraints(n: int, m: int, s_lo: int | None, s_hi: int | None) -> list[Constraint]:
    """e_k >= x_k-x_{k-1}、e_k >= x_{k-1}-x_k、0<=e_k<=M、Σe_k∈[s_lo,s_hi]。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        e_var = n + k - 1
        cons.append((((e_var, 1), (k, -1), (k - 1, 1)), 0, None))
        cons.append((((e_var, 1), (k, 1), (k - 1, -1)), 0, None))
        cons.append((((e_var, 1),), 0, m))
    cons.append((tuple((n + k - 1, 1) for k in range(1, n)), s_lo, s_hi))
    return cons


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows = validate(payload)
    n = len(lengths)
    window_cons = _window_constraints(n, lengths, windows)

    # 前缀和差分闭包：与 M 无关，只算一次。
    p_closure = _prefix_closure(n, lengths, strain_min, strain_max, windows)
    if p_closure is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    base_x_lo, base_x_hi = _x_bounds_from_prefix(
        n, lengths, strain_min, strain_max, p_closure
    )
    full_range = strain_max - strain_min

    def solve(m: int, s_hi: int | None = None) -> tuple[int, ...] | None:
        """|Δ|<=m（若给 s_hi 则 Σ|Δ|<=s_hi）；返回一个可行赋值或 None。"""
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m)
        if bounds is None:
            return None
        x_lo, x_hi = bounds
        constraints = list(window_cons)
        constraints += _diff_constraints(n, m)
        if s_hi is not None:
            x_lo = x_lo + [0] * (n - 1)
            x_hi = x_hi + [m] * (n - 1)
            constraints += _abs_sum_constraints(n, m, None, s_hi)
        return _find_solution(
            list(x_lo), list(x_hi), constraints,
            decision_count=n,
        )

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性，并取得一个可行解，
    # 用它实际达到的 M、S 作为随后二分的紧上界（避免在巨大空量程上空转）。
    base_sol = solve(full_range)
    if base_sol is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    init_m = max(
        abs(base_sol[i] - base_sol[i - 1]) for i in range(1, n)
    )
    init_s = sum(
        abs(base_sol[i] - base_sol[i - 1]) for i in range(1, n)
    )

    # 阶段 1：二分最小可行 M。SAT 探测返回的解其实际 M 往往远小于阈值，
    # 直接跳到它，显著压缩二分区间。
    m_lo, m_hi = 0, init_m
    last_sol: tuple[int, ...] | None = None
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        sol = solve(mid)
        if sol is not None:
            last_sol = sol
            achieved = max(abs(sol[i] - sol[i - 1]) for i in range(1, n))
            m_hi = min(mid, achieved)
        else:
            m_lo = mid + 1
    best_m = m_lo
    if last_sol is None or max(
        abs(last_sol[i] - last_sol[i - 1]) for i in range(1, n)
    ) > best_m:
        last_sol = solve(best_m)
    assert last_sol is not None

    # 阶段 2：最小化 S = Σ|Δ_k|。以阶段 1 的最优-M 解为现任解，反复在
    # 「现任 S − 1」阈值下做构造性搜索：每次成功都带回实际 S 更小的新解，
    # 只有最后一次（证明已到最小）是不可行搜索，避免阈值二分产生的多个
    # 深度不可行证明。
    def achieved_s(sol: tuple[int, ...]) -> int:
        return sum(abs(sol[i] - sol[i - 1]) for i in range(1, n))

    incumbent_s = last_sol
    cur_s = achieved_s(incumbent_s)
    guard = (n - 1) * best_m + 1  # 理论上界，防御性迭代上限
    while cur_s > 0 and guard:
        guard -= 1
        better = solve(best_m, s_hi=cur_s - 1)
        if better is None:
            break
        incumbent_s = better
        cur_s = achieved_s(better)
    best_s = cur_s

    # 阶段 3：逐段钉死字典序最小值（x_0, x_1, ... 顺序）。
    # 以阶段 2 得到的一个可行最优解（M、S 均最优）为「现任解」起点，对每段
    # 反复探测 x_var <= 现任值-1：成功则直接采用返回解（其 x_var 严格更小，
    # 一步跳到新的可行上界），失败则现任值即为该段字典序最优并钉死。每次
    # 成功探测都是构造性的、开销小；每段只有最后一次是不可行证明。
    bounds = _x_diff_closure(n, base_x_lo, base_x_hi, best_m)
    assert bounds is not None
    x_lo, x_hi = bounds
    x_lo = x_lo + [0] * (n - 1)
    x_hi = x_hi + [best_m] * (n - 1)
    constraints = list(window_cons)
    constraints += _diff_constraints(n, best_m)
    constraints += _abs_sum_constraints(n, best_m, best_s, best_s)
    base_compiled = _compile_constraints(constraints)
    base_tight = _tight_counts(len(x_lo), base_compiled)

    # 阶段 2 的现任解若未被任何成功的降 S 探测更新，仍是阶段 1 的长度 n
    # 解（不含松弛变量）；补一次 (M,S) 精确可行的完整赋值作为起点。
    if len(incumbent_s) != len(x_lo):
        full = _find_solution(
            list(x_lo), list(x_hi), constraints, decision_count=n
        )
        assert full is not None
        incumbent_s = full
    incumbent: tuple[int, ...] = incumbent_s  # 已是 (M,S) 最优的可行完整赋值
    strains: list[int] = []
    for var in range(n):
        while True:
            t = incumbent[var] - 1
            if t < x_lo[var]:
                break
            pinned: CompiledConstraint = ((var,), (1,), None, t)
            better = _search(
                list(x_lo), list(x_hi), base_compiled + [pinned],
                decision_count=n,
                _tight=base_tight,  # 钉压不是等式，不改变等式计数
            )
            if better is None:
                break
            incumbent = better  # 返回解的 x_var 必 <= t，严格更优
        value = incumbent[var]
        strains.append(value)
        x_lo[var] = x_hi[var] = value
        narrowed = _propagate(x_lo, x_hi, base_compiled)
        assert narrowed is not None  # 现任解满足钉压，不可能传播出矛盾
        x_lo, x_hi, _residues = narrowed

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return _build_result(lengths, windows, strains, diffs, best_m, best_s)


def _build_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
) -> dict:
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    window_checks = []
    for idx, win in enumerate(windows):
        weighted_sum = 0
        for i in range(win.start, win.end + 1):
            weighted_sum += lengths[i] * strains[i]
        total_length = prefix[win.end + 1] - prefix[win.start]
        window_checks.append(
            {
                "index": idx,
                "start_segment": win.start + 1,
                "end_segment": win.end + 1,
                "total_length": total_length,
                "min_elongation": win.lo,
                "max_elongation": win.hi,
                "weighted_strain_sum": weighted_sum,
                "satisfied": win.lo <= weighted_sum <= win.hi,
            }
        )
    # 两级平滑指标均由相邻差直接复核（重算以自证，不直接采用搜索内部值）。
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    return {
        "segment_count": len(lengths),
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": window_checks,
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }
