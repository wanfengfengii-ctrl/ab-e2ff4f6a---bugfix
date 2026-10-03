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
   —— M 较小时在差分空间 (x0,d_k) 直接以 Σ|d_k| 的一元域收紧精确判定；
      M 很大时用辅助变量 e_k ≥ |Δ_k|、Σe_k≤S 的线性松弛（精确等价）。
3. 在 1、2 的最优解中取应变序列 (x_0,...,x_{n-1}) 字典序最小者。

求解：
- 最长路闭包缩域 + 整数界传播（含精确等式的丢番图余数类传播）+ MRV
  /折半回溯；
- 纯整数（奇偶/模）矛盾先在一组小素数有限域上做高斯消元预检截杀，
  覆盖差分闭包（仅实值域）无法识别的「系数偶、右端奇」一类冲突；
- 最小 M、最小 S 及字典序各值均利用「可行性关于阈值单调」二分钉死，
  可行试解带回的实际目标值直接收紧上界；
- 阶段 2/3 按域宽在「差分空间 / x 空间」间自适应选择；
- 重复观测窗在求解侧按 (段范围, 区间) 去重，结果仍逐窗回算。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20

# 模 p 可行性预检所用素数：整数可行 ⇒ 对每个素数模可行。
# 差分闭包只覆盖实值域，无法识别「全部系数偶而右端奇」这类格/奇偶矛盾；
# 在这些有限域上做高斯消元即可在线性时间级内截杀纯整数矛盾。
_PRESOLVE_PRIMES = (2, 3, 5, 7, 11)


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


# --------------------------------------------------------------------------- #
# 有限域（模素数）可行性预检
# --------------------------------------------------------------------------- #
def _mod_feasible(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    prime: int,
) -> bool:
    """线性等式约束组在 F_p 上是否有解。

    精确窗（lo==hi）与单点域变量（x_i 被钉为常量）都是精确整数等式。
    任何整数解模 p 后必为 F_p 解，故 F_p 无解 ⇒ 整数不可行（可靠的剪枝）。
    不等式约束无法直接搬上有限域，予以略去——有限域系统是整数问题的
    松弛，故其不可行仍是整数不可行的充分条件；反向情形交给下层搜索兜底。
    """
    p = prime
    var_count = len(los)
    # 每个等式用一行系数（常数项放在最后一列）：Σ a_j x_j ≡ c (mod p)。
    rows: list[list[int]] = []
    for terms, a, b in constraints:
        if a != b:
            continue  # 仅等式可直接搬上有限域
        row = [0] * (var_count + 1)
        for j, w in terms:
            row[j] = (row[j] + w) % p
        row[var_count] = a % p
        rows.append(row)
    for i in range(var_count):
        if los[i] == his[i]:
            row = [0] * (var_count + 1)
            row[i] = 1
            row[var_count] = los[i] % p
            rows.append(row)
    # 高斯-若尔当消元（按列选主元）。
    pivot_col = 0
    r = 0
    row_count = len(rows)
    while r < row_count and pivot_col < var_count:
        pivot = -1
        for i in range(r, row_count):
            if rows[i][pivot_col] % p != 0:
                pivot = i
                break
        if pivot == -1:
            pivot_col += 1
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = pow(rows[r][pivot_col], p - 2, p)
        rows[r] = [(v * inv) % p for v in rows[r]]
        for i in range(row_count):
            if i != r and rows[i][pivot_col] % p != 0:
                factor = rows[i][pivot_col]
                rows[i] = [
                    (a - factor * b) % p for a, b in zip(rows[i], rows[r])
                ]
        r += 1
        pivot_col += 1
    for i in range(r, row_count):
        coef_nonzero = any(rows[i][j] % p != 0 for j in range(var_count))
        if not coef_nonzero and rows[i][var_count] % p != 0:
            return False  # 0 ≡ 非零常数
    return True


def _mod_presolve(
    los: list[int], his: list[int], constraints: list[Constraint],
    primes: tuple[int, ...] = _PRESOLVE_PRIMES,
) -> bool:
    """在给定素数上做有限域预检；任一素数无解即整数不可行。"""
    return all(
        _mod_feasible(los, his, constraints, p) for p in primes
    )


# 搜索不动点处只跑最小的两个素数：每节点都要调用，需保持极廉价；
# 完整素数集仅用于搜索前的一次性预检。
_SEARCH_PRIMES = (2, 3)


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


def _propagate(
    los: list[int], his: list[int], constraints: list[Constraint],
    mod_check: bool = True,
) -> tuple[list[int], list[int]] | None:
    """对所有线性区间约束反复做一元界传播；矛盾返回 None。

    每轮对每条约束只扫描其项两遍（O(约束数·项数)）：第一遍求
    和的整体包络 [smin,smax]、自由变量包络与系数 gcd；第二遍用
    「总和减去本项」做一元界收紧。

    mod_check 控制不动点处是否再跑模素数高斯消元。它只是剪枝（绝不影响
    正确性：叶子点由各约束包络直接精确核对）。差分空间搜索的原始精确窗
    方程组已在进入搜索前做过完整素数集预检，且搜索中新增的只是差分/绝对值
    不等式，因此该路径关闭它以省去每节点的高斯消元开销。
    """
    los = list(los)
    his = list(his)
    while True:
        changed = False
        for terms, a, b in constraints:
            # 本约束在本轮开始时的变量划分快照：即使本轮把某变量收紧成单点，
            # 也要等下一轮再重算包络，避免 fixed/free 口径中途不一致。
            fixed = 0
            free_terms: list[tuple[int, int]] = []
            free_min = 0
            free_max = 0
            g = 0
            for j, w in terms:
                if los[j] == his[j]:
                    fixed += w * los[j]
                else:
                    free_terms.append((j, w))
                    g = gcd(g, abs(w))
                    if w > 0:
                        free_min += w * los[j]
                        free_max += w * his[j]
                    else:
                        free_min += w * his[j]
                        free_max += w * los[j]
            smin = free_min + fixed
            smax = free_max + fixed
            # gcd/包络快判：自由变量之和必为系数 gcd 的倍数，且必落在
            # 其当前一元界包络内；与残差区间不交即矛盾（单点变量也在此
            # 被核对，故后加入的精确约束与其冲突时不会漏检）。
            rlo = None if a is None else a - fixed
            rhi = None if b is None else b - fixed
            lo_bound = free_min if rlo is None else max(free_min, rlo)
            hi_bound = free_max if rhi is None else min(free_max, rhi)
            if lo_bound > hi_bound:
                return None
            if g > 1:
                first = lo_bound + ((-lo_bound) % g)
                if first > hi_bound:
                    return None
            # 精确等式（lo==hi）的丢番图余数传播：其余自由变量系数的
            # gcd g_j 决定 w_j·x_j 的余数类，据此把 x_j 的一元域收紧到
            # 某个同余类（区间约束的右端不唯一时一般无此强结论，故仅对
            # 等式做），显著压缩「精确观测窗」的回溯空间。
            # 用前缀/后缀 gcd 令每个 g_j 只花 O(1)（整体 O(自由项数)）。
            exact = a is not None and a == b
            nf = len(free_terms)
            if exact and nf >= 2:
                pref = [0] * (nf + 1)
                for t in range(nf):
                    pref[t + 1] = gcd(pref[t], abs(free_terms[t][1]))
                suffix_g = 0
                gjs: list[int] = [0] * nf
                for t in range(nf - 1, -1, -1):
                    gjs[t] = gcd(pref[t], suffix_g)
                    suffix_g = gcd(suffix_g, abs(free_terms[t][1]))
            else:
                gjs = [0] * nf
            for t, (j, w) in enumerate(free_terms):
                gj = gjs[t]
                if exact and gj > 1:
                    r = a - fixed  # w_j·x_j ≡ r (mod gj)
                    g0 = gcd(abs(w), gj)
                    if r % g0 != 0:
                        return None
                    modulus = gj // g0
                    if modulus > 1:
                        ww = (abs(w) // g0) % modulus
                        rr = ((r // g0) % modulus) * pow(ww, -1, modulus)
                        if w < 0:
                            rr = (-rr) % modulus
                        rr %= modulus
                        low_v = los[j] + (rr - los[j]) % modulus
                        high_v = his[j] - (his[j] - rr) % modulus
                        if low_v > high_v:
                            return None
                        if low_v > los[j]:
                            los[j] = low_v
                            changed = True
                        if high_v < his[j]:
                            his[j] = high_v
                            changed = True
                if w > 0:
                    others_min = smin - w * los[j]
                    others_max = smax - w * his[j]
                else:
                    others_min = smin - w * his[j]
                    others_max = smax - w * los[j]
                if b is not None:  # w*x_j <= b - others_min
                    c = b - others_min
                    if w > 0:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                    else:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                if a is not None:  # w*x_j >= a - others_max
                    c = a - others_max
                    if w > 0:
                        v = _ceil_div(c, w)
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
        if not changed:
            # 传播到不动点后做有限域预检：纯整数（奇偶/模）矛盾在一元界
            # 传播中不可见，但模素数高斯消元可立刻识别，避免回溯爆炸。
            # 每个搜索节点都到这里，故只跑最小的两个素数以保持廉价；
            # 已在上层做过完整素数集预检的路径可显式关闭。
            if mod_check and not _mod_presolve(
                los, his, constraints, _SEARCH_PRIMES
            ):
                return None
            return los, his


def _static_priority(
    var_count: int, constraints: list[Constraint]
) -> list[tuple[int, int]]:
    """每个变量的静态 MRV 平局优先级（只取决于约束结构，搜索中不变，算一次）。

    取包含该变量的约束中 (项数-等式加成, -该变量系数绝对值) 的最小值：
    优先处在短/精确约束且系数大的变量。默认值让不属于任何约束的变量排最后。
    """
    prio: list[tuple[int, int]] = [(10**9, 0)] * var_count
    for terms, a, b in constraints:
        exact_bonus = 1 if a == b else 0
        span = len(terms) - exact_bonus
        for j, w in terms:
            key = (span, -abs(w))
            if key < prio[j]:
                prio[j] = key
    return prio


def _choose_variable(
    los: list[int], his: list[int], priority: list[tuple[int, int]]
) -> int:
    """MRV：选当前域最小的自由变量；平局时按静态优先级。"""
    best = -1
    best_key: tuple[int, tuple[int, int]] | None = None
    for j in range(len(los)):
        if los[j] != his[j]:
            key = (his[j] - los[j], priority[j])
            if best_key is None or key < best_key:
                best = j
                best_key = key
    return best


def _search(
    los: list[int], his: list[int], constraints: list[Constraint],
    priority: list[tuple[int, int]] | None = None,
) -> tuple[int, ...] | None:
    """找一个可行赋值（MRV 变量序，域折半分支）；无可行解返回 None。"""
    if priority is None:
        priority = _static_priority(len(los), constraints)
    narrowed = _propagate(los, his, constraints)
    if narrowed is None:
        return None
    los, his = narrowed
    j = _choose_variable(los, his, priority)
    if j == -1:
        return tuple(los)
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _search(lo2, hi2, constraints, priority)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _search(lo3, hi3, constraints, priority)


# --------------------------------------------------------------------------- #
# 反演
# --------------------------------------------------------------------------- #
def _window_constraints(n: int, lengths: list[int], windows: list[Window]) -> list[Constraint]:
    cons: list[Constraint] = []
    for win in windows:
        terms = tuple(
            (i, lengths[i]) for i in range(win.start, win.end + 1)
        )
        cons.append((terms, win.lo, win.hi))
    return cons


def _diff_constraints(n: int, m: int) -> list[Constraint]:
    """|x_k - x_{k-1}| <= M（亦供通用传播，与差分闭包一致）。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _abs_sum_constraints(n: int, m: int, s_lo: int | None, s_hi: int | None) -> list[Constraint]:
    """e_k >= x_k-x_{k-1}、e_k >= x_{k-1}-x_k、0<=e_k<=M、Σe_k∈[s_lo,s_hi]。

    在最优 M 较大（差分空间 d_k∈[-M,M] 过宽）时，阶段 2/3 仍在 x 空间
    借助这些辅助变量做 S 的线性松弛判定（精确等价于 Σ|Δ|≤S）。
    """
    cons: list[Constraint] = []
    for k in range(1, n):
        e_var = n + k - 1
        cons.append((((e_var, 1), (k, -1), (k - 1, 1)), 0, None))
        cons.append((((e_var, 1), (k, 1), (k - 1, -1)), 0, None))
        cons.append((((e_var, 1),), 0, m))
    cons.append((tuple((n + k - 1, 1) for k in range(1, n)), s_lo, s_hi))
    return cons


def _abs_lb(low: int, high: int) -> int:
    """给定一元域 [low,high]，|d| 的紧下界（0 若域跨过 0）。"""
    if low == high:
        return abs(low)
    if low >= 0:
        return low
    if high <= 0:
        return -high
    return 0


def _dspace_constraints(
    n: int, lengths: list[int], windows: list[Window],
    x_lo: list[int], x_hi: list[int],
) -> list[Constraint]:
    """差分空间 (x_0, d_1,...,d_{n-1}) 上的基础线性约束。

    x_k = x_0 + Σ_{j=1..k} d_j；观测窗以
        W_len·x_0 + Σ_j (Σ_{i∈窗,i≥j} L_i)·d_j ∈ [lo,hi]
    表达。此变量下各 d_k 域仅为 [-M,M]（M 通常很小），
    传播与回溯都远比 x 空间紧致。
    """
    cons: list[Constraint] = []
    for k in range(1, n):  # x_k 的一元界翻译到 (x0, d1..dk)
        terms = tuple([(0, 1)] + [(j, 1) for j in range(1, k + 1)])
        cons.append((terms, x_lo[k], x_hi[k]))
    for win in windows:
        total_len = sum(lengths[i] for i in range(win.start, win.end + 1))
        terms: list[tuple[int, int]] = [(0, total_len)]
        for j in range(1, n):
            coef = sum(
                lengths[i]
                for i in range(win.start, win.end + 1)
                if i >= j
            )
            if coef:
                terms.append((j, coef))
        cons.append((tuple(terms), win.lo, win.hi))
    return cons


def _dspace_search(
    los: list[int],
    his: list[int],
    constraints: list[Constraint],
    s_lo: int | None,
    s_hi: int | None,
    priority: list[tuple[int, int]] | None = None,
) -> tuple[int, ...] | None:
    """差分空间搜索；s_lo/s_hi 给出 Σ|d_k| 的闭区间（可缺省一端）。

    线性界传播与「Σ|d| 对每个 d 域的收紧」联立求不动点：
        |d_k| <= (s_hi - Σ_{j≠k}|d_j|_lb)   （上界收紧）
        |d_k| >= (s_lo - Σ_{j≠k}|d_j|_ub)   （符号已定时下界收紧）
    它会随逐段钉死不断变紧，无需求助 e_k 辅助变量，保持传播紧致。
    """
    if priority is None:
        priority = _static_priority(len(los), constraints)
    active = s_lo is not None or s_hi is not None
    again = True
    while again:
        # 差分空间只是对 x 做了可逆整数线性变换，精确窗方程的格一致性已由
        # 进入阶段 2 前的完整素数集预检覆盖；此处新增的仅为差分/绝对值
        # 不等式。故关闭每节点的模高斯消元（叶子仍由各约束包络精确核对）。
        narrowed = _propagate(los, his, constraints, mod_check=False)
        if narrowed is None:
            return None
        los, his = narrowed
        again = False
        if active:
            edge_lb = [0] * len(los)
            edge_ub = [0] * len(los)
            for k in range(1, len(los)):
                edge_lb[k] = _abs_lb(los[k], his[k])
                edge_ub[k] = max(abs(los[k]), abs(his[k]))
            total_lb = sum(edge_lb)
            total_ub = sum(edge_ub)
            if s_hi is not None and total_lb > s_hi:
                return None
            if s_lo is not None and total_ub < s_lo:
                return None
            for k in range(1, len(los)):
                if s_hi is not None:
                    cap = s_hi - (total_lb - edge_lb[k])
                    if cap < 0:
                        return None
                    new_lo = max(los[k], -cap)
                    new_hi = min(his[k], cap)
                    if new_lo > new_hi:
                        return None
                    if new_lo > los[k]:
                        los[k] = new_lo
                        again = True
                    if new_hi < his[k]:
                        his[k] = new_hi
                        again = True
                if s_lo is not None:
                    need = s_lo - (total_ub - edge_ub[k])
                    if need > 0:
                        # |d_k| >= need；跨 0 域无法用一元界表达，仅在符号
                        # 已定时收紧，否则交给后续分支。
                        if los[k] >= 0:
                            if los[k] < need:
                                los[k] = need
                                again = True
                        elif his[k] <= 0:
                            if his[k] > -need:
                                his[k] = -need
                                again = True
                        if los[k] > his[k]:
                            return None
    j = _choose_variable(los, his, priority)
    if j == -1:
        return tuple(los)
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _dspace_search(lo2, hi2, constraints, s_lo, s_hi, priority)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _dspace_search(lo3, hi3, constraints, s_lo, s_hi, priority)


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows = validate(payload)
    n = len(lengths)
    # 重复观测窗是合法输入，但产生逐字相同的约束；求解侧去重可避免把
    # 传播/消元开销成倍放大。结果回算仍用原始 windows（含每个重复窗）。
    unique_windows: list[Window] = []
    seen_windows: set[Window] = set()
    for win in windows:
        if win not in seen_windows:
            seen_windows.add(win)
            unique_windows.append(win)
    window_cons = _window_constraints(n, lengths, unique_windows)

    # 前缀和差分闭包：与 M 无关，只算一次。
    p_closure = _prefix_closure(n, lengths, strain_min, strain_max, unique_windows)
    if p_closure is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    base_x_lo, base_x_hi = _x_bounds_from_prefix(
        n, lengths, strain_min, strain_max, p_closure
    )
    # 有限域预检：在任何搜索之前截杀纯整数矛盾（如精确窗模 2 奇偶冲突）。
    if not _mod_presolve(base_x_lo, base_x_hi, window_cons):
        raise InfeasibleError("observation windows are mutually inconsistent")
    full_range = strain_max - strain_min

    def feasible_m(m: int) -> bool:
        """|Δ|<=m 下是否存在整数解（x 空间，用于 M 的二分）。"""
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m)
        if bounds is None:
            return False
        x_lo, x_hi = bounds
        constraints = list(window_cons)
        constraints += _diff_constraints(n, m)
        return _search(list(x_lo), list(x_hi), constraints) is not None

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性。
    if not feasible_m(full_range):
        raise InfeasibleError("observation windows are mutually inconsistent")

    # 阶段 1：二分最小可行 M。
    m_lo, m_hi = 0, full_range
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if feasible_m(mid):
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2/3 取最优 M 下的差分闭包域，随后自适应选择求解变量空间。
    bounds = _x_diff_closure(n, base_x_lo, base_x_hi, best_m)
    assert bounds is not None
    x_lo, x_hi = bounds

    # 阶段 2/3 的变量空间自适应选择：
    #  - 最优 M 较小时，差分 d_k∈[-M,M] 极窄，用差分空间（无 e 辅助变量）；
    #  - 最优 M 很大时，差分会把本已被闭包压窄的 x 域重新放宽到 2M+1，
    #    仍在 x 空间用 e_k≥|Δ| 的线性松弛。判据为 d 域宽不超过最宽 x 域。
    max_x_width = max(x_hi[i] - x_lo[i] for i in range(n))
    use_dspace = (2 * best_m + 1) <= max_x_width

    if use_dspace:
        strains = _optimize_dspace(
            n, lengths, unique_windows, x_lo, x_hi, best_m,
        )
    else:
        strains = _optimize_xspace(
            n, window_cons, x_lo, x_hi, best_m,
        )

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    best_s = sum(abs(d) for d in diffs)
    return _build_result(lengths, windows, strains, diffs, best_m, best_s)


def _optimize_dspace(
    n: int, lengths: list[int], unique_windows: list[Window],
    x_lo: list[int], x_hi: list[int], best_m: int,
) -> list[int]:
    """阶段 2/3：在差分空间 (x0,d1,..) 内最小化 Σ|d| 并钉死字典序。"""
    d_base = _dspace_constraints(n, lengths, unique_windows, x_lo, x_hi)

    def d_domains() -> tuple[list[int], list[int]]:
        d_lo = [x_lo[0]] + [-best_m] * (n - 1)
        d_hi = [x_hi[0]] + [best_m] * (n - 1)
        return d_lo, d_hi

    # 一次性的完整素数集预检：阶段 1 可行 ⇒ 此处必有模解，断言仅作防御。
    d0_lo, d0_hi = d_domains()
    if not _mod_presolve(d0_lo, d0_hi, d_base):
        raise InfeasibleError("observation windows are mutually inconsistent")

    def feasible_s(
        s_lo: int | None, s_hi: int | None,
        d_lo: list[int], d_hi: list[int],
    ) -> tuple[int, ...] | None:
        return _dspace_search(list(d_lo), list(d_hi), d_base, s_lo, s_hi)

    # 阶段 2：二分最小可行 S = Σ|Δ_k|；可行试解带回的实际 Σ|d| 可作更紧上界。
    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        d_lo0, d_hi0 = d_domains()
        sol = feasible_s(None, mid, d_lo0, d_hi0)
        if sol is not None:
            achieved = sum(abs(sol[k]) for k in range(1, n))
            s_hi = min(mid, achieved)
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死字典序最小值。前缀固定后「x_k<=t」即「d_k<=t-x_{k-1}」。
    strains: list[int] = []
    d_lo, d_hi = d_domains()
    for var in range(n):
        if var == 0:
            t_lo, t_hi = x_lo[0], x_hi[0]
        else:
            prev = strains[var - 1]
            t_lo = max(x_lo[var], prev + d_lo[var])
            t_hi = min(x_hi[var], prev + d_hi[var])
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial_lo, trial_hi = list(d_lo), list(d_hi)
            if var == 0:
                trial_hi[0] = mid
            else:
                trial_hi[var] = mid - strains[var - 1]
            sol = feasible_s(best_s, best_s, trial_lo, trial_hi)
            if sol is not None:
                achieved = sol[0] if var == 0 else strains[var - 1] + sol[var]
                t_hi = min(mid, achieved)
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        if var == 0:
            d_lo[0] = d_hi[0] = t_lo
        else:
            d_lo[var] = d_hi[var] = t_lo - strains[var - 1]
    return strains


def _optimize_xspace(
    n: int, window_cons: list[Constraint],
    x_lo: list[int], x_hi: list[int], best_m: int,
) -> list[int]:
    """阶段 2/3：最优 M 很大时仍在 x 空间用 e_k≥|Δ| 的线性松弛。"""
    def feasible_x(s_cap: int) -> tuple[int, ...] | None:
        lo = list(x_lo)
        hi = list(x_hi)
        constraints = list(window_cons)
        constraints += _diff_constraints(n, best_m)
        lo += [0] * (n - 1)
        hi += [best_m] * (n - 1)
        constraints += _abs_sum_constraints(n, best_m, None, s_cap)
        return _search(lo, hi, constraints)

    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        sol = feasible_x(mid)
        if sol is not None:
            achieved = sum(abs(sol[k] - sol[k - 1]) for k in range(1, n))
            s_hi = min(mid, achieved)
        else:
            s_lo = mid + 1
    best_s = s_lo

    constraints = list(window_cons)
    constraints += _diff_constraints(n, best_m)
    constraints += _abs_sum_constraints(n, best_m, best_s, best_s)
    lo = list(x_lo) + [0] * (n - 1)
    hi = list(x_hi) + [best_m] * (n - 1)

    strains: list[int] = []
    for var in range(n):
        t_lo, t_hi = lo[var], hi[var]
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial = constraints + [(((var, 1),), None, mid)]
            sol = _search(list(lo), list(hi), trial)
            if sol is not None:
                t_hi = min(mid, sol[var])
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        lo[var] = hi[var] = t_lo
        narrowed = _propagate(lo, hi, constraints)
        assert narrowed is not None  # 已选最优值必然可行
        lo, hi = narrowed
    return strains


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
