"""Screenshot -> PositionSpec: lattice fit, per-hex/port/piece classifiers.

Everything works in board units: the engine topology has unit hex radius,
pointy-top, y down (``catan.engine.board``). A fitted lattice is
``(cx, cy, R)``: board point (x, y) sits at pixel (cx + R*x, cy + R*y).
Screenshots are assumed unrotated (true for browser screenshots).

The classifiers are colour histograms ("backprojection", Swain & Ballard 1991)
and normalised template matching, both learned from labelled screenshots by
``catan.vision.calibrate``. A profile (.npz) holds what was learned.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from catan.engine import TOPOLOGY as T
from catan.engine.board import PORT_TYPES, TERRAIN_COUNTS, TOKEN_SEQUENCE, hex_center, _ring

TERRAINS = ("wood", "brick", "sheep", "wheat", "ore", "desert")
NUMBERS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
PORT_KINDS = ("3:1", "wood", "brick", "sheep", "wheat", "ore")
HEX_XY = np.array(T.hex_xy)
VERT_XY = np.array(T.vertex_xy)
RING3 = np.array([hex_center(q, r) for q, r in _ring(3, 0)])
COAST = T.coast_edges
def _coast_points(k):
    """Per coastal edge: the point ``k`` of the way from its hex centre to the edge midpoint."""
    out = []
    for e in T.coast_edges:
        a, b = T.edge_vertices[e]
        h = next(h for h in range(T.num_hexes) if a in T.hex_vertices[h] and b in T.hex_vertices[h])
        c, m = np.array(T.hex_xy[h]), (np.array(T.vertex_xy[a]) + np.array(T.vertex_xy[b])) / 2
        out.append(c + k * (m - c))
    return np.array(out)


COAST_IN, COAST_OUT = _coast_points(0.8), _coast_points(1.2)  # just inside / outside the coastline: fixes the scale
SAMPLE = ((0.0, -0.6), (0.0, 0.6))   # per-hex terrain sample points: clear of the token, pieces and robber
VERT_BOX = (-0.22, 0.22, -0.30, 0.12)  # window around a vertex (covers a city, little of the roads)
LOW = 0.6                           # fields under this confidence are flagged for review
PROFILE_DIR = Path(__file__).with_name("profiles")


class VisionUnavailable(RuntimeError):
    """An optional dependency for image import is missing."""


def _pil():
    try:
        from PIL import Image
    except ImportError as e:
        raise VisionUnavailable("screenshot import needs Pillow: pip install 'catan-lab[vision]' (or pip install pillow)") from e
    return Image


MAX_PIXELS = 40_000_000  # decompression-bomb guard, checked before decoding
WORK_DIM = 2000          # larger screenshots are downscaled first; the board stays well resolved


def load_image(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG/WebP bytes to an RGB uint8 array. Raises ValueError for bad images."""
    Image = _pil()
    import io
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in ("PNG", "JPEG", "WEBP"):
            raise ValueError(f"unsupported image format {im.format}; use PNG, JPEG or WebP")
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"image is {im.width}x{im.height}; at most {MAX_PIXELS // 1_000_000} megapixels")
        im = im.convert("RGB")
    except (OSError, Image.DecompressionBombError) as e:
        raise ValueError(f"cannot decode image: {e}") from e
    if max(im.size) > WORK_DIM:
        k = WORK_DIM / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.BILINEAR)
    return np.asarray(im)


def bins(img: np.ndarray) -> np.ndarray:
    """16x16x16 RGB histogram bin of every pixel."""
    q = img >> 4
    return (q[..., 0].astype(np.int32) << 8) | (q[..., 1].astype(np.int32) << 4) | q[..., 2]


def load_profile(name_or_path) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = PROFILE_DIR / f"{p.name}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files} | {"name": p.stem}


def profiles(extra_dir: Path | None = None) -> list[Path]:
    out = sorted(PROFILE_DIR.glob("*.npz"))
    if extra_dir and extra_dir.is_dir():
        out += sorted(extra_dir.glob("*.npz"))
    return out


# ------------------------------------------------------------------ lattice
def _boxsum(a: np.ndarray, h: int) -> np.ndarray:
    """Sum of ``a`` over the (2h+1)^2 box centred on every pixel (zero outside)."""
    p = np.pad(a, h)
    c = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.float64)
    c[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * h + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]).astype(np.float32)


def _shift(a: np.ndarray, dy: int, dx: int, fill=0.0) -> np.ndarray:
    """out[y, x] = a[y+dy, x+dx] (``fill`` outside)."""
    out = np.full_like(a, fill)
    H, W = a.shape
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    if H - abs(dy) > 0 and W - abs(dx) > 0:
        out[yd, xd] = a[ys, xs]
    return out


def _score_land(L: np.ndarray, R: float) -> np.ndarray:
    """Lattice score for every centre at hex radius R: land at the 19 hexes, not land on the ring around them."""
    h = max(1, round(0.15 * R))
    B = _boxsum(L, h) / (2 * h + 1) ** 2
    h3 = max(1, round(0.4 * R))
    B3 = _boxsum(L, h3) / (2 * h3 + 1) ** 2
    s = np.zeros_like(L)
    for x, y in HEX_XY:
        for ox, oy in SAMPLE:
            s += _shift(B, round(R * (y + oy)), round(R * (x + ox)))
    s /= 2 * len(HEX_XY)
    w = np.zeros_like(L)
    for x, y in RING3:
        w += _shift(B3, round(R * y), round(R * x))
    hc = max(1, round(0.06 * R))
    Bc = _boxsum(L, hc) / (2 * hc + 1) ** 2
    ci, co = np.zeros_like(L), np.zeros_like(L)
    for (xi, yi), (xo, yo) in zip(COAST_IN, COAST_OUT):
        ci += _shift(Bc, round(R * yi), round(R * xi))
        co += _shift(Bc, round(R * yo), round(R * xo))
    n = len(COAST_IN)
    return s + 1 - w / len(RING3) + ci / n + 1 - co / n  # 4 when everything fits


def _search(score_fn, H: int, W: int, rs) -> tuple[float, float, float, float]:
    best = (-1e9, 0, 0, 0)
    for R in rs:
        m = score_fn(R)
        i = int(np.argmax(m))
        if m.flat[i] > best[0]:
            best = (float(m.flat[i]), i % W, i // W, R)
    return best


def _pool(a: np.ndarray, k: int) -> np.ndarray:
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    a = a[:H, :W]
    return a.reshape(H // k, k, W // k, k, *a.shape[2:]).mean((1, 3))


def fit_lattice(fmap: np.ndarray, score_maker) -> dict:
    """Coarse-to-fine search over (cx, cy, R). ``fmap`` is a per-pixel feature map (land probability,
    or RGB for calibration); ``score_maker(pooled_map) -> (R -> score map)``."""
    H, W = fmap.shape[:2]
    out = None
    for target, span, step in ((320, None, 1.03), (900, 0.07, 1.01)):
        k = max(1, round(max(H, W) / target))
        small = _pool(fmap.astype(np.float32), k)
        fn = score_maker(small)
        if out is None:
            lo, hi = min(small.shape[:2]) / 22, min(small.shape[:2]) / 7.5
        else:
            lo, hi = out["r"] / k * (1 - span), out["r"] / k * (1 + span)
        rs = [lo * step ** i for i in range(int(math.log(hi / lo) / math.log(step)) + 1)]
        sc, x, y, R = _search(fn, small.shape[0], small.shape[1], rs)
        out = {"cx": (x + 0.5) * k - 0.5, "cy": (y + 0.5) * k - 0.5, "r": R * k, "score": sc}
    out["width"], out["height"] = W, H
    return out


def land_scorer(small):
    return lambda R: _score_land(small, R)


def refine_lattice(img: np.ndarray, lat: dict, prof: dict) -> dict:
    """Snap the lattice to the number tokens: find each token by template match over small shifts,
    then least-squares fit (cx, cy, R) to the token centres. Corrects the few-percent scale error
    the colour fit leaves, which the token and piece readers are sensitive to."""
    gray = img.mean(2)
    for span in (0.12, 0.03):
        offs = np.linspace(-span, span, 9)
        rows, obs, scores = [], [], []
        for x, y in HEX_XY:
            best = max(((float((prof["numbers"] @ norm_vec(patch(gray, lat, x + dx, y + dy, 0.32, 24)[TOKEN_MASK])).max()), dx, dy)
                        for dy in offs for dx in offs))
            scores.append(best[0])
            rows.append((x, y))
            obs.append(to_px(lat, (x + best[1], y + best[2])))
        keep = np.array(scores) > 0.5
        if keep.sum() < 8:
            return lat  # too few tokens to trust; keep the colour fit
        xy, o = np.array(rows)[keep], np.array(obs)[keep]
        A = np.zeros((2 * len(xy), 3))
        A[0::2, 0], A[1::2, 1] = 1, 1
        A[0::2, 2], A[1::2, 2] = xy[:, 0], xy[:, 1]
        sol, *_ = np.linalg.lstsq(A, o.ravel(), rcond=None)
        lat = lat | {"cx": float(sol[0]), "cy": float(sol[1]), "r": float(sol[2])}
    return lat


# ------------------------------------------------------------------ sampling
def to_px(lat, xy):
    xy = np.asarray(xy, dtype=float)
    return lat["cx"] + lat["r"] * xy[..., 0], lat["cy"] + lat["r"] * xy[..., 1]


def region(img, lat, x, y, r_in, r_out):
    """Pixels whose board-unit distance from (x, y) is in [r_in, r_out)."""
    px, py = to_px(lat, (x, y))
    R = lat["r"]
    H, W = img.shape[:2]
    x0, x1 = max(0, int(px - r_out * R)), min(W, int(px + r_out * R) + 1)
    y0, y1 = max(0, int(py - r_out * R)), min(H, int(py + r_out * R) + 1)
    if x0 >= x1 or y0 >= y1:
        return img[:0, :0].reshape(0, *img.shape[2:])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - px, yy - py) / R
    return img[y0:y1, x0:x1][(d >= r_in) & (d < r_out)]


def box(img, lat, x, y, b):
    """Pixels in the board-unit box (x+b0..x+b1, y+b2..y+b3)."""
    R = lat["r"]
    px, py = to_px(lat, (x, y))
    H, W = img.shape[:2]
    x0, x1 = max(0, int(px + b[0] * R)), min(W, int(px + b[1] * R) + 1)
    y0, y1 = max(0, int(py + b[2] * R)), min(H, int(py + b[3] * R) + 1)
    return img[y0:y1, x0:x1].reshape(-1, *img.shape[2:])


def band(img, lat, a, b, t0=0.3, t1=0.7, half=0.04):
    """Pixels within ``half`` board units of segment a->b between fractions t0..t1."""
    R = lat["r"]
    (ax, bx), (ay, by) = to_px(lat, np.array([a, b]))
    H, W = img.shape[:2]
    x0, x1 = max(0, int(min(ax, bx) - half * R - 1)), min(W, int(max(ax, bx) + half * R) + 2)
    y0, y1 = max(0, int(min(ay, by) - half * R - 1)), min(H, int(max(ay, by) + half * R) + 2)
    if x0 >= x1 or y0 >= y1:
        return img[:0, :0].reshape(0, *img.shape[2:])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = ((xx - ax) * dx + (yy - ay) * dy) / L2
    d = np.abs((xx - ax) * dy - (yy - ay) * dx) / math.sqrt(L2) / R
    return img[y0:y1, x0:x1][(t >= t0) & (t <= t1) & (d <= half)]


def patch(img, lat, x, y, half, n):
    """``n`` x ``n`` bilinear resample of the board-unit square of half-size ``half`` at (x, y)."""
    px, py = to_px(lat, (x, y))
    s = np.linspace(-half, half, n) * lat["r"]
    gx, gy = np.meshgrid(px + s, py + s)
    H, W = img.shape[:2]
    gx, gy = np.clip(gx, 0, W - 1.001), np.clip(gy, 0, H - 1.001)
    x0, y0 = gx.astype(int), gy.astype(int)
    fx, fy = gx - x0, gy - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    im = img.astype(np.float32)
    return (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy)
            + im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)


def norm_vec(a) -> np.ndarray:
    v = np.asarray(a, dtype=np.float32).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


_G = np.linspace(-1, 1, 24)
TOKEN_MASK = np.hypot(*np.meshgrid(_G, _G)) <= 0.9  # the token disc only; the terrain around it is noise


def number_vec(img, lat, h):
    return norm_vec(patch(img.mean(2), lat, *HEX_XY[h], 0.32, 24)[TOKEN_MASK])


def port_anchor(e, offset):
    a, b = (VERT_XY[v] for v in T.edge_vertices[e])
    m = (a + b) / 2
    return m + m / np.linalg.norm(m) * offset


def port_vec(img, lat, e, offset):
    return norm_vec(patch(img, lat, *port_anchor(e, offset), 0.32, 16))


VGRID = np.stack(np.meshgrid(np.linspace(-0.27, 0.27, 10), np.linspace(-0.36, 0.2, 10)), -1).reshape(-1, 2)


def vertex_shape(pcls, lat, v):
    """(seat or -1, 100-cell 0/1 map of that seat's colour around the vertex). The map's shape, not its
    area, tells a settlement from a city from passing roads."""
    px, py = to_px(lat, VERT_XY[v] + VGRID)
    H, W = pcls.shape
    cells = pcls[np.clip(np.rint(py).astype(int), 0, H - 1), np.clip(np.rint(px).astype(int), 0, W - 1)]
    counts = np.bincount(cells, minlength=5)[1:5]
    if not counts.any():
        return -1, np.zeros(len(VGRID), np.uint8)
    p = int(counts.argmax())
    return p, (cells == p + 1).astype(np.uint8)


def knn(X, y, q, k=5):
    """Majority label of the k nearest rows of X to q, and the fraction of votes it got."""
    d = np.abs(X.astype(np.int16) - q.astype(np.int16)).sum(1)
    near = y[np.argsort(d, kind="stable")[:k]]
    c = np.bincount(near, minlength=3) / np.maximum(np.bincount(y, minlength=3), 1)  # class-balanced vote
    return int(c.argmax()), float(c.max() / c.sum())


def edge_frac(pcls, lat, e):
    a, b = (VERT_XY[v] for v in T.edge_vertices[e])
    px = band(pcls, lat, a, b)
    return np.bincount(px, minlength=5)[1:5] / max(1, px.size)


def piece_classes(img, prof):
    """Per pixel: 0 = background, 1..4 = seat 0..3 colour."""
    return prof["piece"][bins(img)]


def robber_score(rmask, lat, h):
    return float(region(rmask, lat, *HEX_XY[h], 0, 0.85).mean())


# ------------------------------------------------------------------ assignment
def assign(prob: np.ndarray, quota: dict, labels) -> list[int]:
    """Label each row of ``prob`` (rows x labels) honouring the per-label quota.
    shortcut: greedy by probability, not an optimal assignment; fine while the classes are well separated,
    swap for scipy.optimize.linear_sum_assignment if confusions get common."""
    left = {i: quota.get(l, 0) for i, l in enumerate(labels)}
    out = [-1] * prob.shape[0]
    for flat in np.argsort(-prob, axis=None):
        r, c = divmod(int(flat), prob.shape[1])
        if out[r] == -1 and left[c] > 0:
            out[r] = c
            left[c] -= 1
    return out


def softmax(s, temp):
    z = np.exp((s - s.max(-1, keepdims=True)) * temp)
    return z / z.sum(-1, keepdims=True)


# ------------------------------------------------------------------ read the board
def read_board(img: np.ndarray, prof: dict, lat: dict | None = None) -> dict:
    """Detect everything on the board. Returns ``{lattice, layout, vertex_owner, vertex_level,
    edge_owner, robber, confidence}`` (confidence in 0..1, aligned with each field)."""
    idx = bins(img)
    if lat is None:
        lat = refine_lattice(img, fit_lattice(prof["land"][idx], land_scorer), prof)
    # terrain: per-pixel vote with the per-terrain colour likelihoods
    tl = np.log(prof["terrain"] + 1e-4)            # (6, 4096)
    tprob = np.zeros((19, 6))
    for h, (x, y) in enumerate(HEX_XY):
        px = region(idx, lat, x, y, 0.42, 0.72)
        if px.size:
            tprob[h] = np.bincount(tl[:, px].argmax(0), minlength=6) / px.size
    ter = assign(tprob, TERRAIN_COUNTS, TERRAINS)
    terrain = [TERRAINS[t] for t in ter]
    # confidence: the chosen terrain's share of the two strongest votes, stretched to 0..1
    t_conf = [float(np.clip(2 * tprob[h, ter[h]] / (tprob[h, ter[h]] + np.sort(tprob[h])[-2 if tprob[h].argmax() == ter[h] else -1] + 1e-9) - 1, 0, 1))
              for h in range(19)]
    # numbers: template match on the token
    tmpl = prof["numbers"]                         # (10, token-disc cells), unit vectors
    nprob = softmax(np.array([tmpl @ number_vec(img, lat, h) for h in range(19)]), 40)
    land = [h for h in range(19) if terrain[h] != "desert"]
    quota = {n: TOKEN_SEQUENCE.count(n) for n in NUMBERS}
    na = assign(nprob[land], quota, NUMBERS)
    number, n_conf = [None] * 19, [1.0] * 19
    for h, a in zip(land, na):
        number[h], n_conf[h] = NUMBERS[a], float(nprob[h, a])
    n_conf[terrain.index("desert")] = t_conf[terrain.index("desert")]
    # robber
    rmask = prof["robber"][idx]
    rs = np.array([robber_score(rmask, lat, h) for h in range(19)])
    robber = int(rs.argmax())
    srt = np.sort(rs)
    r_conf = float(np.clip((srt[-1] - srt[-2]) / (srt[-1] + 1e-6), 0, 1))
    # ports: 9 coastal edges with the strongest marker, no two adjacent, then the type quota
    off = float(prof["port_offset"])
    pt, none = prof["ports"], prof["port_none"]
    pv = [port_vec(img, lat, e, off) for e in COAST]
    pres = np.array([(pt @ v).max() - none @ v for v in pv])
    chosen = []
    for i in np.argsort(-pres):
        if len(chosen) < 9 and all(min(abs(i - j), len(COAST) - abs(i - j)) > 1 for j in chosen):
            chosen.append(int(i))
    chosen.sort()
    pprob = softmax(np.array([pt @ pv[i] for i in chosen]), 40)
    pa = assign(pprob, {k: PORT_TYPES.count(k) for k in PORT_KINDS}, PORT_KINDS)
    port_edges = [COAST[i] for i in chosen]
    port_types = [PORT_KINDS[a] for a in pa]
    p_conf = [float(min(1 / (1 + math.exp(-12 * pres[i])), pprob[k, pa[k]])) for k, i in enumerate(chosen)]
    # pieces
    pcls = piece_classes(img, prof)
    road = float(prof["edge_road"])
    vo, vl, v_conf = [-1] * len(VERT_XY), [0] * len(VERT_XY), [1.0] * len(VERT_XY)
    for v in range(len(VERT_XY)):
        p, m = vertex_shape(pcls, lat, v)
        if p >= 0:
            lvl, v_conf[v] = knn(prof["vertex_X"], prof["vertex_y"], m)
            if lvl:
                vo[v], vl[v] = p, lvl
    eo, e_conf = [-1] * T.num_edges, [1.0] * T.num_edges
    for e in range(T.num_edges):
        f = edge_frac(pcls, lat, e)
        p, m = int(f.argmax()), float(f.max())
        if m >= road:
            eo[e] = p
        e_conf[e] = float(np.clip(abs(m - road) / (0.5 * road), 0, 1))
    return {
        "lattice": {k: float(lat[k]) for k in ("cx", "cy", "r", "score")} | {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "layout": {"hex_terrain": terrain, "hex_number": number, "port_edges": port_edges, "port_types": port_types},
        "vertex_owner": vo, "vertex_level": vl, "edge_owner": eo, "robber": robber,
        "confidence": {"hex_terrain": t_conf, "hex_number": n_conf, "ports": p_conf, "robber": r_conf,
                       "vertices": v_conf, "edges": e_conf, "lattice": float(np.clip(lat["score"] / 4, 0, 1))},
    }
