"""
material_id_emulator_v3.py   •   2025-04-26
-------------------------------------------------
Generate two files from ANY SketchUp/3-D viewport PNG:

    material_id.png        flat, random colour per merged face   (debug only)
    material_id_mask.png   blue = fragile faces, white = unlock  (for <MASK_IMAGE>)

Requires:
    pip install scikit-image pillow numpy opencv-python-headless colour-science
"""

from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from collections import defaultdict
from colour import delta_E                         # ΔE00 implementation
import cv2, random, os, sys

# ------------ INPUT FILES (edit paths) -----------------
IMG_RGB     = "input.png"          # SketchUp viewport
CANNY_GRAY  = "soft_canny.png"     # pre-computed Canny, same resolution
DEPTH_GRAY  = "depth.png"          # optional 8-bit depth map
# -------------------------------------------------------

# ---------- MASTER TUNING KNOBS (balanced defaults) ----
N_SEGMENTS      = 350          # fewer seeds → fewer checkerboards
EDGE_RATIO_CUT  = 0.25         # >25 % Canny on border ⇒ keep split
COLOR_MERGE     = 6.0          # ΔE00 ≤ 6 merges similar hues
DEPTH_MERGE     = 2.5          # ≤1 % depth jump merges planes
ASPECT_MIN      = 6            # keep if long-skinny  (≥8 : 1)
SMALL_AREA      = 4000         # …or if pixel area ≤ 1 200
WHITE_THR       = 252          # ≥252 on *all* channels = background
# -------------------------------------------------------

# 0 ──────────────────────────────────────────────────────
rgb = np.array(Image.open(IMG_RGB).convert("RGB"))
canny = np.array(Image.open(CANNY_GRAY).convert("L"))
depth = np.array(Image.open(DEPTH_GRAY).convert("L")) if os.path.exists(DEPTH_GRAY) else None
h, w, _ = rgb.shape

# 1  pre-smooth (reduces AA speckles)
rgb_s = cv2.bilateralFilter(rgb, d=7, sigmaColor=30, sigmaSpace=3)

# 2  SLIC seeds
segments = slic(rgb_s, n_segments=N_SEGMENTS, compactness=12,
                start_label=0, convert2lab=False)
num_seg  = segments.max() + 1

lab      = rgb2lab(rgb_s)
mean_lab = np.zeros((num_seg, 3))
mean_d   = np.zeros(num_seg)
area     = np.bincount(segments.flatten())

for s in range(num_seg):
    if area[s]:
        sel = segments == s
        mean_lab[s] = lab[sel].mean(0)
        mean_d[s]   = depth[sel].mean() if depth is not None else 0

# 3  adjacency stats
border, support = defaultdict(int), defaultdict(int)
for y in range(h):
    for x in range(w-1):
        a, b = segments[y,x], segments[y,x+1]
        if a != b:
            k = tuple(sorted((a,b)))
            border[k]  += 1
            support[k] += (canny[y,x]>0 or canny[y,x+1]>0)
for y in range(h-1):
    for x in range(w):
        a, b = segments[y,x], segments[y+1,x]
        if a != b:
            k = tuple(sorted((a,b)))
            border[k]  += 1
            support[k] += (canny[y,x]>0 or canny[y+1,x]>0)

# 4  greedy merge (union-find)
parent = list(range(num_seg))
def find(x):
    while parent[x]!=x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a,b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for (a,b), blen in border.items():
    if blen < 20:
        continue
    if support[(a,b)] / blen > EDGE_RATIO_CUT:
        continue                          # strong edge ⇒ keep split
    if delta_E(mean_lab[a], mean_lab[b]) > COLOR_MERGE:
        continue
    if depth is not None and abs(mean_d[a] - mean_d[b]) > DEPTH_MERGE:
        continue
    union(a,b)

root_lbl = {s: find(s) for s in range(num_seg)}
merged   = np.vectorize(root_lbl.get)(segments)

# 5  random flat colour per merged region (debug)  ── FIXED
rand = random.Random(42)
unique_roots = np.unique(merged)
lookup = np.zeros((unique_roots.max() + 1, 3), dtype=np.uint8)

for r in unique_roots:
    lookup[r] = [rand.randint(0, 255) for _ in range(3)]

mat_id = lookup[merged]              # ← fast NumPy indexing
Image.fromarray(mat_id).save("material_id.png")

# 6  skinny / tiny selector  →  mask
mask = np.zeros((h, w), np.uint8)                 # 0 = white
n, labels, stats, _ = cv2.connectedComponentsWithStats(
    cv2.cvtColor(mat_id, cv2.COLOR_RGB2GRAY), connectivity=8)

kept = 0
for i in range(1, n):                             # skip background
    x, y, w_, h_, a_ = stats[i]
    aspect = max(w_, h_) / max(1, min(w_, h_))
    if aspect >= ASPECT_MIN or a_ <= SMALL_AREA:
        mask[labels == i] = 255
        kept += 1

# wipe pure-white background
bg_pix = np.all(mat_id >= WHITE_THR, axis=-1)
mask[bg_pix] = 0

# 7  blue-on-white RGB mask
mask_rgb = np.ones_like(mat_id) * 255
mask_rgb[mask > 0] = (0, 0, 255)
Image.fromarray(mask_rgb).save("material_id_mask.png")

print(f"[OK] wrote material_id.png + material_id_mask.png  | blue blobs: {kept}")

