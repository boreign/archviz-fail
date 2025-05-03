from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
import cv2

EDGE_RATIO     = 0.05
DE_THRESH      = 3.0
DEPTH_THRESH   = 2.5
N_SEGMENTS     = 1000

img    = np.array(Image.open("input.png").convert("RGB"))
canny  = np.array(Image.open("soft_canny.png").convert("L"))
depth  = np.array(Image.open("depth.png").convert("L"))
h, w, _ = img.shape

segments = slic(img, n_segments=N_SEGMENTS, compactness=12, start_label=0)
num_seg  = segments.max() + 1

lab_img   = rgb2lab(img)
mean_lab  = np.zeros((num_seg, 3))
mean_d    = np.zeros(num_seg)
area      = np.bincount(segments.flatten())

for s in range(num_seg):
    if area[s]:
        m            = segments == s
        mean_lab[s]  = lab_img[m].mean(0)
        mean_d[s]    = depth[m].mean()

from collections import defaultdict
border, support = defaultdict(int), defaultdict(int)
for y in range(h):
    for x in range(w-1):
        a, b = segments[y,x], segments[y,x+1]
        if a != b:
            k = tuple(sorted((a,b)))
            border[k]  += 1
            support[k] += canny[y,x] > 0 or canny[y,x+1] > 0
for y in range(h-1):
    for x in range(w):
        a, b = segments[y,x], segments[y+1,x]
        if a != b:
            k = tuple(sorted((a,b)))
            border[k]  += 1
            support[k] += canny[y,x] > 0 or canny[y+1,x] > 0

lock = set()
def dE(u,v): return np.linalg.norm(u-v)

for (a,b), blen in border.items():
    if blen < 20: continue
    if support[(a,b)] / blen < EDGE_RATIO: continue
    if dE(mean_lab[a], mean_lab[b]) > DE_THRESH or abs(mean_d[a]-mean_d[b]) > DEPTH_THRESH:
        lock.add(a if area[a] < area[b] else b)

mask = np.ones_like(img)*255
for s in lock:
    mask[segments==s] = [0,0,255]
Image.fromarray(mask).save("auto_mask_strict.png")
print("Saved auto_mask_strict.png with", len(lock), "regions")

