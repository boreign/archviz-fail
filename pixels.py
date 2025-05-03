import os
import cv2
import argparse
import numpy as np
from skimage import io, color, segmentation
from skimage.segmentation import slic, felzenszwalb, mark_boundaries

def segment_slic(img_bgr, n_segments, compactness):
    img_lab = color.rgb2lab(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    labels = slic(img_lab, n_segments=n_segments, compactness=compactness, start_label=0)
    return labels

def segment_felzenszwalb(img_bgr, scale, sigma, min_size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    labels = felzenszwalb(img_rgb, scale=scale, sigma=sigma, min_size=min_size)
    return labels

def save_results(img_bgr, labels, canny, out_dir, base):
    # cria pasta
    os.makedirs(out_dir, exist_ok=True)

    # 1) segmentação colorida
    out_seg = os.path.join(out_dir, f"{base}_segmented.png")
    seg_vis = color.label2rgb(labels, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), kind='avg')
    seg_vis = cv2.cvtColor((seg_vis*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_seg, seg_vis)

    # 2) máscaras individuais
    unique = np.unique(labels)
    for u in unique:
        mask = (labels == u).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(out_dir, f"{base}_mask_{u}.png"), mask)

    # 3) overlay de fronteiras + Canny
    # fronteiras superpixel
    bnd = mark_boundaries(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), labels, color=(1,0,0))
    bnd = (bnd*255).astype(np.uint8)
    bnd = cv2.cvtColor(bnd, cv2.COLOR_RGB2BGR)
    # sobrepõe Canny em azul
    if canny is not None:
        edges = (canny>0).astype(np.uint8)*255
        bnd[edges==255] = (255,0,0)  # azul puro
    cv2.imwrite(os.path.join(out_dir, f"{base}_boundaries.png"), bnd)

    print(f"Resultados salvos em {out_dir}:")
    print(f" • {base}_segmented.png")
    print(f" • {len(unique)} masks ({base}_mask_*.png)")
    print(f" • {base}_boundaries.png")

def main():
    p = argparse.ArgumentParser("Superpixel Segmentation (SLIC / Felzenszwalb)")
    p.add_argument("-i","--input",      required=True, help="Imagem de entrada")
    p.add_argument("-o","--output_dir", default="superpixels", help="Pasta de saída")
    p.add_argument("--method",          choices=["slic","felzenszwalb"], default="slic")
    # SLIC params
    p.add_argument("--n_segments", "-n", type=int, default=400,
                   help="Número de superpixels (SLIC)")
    p.add_argument("--compactness",    type=float, default=10.0,
                   help="Compacidade do cluster (SLIC)")
    # Felzenszwalb params
    p.add_argument("--scale",          type=float, default=100.0,
                   help="Controle de tamanho (Felzenszwalb)")
    p.add_argument("--sigma",          type=float, default=0.5,
                   help="Suavização prévia (Felzenszwalb)")
    p.add_argument("--min_size",       type=int, default=50,
                   help="Tamanho mínimo de regiões (Felzenszwalb)")
    # Canny para overlay
    p.add_argument("--canny",          help="Mapa de Canny para sobrepor (opcional)")
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir {args.input}")

    canny = None
    if args.canny:
        canny = cv2.imread(args.canny, cv2.IMREAD_GRAYSCALE)
        if canny is None:
            raise FileNotFoundError(f"Não foi possível abrir {args.canny}")

    base = os.path.splitext(os.path.basename(args.input))[0]

    if args.method == "slic":
        labels = segment_slic(img, args.n_segments, args.compactness)
    else:
        labels = segment_felzenszwalb(img, args.scale, args.sigma, args.min_size)

    save_results(img, labels, canny, args.output_dir, base)

if __name__ == "__main__":
    main()

