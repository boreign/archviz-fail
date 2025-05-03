import cv2
import argparse
import os

def compute_edge_maps(input_path, output_dir, ksize=3):
    # lê em escala de cinza
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler {input_path}")

    # Sobel X e Y
    sobelx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobely = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    # converte para 8-bits absolutos
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    # Sobel combinado (magnitude aproximada)
    sobel_comb = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # Laplaciano
    laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F, ksize=ksize)
    laplacian = cv2.convertScaleAbs(laplacian)

    # garante diretório
    os.makedirs(output_dir, exist_ok=True)

    # nome base do arquivo
    base = os.path.splitext(os.path.basename(input_path))[0]

    # salva resultados
    cv2.imwrite(os.path.join(output_dir, f"{base}_sobelx.png"), sobelx)
    cv2.imwrite(os.path.join(output_dir, f"{base}_sobely.png"), sobely)
    cv2.imwrite(os.path.join(output_dir, f"{base}_sobel_combined.png"), sobel_comb)
    cv2.imwrite(os.path.join(output_dir, f"{base}_laplacian.png"), laplacian)

    print(f"Mapas de borda salvos em {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera mapas Sobel e Laplaciano de uma imagem")
    parser.add_argument("--input", "-i", required=True, help="Caminho para a imagem de entrada")
    parser.add_argument("--output_dir", "-o", default="edge_maps", help="Pasta de saída")
    parser.add_argument("--ksize", "-k", type=int, default=3,
                        help="Tamanho do kernel (deve ser ímpar, ex: 1,3,5,7)")

    args = parser.parse_args()
    compute_edge_maps(args.input, args.output_dir, args.ksize)

