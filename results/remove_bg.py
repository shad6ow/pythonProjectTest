from pathlib import Path
from PIL import Image
import numpy as np

FILES = [
    "patent_fig1_overall_flow_transparent.png",
    "patent_fig2_feature_engineering_transparent.png",
    "patent_fig3_rmt_module_transparent.png",
    "patent_fig4_dual_path_transformer_transparent.png",
    "patent_fig5_fusion_ensemble_transparent.png",
]

def process(path: Path):
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    visible = arr[..., 3] > 0
    arr[visible, 0] = 0
    arr[visible, 1] = 0
    arr[visible, 2] = 0
    Image.fromarray(arr).save(path)
    print(f"updated {path.name}")

def main():
    base = Path(__file__).parent
    for name in FILES:
        process(base / name)

if __name__ == "__main__":
    main()
