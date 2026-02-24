#!/usr/bin/env python3
"""Download Hi-SAM model checkpoints from HuggingFace.

Usage:
    python download_models.py --model-type vit_l
    python download_models.py --model-type vit_l --hier-only
    python download_models.py --all
"""

import argparse
import os
import sys

REPO_ID = "GoGiants1/Hi-SAM"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

MODELS = {
    "vit_b": {
        "stroke": "sam_tss_b_hiertext.pth",
        "hier": "hi_sam_b.pth",
        "encoder": "sam_vit_b_01ec64.pth",
    },
    "vit_l": {
        "stroke": "sam_tss_l_hiertext.pth",
        "hier": "hi_sam_l.pth",
        "encoder": "sam_vit_l_0b3195.pth",
    },
    "vit_h": {
        "stroke": "sam_tss_h_hiertext.pth",
        "hier": "hi_sam_h.pth",
        "encoder": "sam_vit_h_4b8939.pth",
    },
}


def download_file(url: str, dest: str):
    """Download a file with progress bar using requests."""
    import requests
    from tqdm import tqdm

    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    print(f"  Saved: {dest}")


def download_model(model_type: str, output_dir: str, stroke: bool = True, hier: bool = True):
    """Download checkpoints for a specific model type."""
    if model_type not in MODELS:
        print(f"Unknown model type: {model_type}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    files = MODELS[model_type]
    to_download = []
    if stroke:
        to_download.append(files["stroke"])
    if hier:
        to_download.append(files["hier"])

    for filename in to_download:
        url = f"{BASE_URL}/{filename}"
        dest = os.path.join(output_dir, filename)
        download_file(url, dest)


def main():
    parser = argparse.ArgumentParser(description="Download Hi-SAM model checkpoints")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vit_b", "vit_l", "vit_h"],
        help="Model variant to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("MODEL_DIR", "/models"),
        help="Directory to save checkpoints (default: /models)",
    )
    parser.add_argument("--all", action="store_true", help="Download all model variants")
    parser.add_argument("--stroke-only", action="store_true", help="Only download stroke model")
    parser.add_argument("--hier-only", action="store_true", help="Only download hierarchical model")
    args = parser.parse_args()

    if not args.all and not args.model_type:
        parser.error("Specify --model-type or --all")

    os.makedirs(args.output_dir, exist_ok=True)
    stroke = not args.hier_only
    hier = not args.stroke_only

    if args.all:
        for mt in MODELS:
            print(f"\nDownloading {mt} models...")
            download_model(mt, args.output_dir, stroke=stroke, hier=hier)
    else:
        print(f"\nDownloading {args.model_type} models...")
        download_model(args.model_type, args.output_dir, stroke=stroke, hier=hier)

    print("\nDone!")


if __name__ == "__main__":
    main()
