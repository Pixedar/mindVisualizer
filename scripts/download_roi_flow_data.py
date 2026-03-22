#!/usr/bin/env python
"""Download ROI flow data from HuggingFace.

Downloads the probe embedding, ROI vectors, ROI centers, and OOS manifold
points needed for ROI flow mode.

Usage:
    python scripts/download_roi_flow_data.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "roi_flow"


def _fix_ssl():
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    except ImportError:
        pass


def main():
    _fix_ssl()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Install with:")
        print("  pip install huggingface_hub")
        sys.exit(1)

    REPO_ID = "Pixedar/mindVisualizer-roi-flow-data"
    FILES = [
        # Probe data
        "probe_embed.npy",
        "probe_roi.npy",
        "probe_roi_centers.npy",
        # Manifold OOS points
        "universal_soul_2sdm_rest_points.ply",
        # MDN flow field (manifold space, not brain space)
        "mdn_training_all_points.npy",
        "mdn_universal_raw_grid64_meta.json",
        "mdn_universal_all_grid64_mean_xyz3.bin",
        "mdn_universal_all_grid64_mu1_xyz3.bin",
        "mdn_universal_all_grid64_mu2_xyz3.bin",
        "mdn_universal_all_grid64_pi.bin",
        "mdn_universal_all_grid64_entropy.bin",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  [skip] {fname} (already exists)")
            continue

        print(f"  Downloading {fname}...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )
            print(f"  [ok] {fname}")
        except Exception as e:
            print(f"  [error] {fname}: {e}")

    print(f"\nDone! Files saved to: {DATA_DIR}")
    print("\nYou can now run ROI flow mode:")
    print("  python examples/roi_flow_mode.py")


if __name__ == "__main__":
    main()
