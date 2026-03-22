#!/usr/bin/env python3
"""Build a combined brain parcellation atlas for mindVisualizer.

Combines three complementary atlases (all in MNI152 1mm space) into a single
NIfTI volume with a unified label map:

  Layer 1 — Harvard-Oxford Cortical (48 regions):
      Broad cortical coverage with LLM-friendly names.
  Layer 2 — Harvard-Oxford Subcortical (17 regions):
      Thalamus, putamen, caudate, hippocampus, amygdala, etc.
  Layer 3 — Julich-Brain cytoarchitectonic (62 regions, HIGHEST PRIORITY):
      Fine-grained motor (BA4a/4p), somatosensory (BA1-3), visual (V1-V5),
      auditory (TE1.0-1.2), Broca's (BA44/45), hippocampal subfields,
      amygdala subdivisions, white matter tracts.

Where Julich-Brain has a label, it overrides the coarser Harvard-Oxford label.
This gives maximum spatial coverage with maximum detail where available.

Output:
  data/extra_parcellation/combined_atlas.nii.gz       (NIfTI volume, ~700KB)
  data/extra_parcellation/combined_atlas_labels.json   (label ID → name map)

Requirements:
  pip install nilearn nibabel

Usage:
  python scripts/setup_extra_parcellation.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "extra_parcellation"


def main():
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        print("ERROR: nibabel and numpy are required.")
        print("  pip install nibabel numpy")
        sys.exit(1)

    try:
        import nilearn.datasets as ds
    except ImportError:
        print("ERROR: nilearn is required to fetch the source atlases.")
        print("  pip install nilearn")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nii = OUT_DIR / "combined_atlas.nii.gz"
    out_labels = OUT_DIR / "combined_atlas_labels.json"

    if out_nii.exists() and out_labels.exists():
        print(f"[setup] Combined atlas already exists: {out_nii}")
        print("[setup] Delete it and re-run to rebuild.")
        return

    # ---- Fetch source atlases via nilearn (auto-downloads) ----
    print("[setup] Fetching Harvard-Oxford cortical atlas ...")
    ho_cort = ds.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")

    print("[setup] Fetching Harvard-Oxford subcortical atlas ...")
    ho_sub = ds.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")

    print("[setup] Fetching Julich-Brain cytoarchitectonic atlas ...")
    juelich = ds.fetch_atlas_juelich("maxprob-thr25-1mm")

    # ---- Load NIfTI images ----
    def _load(maps):
        return maps if hasattr(maps, "dataobj") else nib.load(maps)

    ho_cort_img = _load(ho_cort["maps"])
    ho_sub_img = _load(ho_sub["maps"])
    juelich_img = _load(juelich["maps"])

    ho_cort_data = np.asarray(ho_cort_img.dataobj)
    ho_sub_data = np.asarray(ho_sub_img.dataobj)
    juelich_data = np.asarray(juelich_img.dataobj)

    assert ho_cort_data.shape == ho_sub_data.shape == juelich_data.shape, \
        "Atlas shapes do not match — cannot combine"
    assert np.allclose(ho_cort_img.affine, juelich_img.affine), \
        "Atlas affines do not match — not in the same MNI space"

    # ---- Combine: lowest priority first, highest last ----
    combined = np.zeros(ho_cort_data.shape, dtype=np.int32)
    combined_labels = {}

    # Layer 1: Harvard-Oxford Cortical (label IDs 1–48)
    for i in range(1, len(ho_cort["labels"])):
        combined[ho_cort_data == i] = i
        combined_labels[str(i)] = str(ho_cort["labels"][i])

    # Layer 2: Harvard-Oxford Subcortical (label IDs 100+)
    for i in range(1, len(ho_sub["labels"])):
        name = str(ho_sub["labels"][i])
        # Skip overly broad labels
        if "Cortex" in name or "White Matter" in name:
            continue
        label_id = 100 + i
        combined[ho_sub_data == i] = label_id
        combined_labels[str(label_id)] = name

    # Layer 3: Julich-Brain (label IDs 200+, overwrites everything)
    for i in range(1, len(juelich["labels"])):
        label_id = 200 + i
        combined[juelich_data == i] = label_id
        name = str(juelich["labels"][i])
        # Clean up prefixes
        if name.startswith("GM "):
            name = name[3:]
        elif name.startswith("WM "):
            name = "WM: " + name[3:]
        combined_labels[str(label_id)] = name

    # ---- Save ----
    combined_img = nib.Nifti1Image(combined, ho_cort_img.affine)
    nib.save(combined_img, str(out_nii))

    with open(out_labels, "w", encoding="utf-8") as f:
        json.dump(combined_labels, f, indent=2, ensure_ascii=False)

    n_labels = len(combined_labels)
    coverage = int((combined > 0).sum())
    total = int(combined.size)
    pct = 100 * coverage / total

    print(f"\n[setup] Combined atlas saved:")
    print(f"  NIfTI:  {out_nii}  ({os.path.getsize(out_nii):,} bytes)")
    print(f"  Labels: {out_labels}  ({n_labels} labels)")
    print(f"  Coverage: {coverage:,} / {total:,} voxels ({pct:.1f}%)")
    print(f"  Sources:")
    n_ho_c = sum(1 for k in combined_labels if 1 <= int(k) <= 99)
    n_ho_s = sum(1 for k in combined_labels if 100 <= int(k) <= 199)
    n_jue = sum(1 for k in combined_labels if 200 <= int(k) <= 299)
    print(f"    Harvard-Oxford Cortical:    {n_ho_c} labels")
    print(f"    Harvard-Oxford Subcortical: {n_ho_s} labels")
    print(f"    Julich-Brain:               {n_jue} labels")


if __name__ == "__main__":
    main()
