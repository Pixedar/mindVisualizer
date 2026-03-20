# mindVisualizer

A tool for exploring how information may move through the brain at rest, using both a continuous flow model and a raw effective connectivity graph.

## What does it do?

mindVisualizer has two complementary modes for looking at resting-state brain dynamics.

The first is a **general information flow model** — an estimated field of how information tends to move through the brain during rest. This is based on my preprint, where **rDCIM** is combined with **anatomical geometry and streamlines** to produce a continuous flow field. In this mode, the system does not just show which regions are connected, but also gives a spatial, dynamic picture of how information may propagate through the brain.

The second is a **raw rDCIM connectivity mode**. Here, the brain is represented more directly as a graph of connections between ROIs. You can think of it as a 3D graph showing how different brain regions are linked during rest through effective connectivity.

## Flow mode
The flow mode is designed for interactive exploration.

You can press **`G`** to place a probe somewhere in the flow field. Once placed, the probe is carried by the flow itself, tracing a path through the brain as if it were “grabbed” by the underlying information dynamics.

Then, by pressing **`Shift + G`**, you can ask the LLM to explain the meaning of that exact flow trajectory. The model does this by identifying which anatomical regions the probe passed through and interpreting what that sequence of regions could mean based on neuroscience knowledge retrieved from the RAG database. In other words, it tries to explain what kind of information transfer this path may correspond to, and what functional role it could reflect in the resting brain.

This path-based interpretation is conceptually similar to the mechanism used in my related project, [TraceScope](https://github.com/Pixedar/TraceScope).

> The MDN flow field is built from: [https://zenodo.org/records/18200415](https://zenodo.org/records/18200415)

![Flow mode demo](assets/gifA.gif)

## Raw rDCIM mode
In the raw **rDCIM** version, the interaction is different.

Here, you can initialize each ROI in the connectivity graph with some state. For example, you might assign a visual region a state like *currently processing a human face*, and do this across multiple regions to build a rough simulation of what the brain is doing at a given moment.

You can then select any ROI and perturb its state. The system propagates that perturbation through the rDCIM connectivity graph in real time, showing how the change spreads to other regions according to the effective connectivity structure.

At the end, the LLM provides an interpretation of what this perturbation changed in the broader brain network.

![Raw rDCIM demo](assets/gifB.gif)
---

## Installation

```bash
git clone <repo-url>
cd mindVisualizer
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -e .
```

### API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

By default the app uses **GPT-5.4-mini** — a fast, cost-efficient model suitable for testing and general use. For higher-quality neuroscience interpretations, pass `--hq` to switch to **GPT-5.4** (see Usage below).

> **Note:** Without `--hq`, all LLM calls use **GPT-5.4-mini**. This is recommended for testing or if you want to avoid spending on expensive tokens. The `--hq` flag switches every LLM call to **GPT-5.4**, which produces significantly better neuroscience interpretations but costs more.

---

## Data required

### 1) MDN flow artifact (alredy included in the repo)

Place these files in `data/mdn/`:

- `mdn_particles_rdcim_teacher_edge_hq_grid125_meta.json`
- `mdn_particles_rdcim_teacher_edge_hq_training_points.npy`
- the 9 binary grid files referenced by the metadata JSON

This is the precomputed MDN flow artifact used for the continuous flow mode.

### 2) Brain region meshes and labels

This project uses:

- custom STL region meshes in `data/meshes/`
- Allen Human Brain Atlas OBJ meshes in `data/meshes_obj/`
- cached anatomical lookup data generated during setup

First-time setup:

```bash
pip install brainglobe-atlasapi
python setup_brain_data.py
```

This downloads the **Allen Human Brain Atlas** via BrainGlobe (`allen_human_500um`), prepares meshes, and builds a cached voxel label grid for fast anatomical region lookup.

### 3) rDCIM graph data (~636 KB)

Place in `data/`:

- `sch400_rDCM_A.npy` — 400×400 directed effective connectivity matrix
- `schaefer400_centroids_MNI.npy` — 400×3 ROI centroid coordinates in MNI space

### 4) Other supporting files

- `data/brain_alignment.json` — spatial transform / alignment matrix
- `data/structures.json` — atlas structure name mappings

---

## Usage

### MDN flow mode

```bash
# Basic flow visualization (recommended: --no-rag, see note below)
# -- hq High-quality model for better interpretations
python -m src.main --no-rag --hq

```

### Raw rDCIM mode

```bash
# Interactive graph visualization
python examples/rdcim_propagation.py --hq

# Pre-initialize global brain state
python examples/rdcim_propagation.py --hq --global-state "someone feeling anxious"

```

> **RAG note:** I currently recommend using `--no-rag` for the flow mode. The RAG database is implemented but not yet populated with medical-grade neuroscience knowledge — this is underway. Use caution: this feature is still in a testing stage.

---

## Controls

### Flow mode

- **`G` + click** — place a probe in the flow field
- **`Shift + G`** — ask the LLM to interpret the recorded probe trajectory
- **`S`** — initialize brain states
- **`Shift + S`** — propagate state changes through the probe path

### Raw rDCIM mode

- **click ROI** — select a parcel / region
- **`P`** — perturb the selected ROI
- **`Shift + P`** — propagate the perturbation through the connectivity graph

---

## Data provenance and attribution

This repository combines several public resources. If you reuse, redistribute, or publish results generated with this project, please credit the original sources below and check their individual licenses / terms of use.

### Flow field source

The continuous flow mode is built on the MDN flow artifact released with my preprint:

Continuous, Tract-Constrained Directional Vector Fields from rDCM Effective Connectivity Using Mixture Density Networks  
  Zenodo record: [https://zenodo.org/records/18200415](https://zenodo.org/records/18200415)

That work describes the fusion of:

- **rDCM effective connectivity on the Schaefer-400 parcellation**
- **HCP-1065 whole-brain tractography geometry**

into a continuous MDN-based directional flow field.

### Effective connectivity and ROI definition

The raw rDCIM graph data used here follows the same setup as the preprint:

- **Schaefer et al. (2018)** — Schaefer-400 cortical parcellation
- **Frässle et al. (2021)** — regression dynamic causal modeling (rDCM)
- rDCIM connectivity inputs derived from publicly available resources including:
  - **Royer et al. (2022)** — MICA-MICs
  - **Van Essen et al. (2013)** — Human Connectome Project

In this repository, the same **rDCIM matrix family** and the same **Schaefer-400 ROI definition / centroids** are used for the direct graph mode and for compatibility with the flow artifact.

### Structural geometry behind the flow artifact

The structural geometry used upstream for the MDN flow generation comes from:

- **Yeh et al. (2022)** — population-based tract-to-region connectome / HCP-1065 tractography atlas

### Anatomical meshes used in the app

For anatomical region lookup and mesh visualization, this repository uses the **Allen Human Brain Atlas** meshes through BrainGlobe:

- **Allen Human Brain Atlas** (`allen_human_500um`) via **BrainGlobe AtlasAPI** / **Allen Brain Map**

These atlas meshes are used here for visualization and anatomical labeling in the app. They are not the source of the MDN flow field itself.
## Limitations

- **Not medical or clinical advice.** This is a research and visualization tool.
- The flow field is a **model-based approximation**, not a direct measurement of neural signal transmission.
- The LLM interpretations are **neuroscience-informed explanations**, not ground truth.
- The MDN field is strongly shaped by anatomical geometry, so it should be understood as an interpretable proxy rather than a literal simulation of biological information flow.
- Allen atlas coverage and mesh-based anatomical lookup are useful for interpretation, but they are not a replacement for task-specific neuroanatomical analysis pipelines.
- **RAG quality is crucial.** The quality of LLM interpretations depends heavily on the RAG knowledge base. The current RAG dataset is a placeholder — it must be populated with curated, medical-grade neuroscience knowledge for best results. This work is underway.
- **Preprint status.** The underlying research (MDN flow field construction from rDCM effective connectivity) is a preprint and has not yet undergone peer review. However, the app itself remains a valid and usable interpretability tool for human brain information flow even with different or improved MDN flow fields — if you supply a better-quality flow artifact, the visualization and analysis pipeline will still work.

---

## References
- **Schaefer, A. et al. (2018).** *Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI*. Cerebral Cortex.
- **Frässle, S. et al. (2021).** *Regression dynamic causal modeling for resting-state fMRI*. Human Brain Mapping.
- **Royer, J. et al. (2022).** *An Open MRI Dataset For Multiscale Neuroscience*. Scientific Data.
- **Van Essen, D. C. et al. (2013).** *The WU-Minn Human Connectome Project: An overview*. NeuroImage.
- **Yeh, F.-C. et al. (2022).** *Population-based tract-to-region connectome of the human brain*. Nature Communications.

---

## License

Research use. See the original data sources and their respective licenses / terms for any reused external assets.
