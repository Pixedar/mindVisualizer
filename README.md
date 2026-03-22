# mindVisualizer

A tool for exploring how information may move through the brain at rest, using both a continuous flow model and a raw effective connectivity graph.

## What does it do?

mindVisualizer has three modes for looking at resting-state brain dynamics.

The first is a **general information flow model** — an estimated field of how information tends to move through the brain during rest. This is based on my preprint, where **rDCIM** is combined with **anatomical geometry and streamlines** to produce a continuous flow field. In this mode, the system does not just show which regions are connected, but also gives a spatial, dynamic picture of how information may propagate through the brain.

The second is a **raw rDCIM connectivity mode**. Here, the brain is represented more directly as a graph of connections between ROIs. You can think of it as a 3D graph showing how different brain regions are linked during rest through effective connectivity.

The third is an **ROI flow mode** — a dual-window visualization where the left panel shows particle flow through a learned neural manifold, and the right panel shows the corresponding ROI activation pattern. As you trace a path through the manifold, the system maps each position to a brain-region activation vector using kNN interpolation, and the LLM interprets what the resulting flow pattern means.

## Flow mode
The flow mode is designed for interactive exploration.

You can press **`G`** to place a probe somewhere in the flow field. Once placed, the probe is carried by the flow itself, tracing a path through the brain as if it were "grabbed" by the underlying information dynamics.

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

## ROI flow mode

In this mode, you explore a learned neural manifold (not the brain directly). Each position in this manifold corresponds to a particular pattern of brain region activations, derived from resting-state fMRI data.

You place a probe in the manifold, let it follow the flow, and the right panel updates in real time to show which brain regions are active or suppressed at each point along the path. When you freeze the probe (`Shift+G`), the system computes the delta between start and end ROI activations and asks the LLM to interpret what this specific flow pattern means — which networks were affected, what direction the information flowed, and what cognitive process could produce it.

---

## Installation

```bash

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
#.venv\Scripts\activate  # Windows

pip install -e .

#This downloads the **Allen Human Brain Atlas** via BrainGlobe and builds cached lookup data
pip install brainglobe-atlasapi
python setup_brain_data.py

# Extra parcellation (optional but strongly recommended - finer subregion labels in flow mode)
python setup_brain_data.py



```


### API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-key-here
```

By default the app uses **GPT-5.4-mini** (fast and cheap). Pass `--hq` to switch to **GPT-5.4** for higher-quality interpretations.

---

## Data

Most data is already included in the repo or generated during setup. You only need to manually manage these:

### API key (`.env`)
Required for all LLM features. See Installation above.

### Brain states (`data/brain_states_rdcim.json`)
Auto-generated when you run `--global-state`. You can manually edit this JSON to set expert-level neuroscience descriptions for each ROI.

### RAG knowledge base (`data/rag_knowledge/`)
Drop JSON or TXT files here to improve LLM interpretations. The default includes 24 brain region descriptions. Add your own medical-grade knowledge to get better results. See `data/rag_knowledge/README.md` for format docs.

### ROI flow data (`data/roi_flow/`)
Download with `python scripts/download_roi_flow_data.py`. Source: [HuggingFace](https://huggingface.co/datasets/Pixedar/mindVisualizer-roi-flow-data)

### Extra parcellation (`data/extra_parcellation/`)
Run `python scripts/setup_extra_parcellation.py` to build a combined atlas (127 regions from Harvard-Oxford + Julich-Brain). Auto-detected by flow mode. See `data/extra_parcellation/README.md`.

---

## Usage

### MDN flow mode

```bash
# Basic flow visualization (recommended: --no-rag, see note below)
python -m src.main --no-rag --hq

```
> **RAG note:**  For best results, add more medical-grade knowledge to `data/rag_knowledge/`. Alternatively, use `--no-rag` to skip RAG entirely and let the LLM use its own training knowledge.
### Raw rDCIM mode

```bash

# If this is first execution pre-initialize global brain state, otherwise it will be empty and the LLM won't have any context to interpret perturbations. It auto fills rois based on some global brain state description
python examples/rdcim_propagation.py --hq --global-state "someone feeling anxious"

# Interactive graph visualization
python examples/rdcim_propagation.py --hq

```
> **Global state note:** The `--global-state` currently is using the llm to auto fill the inital states of roi, but for best results its recommend to manually create or edit data\brain_states_rdcim.json with expert-level neuroscience descriptions for each ROI

### ROI flow mode

```bash
# Download data first (one-time)
pip install huggingface_hub
python scripts/download_roi_flow_data.py

# Run (defaults point to data/roi_flow/)
python examples/roi_flow_mode.py
python examples/roi_flow_mode.py --hq --debug
```



---

## Controls

### Flow mode
1. **`G` + click** — place a probe in the flow field
2. **`Shift + G`** — ask the LLM to interpret the probe trajectory
3. **`C`** — clear all probes
Advanced:
 **`B`** — toggle branching mode
 **`+/-`** — speed scale
 **`S`** — initialize brain states
 **`Shift + S`** — propagate state changes through probe path

> **Note:** Place the probe a bit deeper into the brain for best results. Near-surface placements may not be picked up by the flow.

### Raw rDCIM mode
1. **click ROI** — select a parcel
2. **`P`** — propose perturbations for selected ROI
3. **`Shift + P`** — propagate perturbation through graph

### ROI flow mode
1. **`G` + click** — place probe in manifold
2. **`Shift + G`** — freeze probe, compute ROI delta, LLM interpretation
3. **`C`** — clear probe and ROI display
4. **`+/-`** — speed scale

---

## Debug mode and LLM prompts

Pass `--debug` to any script to print the full LLM prompt to console before each call. Useful for tuning prompts or debugging responses.

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

- **Schaefer et al. (2018)** — Schaefer-400 cortical parcellation
- **Frässle et al. (2021)** — regression dynamic causal modeling (rDCM)
- **Royer et al. (2022)** — MICA-MICs
- **Van Essen et al. (2013)** — Human Connectome Project

### Structural geometry

- **Yeh et al. (2022)** — population-based tract-to-region connectome / HCP-1065 tractography atlas

### Anatomical meshes

- **Allen Human Brain Atlas** (`allen_human_500um`) via **BrainGlobe AtlasAPI**

### Extra parcellation atlases (optional)

The built-in combined atlas merges Harvard-Oxford (Desikan et al., 2006) and Julich-Brain (Amunts et al., 2020) for broad cortical/subcortical coverage with cytoarchitectonic detail. Custom NIfTI atlases in MNI space are also supported.

## Limitations

- **Not medical or clinical advice.** This is a research and visualization tool.
- The flow field is a **model-based approximation**, not a direct measurement of neural signal transmission.
- The LLM interpretations are **neuroscience-informed explanations**, not ground truth.
- **RAG quality matters.** Add curated medical knowledge to `data/rag_knowledge/` for better LLM interpretations.
- **Preprint status.** The underlying research is a preprint and has not yet undergone peer review. However, the app works with any MDN flow field — if you supply a better one, everything still works.

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
