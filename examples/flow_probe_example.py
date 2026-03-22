#!/usr/bin/env python3
"""Example: MDN Flow Probe with brain state propagation.

Demonstrates the flow-based visualization mode:
  1. Launch particle flow with default settings
  2. Place probe(s) in the flow field
  3. Watch probes follow the flow and highlight brain regions
  4. Analyze transitions with GPT
  5. (Optional) Initialize brain states and propagate perturbations

Usage:
  # Basic flow visualization with single probe
  python examples/flow_probe_example.py

  # Multi-probe with branching (for state propagation)
  python examples/flow_probe_example.py --multi-probe 4 --branching

  # Skip RAG, use GPT's own knowledge
  python examples/flow_probe_example.py --no-rag

  # High-quality model with debug (prints LLM prompts)
  python examples/flow_probe_example.py --no-rag --hq --debug

  # Extra parcellation is auto-detected if you run the setup script:
  #   python scripts/setup_extra_parcellation.py
  # Or provide a custom NIfTI atlas:
  #   python examples/flow_probe_example.py --extra-parcellation my_atlas.nii.gz

  # State propagation mode (4 probes + branching)
  python examples/flow_probe_example.py --multi-probe 4 --branching

Interactive keys:
  g           → place probe (then click)
  G (shift)   → analyze path with GPT (async, non-blocking)
  b           → toggle branching (MDN component uncertainty splitting)
  n           → cycle multi-probe count (1/4/8)
  s           → initialize brain region states
  S (shift)   → propagate state changes through probe path
  c           → clear all probes
"""

import sys
from pathlib import Path

# This example simply launches the main visualizer with recommended defaults.
# All the probe, branching, multi-probe, and brain state features are built
# into the main module.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.main import main

if __name__ == "__main__":
    main()
