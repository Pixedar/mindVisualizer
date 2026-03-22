"""ROI Flow mode: manifold-to-ROI mapping and LLM interpretation.

Maps probe positions in a learned neural manifold to ROI activation vectors
using k-nearest-neighbor interpolation, then analyzes how the ROI pattern
changes along the probe path.

This module is used by examples/roi_flow_mode.py for the dual-window
manifold + ROI visualization.
"""

import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Manifold-to-ROI mapper
# ---------------------------------------------------------------------------

class ManifoldToROIKNN:
    """Map manifold positions to ROI activation vectors via kNN + Gaussian weighting.

    Given a library of (embedding_point, roi_vector) pairs, interpolates
    the ROI vector at any manifold position using Gaussian-weighted kNN.
    """

    def __init__(self, embed_points: np.ndarray, roi_vectors: np.ndarray,
                 k: int = 256, sigma: float = 0.0):
        """
        Args:
            embed_points: (N, D) embedding coordinates (typically D=3).
            roi_vectors: (N, R) ROI activation vectors.
            k: number of nearest neighbors for interpolation.
            sigma: Gaussian bandwidth. If 0, auto-computed from median kNN distance.
        """
        assert embed_points.shape[0] == roi_vectors.shape[0], \
            f"embed ({embed_points.shape[0]}) and roi ({roi_vectors.shape[0]}) must match"
        self.X = embed_points.astype(np.float32)
        self.Y = roi_vectors.astype(np.float32)
        self.k = int(max(8, k))
        self.sigma = float(max(0.0, sigma))
        self.n_rois = self.Y.shape[1]
        self.tree = cKDTree(self.X)

    def query(self, point: np.ndarray) -> np.ndarray:
        """Interpolate ROI vector at a manifold position.

        Args:
            point: (D,) or (1,D) position in manifold space.

        Returns:
            (R,) ROI activation vector.
        """
        p = point.reshape(1, -1).astype(np.float32)
        k_actual = min(self.k, len(self.X))
        d, idx = self.tree.query(p, k=k_actual)
        d = np.asarray(d).ravel().astype(np.float32)
        idx = np.asarray(idx).ravel()

        if len(idx) == 0:
            return np.zeros(self.n_rois, dtype=np.float32)

        sig = self.sigma if self.sigma > 0 else float(np.median(d) + 1e-9)
        w = np.exp(-(d ** 2) / (2.0 * sig ** 2)).astype(np.float64)
        sw = float(np.sum(w)) + 1e-12
        mu = (w[:, None] * self.Y[idx]).sum(axis=0) / sw
        return mu.astype(np.float32)


# ---------------------------------------------------------------------------
# ROI flow analyzer
# ---------------------------------------------------------------------------

class ROIFlowAnalyzer:
    """Analyze ROI delta vectors to extract flow patterns for LLM interpretation."""

    def __init__(self, roi_names: list[str], roi_centers: np.ndarray):
        """
        Args:
            roi_names: (R,) human-readable ROI names.
            roi_centers: (R, 3) MNI coordinates of ROI centers.
        """
        self.roi_names = roi_names
        self.roi_centers = roi_centers.astype(np.float32)
        self.n_rois = len(roi_names)

    def compute_delta(self, start_roi: np.ndarray, end_roi: np.ndarray) -> np.ndarray:
        """Compute change in ROI activation from start to end of path."""
        return (end_roi - start_roi).astype(np.float32)

    def analyze_flow_pattern(self, delta: np.ndarray) -> dict:
        """Extract structured information about the ROI flow pattern.

        Returns dict with:
            top_positive: list of (name, delta_val, center) for top increased ROIs
            top_negative: list of (name, delta_val, center) for top decreased ROIs
            direction: dict with anterior_posterior, left_right, superior_inferior scores
            pattern_type: "one_to_many", "many_to_one", "distributed", "bilateral_split"
            bulk_direction: human-readable description of dominant flow direction
        """
        abs_delta = np.abs(delta)
        threshold = np.percentile(abs_delta, 85)

        # Top changed ROIs
        significant = abs_delta > threshold
        pos_mask = (delta > 0) & significant
        neg_mask = (delta < 0) & significant

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        # Sort by magnitude
        pos_sorted = pos_indices[np.argsort(-delta[pos_indices])]
        neg_sorted = neg_indices[np.argsort(delta[neg_indices])]

        top_positive = [(self.roi_names[i], float(delta[i]),
                         self.roi_centers[i].tolist()) for i in pos_sorted[:10]]
        top_negative = [(self.roi_names[i], float(delta[i]),
                         self.roi_centers[i].tolist()) for i in neg_sorted[:10]]

        # Compute directional bias using weighted centroids
        pos_weighted_center = np.zeros(3)
        neg_weighted_center = np.zeros(3)
        if len(pos_indices) > 0:
            w = delta[pos_indices]
            pos_weighted_center = np.average(self.roi_centers[pos_indices], axis=0,
                                             weights=w)
        if len(neg_indices) > 0:
            w = np.abs(delta[neg_indices])
            neg_weighted_center = np.average(self.roi_centers[neg_indices], axis=0,
                                             weights=w)

        # Direction: from negative (source) to positive (target) centroids
        flow_vec = pos_weighted_center - neg_weighted_center
        direction = {
            "left_right": float(flow_vec[0]),      # +X = right
            "anterior_posterior": float(flow_vec[1]),  # +Y = anterior
            "superior_inferior": float(flow_vec[2]),   # +Z = superior
        }

        # Pattern type
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        if n_neg <= 3 and n_pos > 8:
            pattern_type = "one_to_many"
        elif n_pos <= 3 and n_neg > 8:
            pattern_type = "many_to_one"
        elif n_pos > 0 and n_neg > 0:
            # Check bilateral split
            pos_x = self.roi_centers[pos_indices, 0]
            neg_x = self.roi_centers[neg_indices, 0]
            pos_mean_x = float(np.mean(pos_x))
            neg_mean_x = float(np.mean(neg_x))
            if abs(pos_mean_x - neg_mean_x) > 20:  # significant L/R separation
                pattern_type = "bilateral_split"
            else:
                pattern_type = "distributed"
        else:
            pattern_type = "distributed"

        # Human-readable bulk direction
        parts = []
        if abs(direction["anterior_posterior"]) > 10:
            parts.append("anterior" if direction["anterior_posterior"] > 0 else "posterior")
        if abs(direction["left_right"]) > 10:
            parts.append("right" if direction["left_right"] > 0 else "left")
        if abs(direction["superior_inferior"]) > 10:
            parts.append("superior" if direction["superior_inferior"] > 0 else "inferior")
        bulk_direction = " and ".join(parts) if parts else "no dominant direction"

        return {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "direction": direction,
            "pattern_type": pattern_type,
            "bulk_direction": bulk_direction,
            "n_significant_positive": n_pos,
            "n_significant_negative": n_neg,
        }

    def build_llm_context(self, delta: np.ndarray,
                          path_regions: list[str] | None = None) -> str:
        """Build a structured text context for LLM interpretation.

        Args:
            delta: (R,) ROI delta vector.
            path_regions: optional list of brain region names the probe traversed.

        Returns:
            Formatted context string for the LLM prompt.
        """
        analysis = self.analyze_flow_pattern(delta)

        lines = []
        lines.append("=== ROI FLOW ANALYSIS ===\n")

        # Pattern overview
        lines.append(f"Flow pattern type: {analysis['pattern_type']}")
        lines.append(f"Bulk information flow direction: {analysis['bulk_direction']}")
        lines.append(f"Significant ROIs with increased activation: "
                      f"{analysis['n_significant_positive']}")
        lines.append(f"Significant ROIs with decreased activation: "
                      f"{analysis['n_significant_negative']}")
        lines.append("")

        # Top receivers (positive delta)
        if analysis["top_positive"]:
            lines.append("TOP RECEIVING ROIs (activation INCREASED):")
            for name, val, center in analysis["top_positive"][:8]:
                side = "left" if center[0] < 0 else "right"
                depth = "anterior" if center[1] > 0 else "posterior"
                lines.append(f"  - {name} ({side}, {depth}): delta = +{val:.4f}, "
                              f"MNI = ({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})")
            lines.append("")

        # Top donors (negative delta)
        if analysis["top_negative"]:
            lines.append("TOP DONOR ROIs (activation DECREASED):")
            for name, val, center in analysis["top_negative"][:8]:
                side = "left" if center[0] < 0 else "right"
                depth = "anterior" if center[1] > 0 else "posterior"
                lines.append(f"  - {name} ({side}, {depth}): delta = {val:.4f}, "
                              f"MNI = ({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})")
            lines.append("")

        # Path context
        if path_regions:
            lines.append("MANIFOLD PATH TRAVERSED THROUGH THESE REGIONS:")
            for i, r in enumerate(path_regions, 1):
                lines.append(f"  {i}. {r}")
            lines.append("")

        # Directional summary
        d = analysis["direction"]
        lines.append("DIRECTIONAL ANALYSIS:")
        lines.append(f"  Left-Right shift: {d['left_right']:.1f} mm "
                      f"({'rightward' if d['left_right'] > 0 else 'leftward'})")
        lines.append(f"  Anterior-Posterior shift: {d['anterior_posterior']:.1f} mm "
                      f"({'anterior' if d['anterior_posterior'] > 0 else 'posterior'})")
        lines.append(f"  Superior-Inferior shift: {d['superior_inferior']:.1f} mm "
                      f"({'superior' if d['superior_inferior'] > 0 else 'inferior'})")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ROI flow LLM interpreter
# ---------------------------------------------------------------------------

class ROIFlowLLM:
    """Send ROI flow analysis to LLM for interpretation."""

    def __init__(self, model: str = "gpt-5.4-mini", debug: bool = False):
        self.model = model
        self.debug = debug

    def interpret_roi_flow(self, context: str) -> str:
        """Interpret an ROI flow pattern using the LLM.

        Args:
            context: structured text from ROIFlowAnalyzer.build_llm_context()

        Returns:
            LLM interpretation text.
        """
        from src.region_analyzer import _ensure_ssl
        _ensure_ssl()

        from dotenv import load_dotenv
        load_dotenv()

        instructions = (
            "You are a neuroscientist interpreting brain state dynamics from a "
            "manifold flow simulation. The user traced a path through a learned "
            "neural manifold (a dimensionality-reduced representation of resting-state "
            "brain dynamics). You are given how each ROI's contribution changed along "
            "this path.\n\n"
            "IMPORTANT: The delta values do NOT mean regions became more or less "
            "'active' in a simple sense. They measure how each ROI's CONTRIBUTION "
            "to the overall brain state shifted — some regions contribute more to the "
            "new state, some less. A positive delta means the region became a stronger "
            "contributor; negative means it became a weaker contributor. This is a "
            "transition in the brain's dynamic state.\n\n"
            "Analyze what this particular state transition might mean. Do NOT list "
            "the individual ROI changes — instead, synthesize them into ONE coherent "
            "picture of what cognitive or neural process could underlie this specific "
            "shift in brain dynamics. Consider the spatial pattern (which networks "
            "gained vs lost contribution), the directionality, and what spontaneous "
            "resting-state process would produce this exact transition.\n\n"
            "Keep your response SHORT — 2-3 concise paragraphs maximum."
        )

        question = context

        if self.debug:
            import sys
            print(f"\n{'='*60}")
            print(f"[DEBUG PROMPT] ROIFlowLLM.interpret_roi_flow")
            print(f"{'='*60}")
            print(f"INSTRUCTIONS:\n{instructions}")
            print(f"\nINPUT:\n{question}")
            print(f"{'='*60}\n")
            sys.stdout.flush()

        try:
            from openai import OpenAI
            client = OpenAI(timeout=60.0)

            resp = client.responses.create(
                model=self.model,
                instructions=instructions,
                input=question,
                reasoning={"effort": "low"},
                max_output_tokens=2500,
            )

            text = (resp.output_text or "").strip()
            if text:
                return text

            if getattr(resp, "status", None) == "incomplete":
                reason = getattr(resp.incomplete_details, "reason", "unknown")
                return f"[GPT ERROR] Responses API incomplete: {reason}"

            return "[GPT ERROR] Responses API returned no visible text."

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[GPT ERROR] {e}"
