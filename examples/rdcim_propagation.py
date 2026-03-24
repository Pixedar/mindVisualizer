#!/usr/bin/env python3
"""rDCIM Direct Propagation Visualizer.

Uses the rDCIM (regression Dynamic Causal Modeling) effective connectivity
matrix directly to simulate information propagation between brain regions.

This is an alternative to the MDN flow-based visualization: instead of following
a continuous vector field, we use the discrete ROI-to-ROI connectivity matrix
to model how perturbations in one region propagate through the brain network.

Usage:
  python examples/rdcim_propagation.py
  python examples/rdcim_propagation.py --global-state "someone feeling anxious"
  python examples/rdcim_propagation.py --perturb 42 --perturbation "sudden fear response"

Key bindings:
  Click         select ROI to perturb
  p             perturb selected ROI (prompts in console)
  P (shift+p)   propagate perturbation through network
  s             initialize brain states from global state
  r             reset all states
  +/-           increase/decrease connection threshold
  t             toggle connection labels
  Escape        quit
"""

import argparse
import json
import os
import sys

from pathlib import Path

import numpy as np
import vtk

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.brain_state import BrainStateDB


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RDCIM = DATA_DIR / "sch400_rDCM_A.npy"
DEFAULT_CENTROIDS = DATA_DIR / "schaefer400_centroids_MNI.npy"
DEFAULT_STATE_FILE = DATA_DIR / "brain_states_rdcim.json"

# Schaefer 400 network names (7-network parcellation)
NETWORK_COLORS = {
    "Vis": (0.55, 0.0, 0.75),     # purple
    "SomMot": (0.0, 0.4, 0.8),    # blue
    "DorsAttn": (0.0, 0.7, 0.0),  # green
    "SalVentAttn": (0.8, 0.0, 0.5), # magenta
    "Limbic": (0.9, 0.7, 0.0),    # gold
    "Cont": (0.9, 0.4, 0.0),      # orange
    "Default": (0.8, 0.2, 0.2),   # red
}


def _ensure_ssl():
    try:
        import certifi
        cert_file = certifi.where()
        cur = os.environ.get("SSL_CERT_FILE", "")
        if not cur or not os.path.isfile(cur):
            os.environ["SSL_CERT_FILE"] = cert_file
        cur2 = os.environ.get("REQUESTS_CA_BUNDLE", "")
        if not cur2 or not os.path.isfile(cur2):
            os.environ["REQUESTS_CA_BUNDLE"] = cert_file
    except ImportError:
        pass


def load_rdcim(path: Path) -> np.ndarray:
    """Load rDCIM effective connectivity matrix."""
    if not path.exists():
        raise FileNotFoundError(f"Missing rDCIM file: {path}")
    A = np.load(path)
    A = np.asarray(A, dtype=np.float32)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise RuntimeError(f"rDCIM must be square, got {A.shape}")
    print(f"[rDCIM] loaded {A.shape[0]}x{A.shape[1]} matrix")
    return A


def load_centroids(path: Path) -> np.ndarray:
    """Load ROI centroid positions in MNI space."""
    if not path.exists():
        raise FileNotFoundError(f"Missing centroids: {path}")
    P = np.load(path).astype(np.float32)
    if P.ndim != 2 or P.shape[1] not in (2, 3):
        raise RuntimeError(f"Centroids must be Nx3, got {P.shape}")
    if P.shape[1] == 2:
        P = np.c_[P, np.zeros((P.shape[0], 1), np.float32)]
    print(f"[centroids] loaded {P.shape[0]} ROIs")
    return P


def get_roi_names(R: int) -> list[str]:
    """Generate Schaefer-style ROI names."""
    names = []
    for i in range(R):
        hemi = "LH" if i < R // 2 else "RH"
        names.append(f"{hemi}_ROI_{i+1:03d}")
    return names


def map_rois_to_regions(centroids: np.ndarray, roi_names: list[str]) -> list[str]:
    """Map each ROI centroid to its nearest Allen atlas brain region.

    Uses a voxel label grid (same as mesh_overlay) for robust mapping.
    Returns updated ROI names like 'LH_ROI_042 (precentral gyrus)'.
    """
    from src.mesh_overlay import FlowMeshOverlay
    import vtk

    alignment_file = DATA_DIR / "brain_alignment.json"
    mesh_dir = DATA_DIR / "meshes"

    # Dummy renderer/window (no display needed)
    ren = vtk.vtkRenderer()
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)

    try:
        overlay = FlowMeshOverlay(ren=ren, win=win,
                                   mesh_dir=mesh_dir,
                                   alignment_file=alignment_file)
        grid_cache = DATA_DIR / "label_grid_cache.npz"
        if not overlay.load_label_grid(grid_cache):
            overlay.build_label_grid()
            overlay.save_label_grid(grid_cache)
    except Exception as e:
        print(f"[roi-map] Could not build label grid: {e}")
        return roi_names

    mapped_names = []
    matched = 0
    for i, (name, pos) in enumerate(zip(roi_names, centroids)):
        key = overlay.get_region_at_point(pos)
        if key is None:
            key = overlay.find_nearest_region(pos, search_radius=4)
        if key is not None:
            region_name = overlay.get_region_name(key)
            hemi = overlay.get_hemisphere_label(key, pos)
            hemi_tag = f", {hemi}" if hemi else ""

            # Add relative position within the region
            center = overlay.get_mesh_center(key)
            bounds = overlay.get_mesh_bounds(key)
            pos_tag = ""
            if center is not None and bounds is not None:
                diff = pos - center
                extent = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2],
                                   bounds[5]-bounds[4]])
                extent = np.maximum(extent, 1e-6)
                rel = diff / (extent * 0.5)
                parts = []
                if abs(rel[2]) > 0.3:
                    parts.append("dorsal" if rel[2] > 0 else "ventral")
                if abs(rel[0]) > 0.3:
                    parts.append("lateral" if abs(rel[0]) > 0.5 else "medial")
                if abs(rel[1]) > 0.3:
                    parts.append("anterior" if rel[1] > 0 else "posterior")
                if parts:
                    pos_tag = f", near {' '.join(parts)} section"
                else:
                    pos_tag = ", near center"

            mapped_names.append(f"{name} ({region_name}{hemi_tag}{pos_tag})")
            matched += 1
        else:
            # Still add hemisphere based on x-coordinate for unmapped ROIs
            hemi_label = "left hemisphere" if pos[0] < 0 else "right hemisphere"
            mapped_names.append(f"{name} ({hemi_label})")

    print(f"[roi-map] Mapped {matched}/{len(roi_names)} ROIs to Allen regions")
    return mapped_names


def get_strongest_connections(A: np.ndarray, source_idx: int,
                               top_k: int = 20, depth: int = 2) -> list[dict]:
    """Get strongest outgoing connections from source, with multi-hop propagation.

    Args:
        A: connectivity matrix (R x R)
        source_idx: source ROI index
        top_k: max connections per level
        depth: propagation depth

    Returns:
        List of {target_idx, weight, depth, path}
    """
    R = A.shape[0]
    Ac = A.copy()
    np.fill_diagonal(Ac, 0.0)

    results = []
    visited = {source_idx}
    current_level = [source_idx]

    for d in range(1, depth + 1):
        next_level = []
        for src in current_level:
            weights = Ac[src]
            abs_w = np.abs(weights)
            # Get top-k strongest
            if len(abs_w) > top_k:
                top_idx = np.argpartition(abs_w, -top_k)[-top_k:]
            else:
                top_idx = np.arange(len(abs_w))
            top_idx = top_idx[abs_w[top_idx] > 0]

            for ti in top_idx:
                if ti in visited:
                    continue
                results.append({
                    "target_idx": int(ti),
                    "weight": float(weights[ti]),
                    "abs_weight": float(abs_w[ti]),
                    "depth": d,
                    "source_idx": int(src),
                })
                visited.add(int(ti))
                next_level.append(int(ti))

        current_level = next_level
        if not current_level:
            break

    # Sort by absolute weight
    results.sort(key=lambda x: x["abs_weight"], reverse=True)
    return results


class RDCIMVisualizer:
    """Interactive VTK visualizer for rDCIM connectivity."""

    def __init__(self, A: np.ndarray, centroids: np.ndarray,
                 roi_names: list[str], brain_state_db: BrainStateDB):
        self.A = A
        self.centroids = centroids
        self.R = A.shape[0]
        self.roi_names = roi_names
        self.brain_state_db = brain_state_db
        self.selected_roi = None
        self.connection_threshold = 0.02  # fraction of max |A|
        self.show_labels = False
        self._connection_actors = []
        self._label_actors = []
        self._roi_actors = []
        self._highlight_actor = None
        self._state_label_actors = []  # 3D text labels showing states next to ROIs
        self._propagation_line_actors = []  # animated connection lines during propagation
        self._perturb_proposals = None   # list of 4 proposals when ready
        self._perturb_region = None      # region name being perturbed
        self._perturb_waiting = False    # True while waiting for user to press 1-5
        self._anim_steps = []            # propagation steps for looped animation
        self._anim_source_idx = None     # source ROI for animation
        self._anim_frame = 0             # current animation frame
        self._anim_playing = False       # True while loop animation is active
        self.propagation_depth = 2       # user-adjustable depth for Shift+P
        self.propagation_top_k = 10      # connections per level

        # Compute stats
        Ac = A.copy()
        np.fill_diagonal(Ac, 0.0)
        self.max_weight = float(np.abs(Ac).max()) if Ac.size else 1.0

        # Setup VTK
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.05, 0.05, 0.1)

        self.win = vtk.vtkRenderWindow()
        self.win.AddRenderer(self.ren)
        self.win.SetSize(1400, 900)
        self.win.SetWindowName("rDCIM Propagation Visualizer")

        self._create_roi_spheres()
        self._create_text_overlay()

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)

    def _create_roi_spheres(self):
        """Create sphere actors for each ROI."""
        # Normalize centroids to reasonable scale
        center = self.centroids.mean(axis=0)
        scale = max(1.0, float(np.abs(self.centroids - center).max()))

        for i in range(self.R):
            sphere = vtk.vtkSphereSource()
            pos = self.centroids[i]
            sphere.SetCenter(float(pos[0]), float(pos[1]), float(pos[2]))
            sphere.SetRadius(1.5)
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(12)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Color by network (based on position heuristic)
            r, g, b = 0.6, 0.6, 0.6
            actor.GetProperty().SetColor(r, g, b)
            actor.GetProperty().SetOpacity(0.7)

            self.ren.AddActor(actor)
            self._roi_actors.append(actor)

        self.ren.ResetCamera()

    def _create_text_overlay(self):
        """Create info text actors."""
        self._info_actor = vtk.vtkTextActor()
        self._info_actor.SetInput("Click ROI to select | p: perturb | "
                                  "Shift+P: propagate | s: init states")
        tp = self._info_actor.GetTextProperty()
        tp.SetColor(1, 1, 1)
        tp.SetFontSize(13)
        tp.SetFontFamilyToCourier()
        self._info_actor.SetPosition(10, 10)
        self.ren.AddActor(self._info_actor)

        # Selected ROI info / perturbation options (top-center yellow)
        self._sel_actor = vtk.vtkTextActor()
        self._sel_actor.SetInput("")
        tp2 = self._sel_actor.GetTextProperty()
        tp2.SetColor(1.0, 0.9, 0.3)
        tp2.SetFontSize(14)
        tp2.SetFontFamilyToCourier()
        tp2.SetJustificationToCentered()
        tp2.SetVerticalJustificationToTop()
        self._sel_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self._sel_actor.GetPositionCoordinate().SetValue(0.5, 0.97)
        self.ren.AddActor(self._sel_actor)

        # Propagation summary (top-left, green)
        self._summary_actor = vtk.vtkTextActor()
        self._summary_actor.SetInput("")
        tp3 = self._summary_actor.GetTextProperty()
        tp3.SetColor(0.8, 1.0, 0.8)
        tp3.SetFontSize(11)
        tp3.SetFontFamilyToCourier()
        tp3.SetJustificationToLeft()
        tp3.SetVerticalJustificationToTop()
        self._summary_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self._summary_actor.GetPositionCoordinate().SetValue(0.01, 0.97)
        self._summary_actor.VisibilityOff()
        self.ren.AddActor(self._summary_actor)

        # Information flow story (top-right, light blue)
        self._story_actor = vtk.vtkTextActor()
        self._story_actor.SetInput("")
        tp4 = self._story_actor.GetTextProperty()
        tp4.SetColor(0.7, 0.9, 1.0)
        tp4.SetFontSize(11)
        tp4.SetFontFamilyToCourier()
        tp4.SetJustificationToRight()
        tp4.SetVerticalJustificationToTop()
        self._story_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self._story_actor.GetPositionCoordinate().SetValue(0.99, 0.97)
        self._story_actor.VisibilityOff()
        self.ren.AddActor(self._story_actor)

    def _show_connections(self, source_idx: int, depth: int = 2, top_k: int = 15):
        """Show connections from selected ROI.

        Always draws direct connections (depth=1) from the source ROI first,
        then adds multi-hop connections from intermediate ROIs.
        """
        # Clear old connections
        for act in self._connection_actors:
            self.ren.RemoveActor(act)
        self._connection_actors.clear()

        connections = get_strongest_connections(self.A, source_idx,
                                                top_k=top_k, depth=depth)
        if not connections:
            return

        # Separate by depth to ensure direct connections are always shown
        depth1 = [c for c in connections if c["depth"] == 1]
        depth2 = [c for c in connections if c["depth"] > 1]

        # Show up to 10 direct + 5 multi-hop
        shown = depth1[:10] + depth2[:5]
        if not shown:
            return

        max_w = max(c["abs_weight"] for c in shown)
        src_pos = self.centroids[source_idx]

        info_lines = [f"Selected: {self.roi_names[source_idx]}"]
        state = self.brain_state_db.get(self.roi_names[source_idx])
        if state:
            info_lines.append(f"State: {state[:80]}")
        info_lines.append(f"Connections ({len(depth1)} direct, {len(depth2)} multi-hop):")

        for c in shown:
            ti = c["target_idx"]
            w = c["weight"]
            # For depth-1: line goes from selected ROI to target
            # For depth-2+: line goes from intermediate source to target
            from_idx = c["source_idx"]
            from_pos = self.centroids[from_idx]
            to_pos = self.centroids[ti]

            # Create line
            line = vtk.vtkLineSource()
            line.SetPoint1(float(from_pos[0]), float(from_pos[1]), float(from_pos[2]))
            line.SetPoint2(float(to_pos[0]), float(to_pos[1]), float(to_pos[2]))

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Color: green for positive, red for negative; dimmer for multi-hop
            norm_w = c["abs_weight"] / max(max_w, 1e-9)
            is_multihop = c["depth"] > 1
            if w > 0:
                if is_multihop:
                    actor.GetProperty().SetColor(0.4, 0.7, 0.4)  # dimmer green
                else:
                    actor.GetProperty().SetColor(0.2, 0.9, 0.2)
            else:
                if is_multihop:
                    actor.GetProperty().SetColor(0.7, 0.4, 0.4)  # dimmer red
                else:
                    actor.GetProperty().SetColor(0.9, 0.2, 0.2)

            line_width = max(1.0, 4.0 * norm_w) if not is_multihop else max(1.0, 2.5 * norm_w)
            actor.GetProperty().SetLineWidth(line_width)
            actor.GetProperty().SetOpacity(max(0.3, norm_w * (0.7 if is_multihop else 1.0)))

            self.ren.AddActor(actor)
            self._connection_actors.append(actor)

            # Highlight target ROI
            if ti < len(self._roi_actors):
                if w > 0:
                    self._roi_actors[ti].GetProperty().SetColor(0.2, 0.9, 0.2)
                else:
                    self._roi_actors[ti].GetProperty().SetColor(0.9, 0.2, 0.2)
                opacity = min(1.0, 0.5 + norm_w * 0.5) if not is_multihop else min(0.9, 0.3 + norm_w * 0.4)
                self._roi_actors[ti].GetProperty().SetOpacity(opacity)

            depth_tag = f"d{c['depth']}" if is_multihop else " "
            sign_tag = "inh" if w < 0 else "exc"
            info_lines.append(f"  {depth_tag}: {self.roi_names[ti]} "
                              f"w={w:+.4f} ({sign_tag})")

        self._sel_actor.SetInput("\n".join(info_lines))
        self.win.Render()

    def _clear_state_labels(self):
        """Remove all 3D state text labels and propagation lines."""
        for act in self._state_label_actors:
            self.ren.RemoveActor(act)
        self._state_label_actors.clear()
        for act in self._propagation_line_actors:
            self.ren.RemoveActor(act)
        self._propagation_line_actors.clear()

    @staticmethod
    def _word_wrap(text: str, width: int = 45, max_lines: int = 20) -> str:
        """Word-wrap text to fit in a VTK text actor."""
        lines = []
        for line in text.split("\n"):
            while len(line) > width:
                brk = line.rfind(" ", 0, width)
                if brk <= 0:
                    brk = width
                lines.append(line[:brk])
                line = line[brk:].lstrip()
            lines.append(line)
        return "\n".join(lines[:max_lines])

    def _add_state_label(self, roi_idx: int, text: str, color=(1.0, 0.4, 0.4)):
        """Add a small 3D text label centered on an ROI showing its updated state."""
        pos = self.centroids[roi_idx]

        actor = vtk.vtkBillboardTextActor3D()
        # Truncate to keep labels compact
        short = text[:60] + "..." if len(text) > 60 else text
        actor.SetInput(short)
        # Position directly at the ROI center — billboard text will face camera
        actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        tp = actor.GetTextProperty()
        tp.SetColor(*color)
        tp.SetFontSize(10)
        tp.SetFontFamilyToCourier()
        tp.SetBold(True)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        self.ren.AddActor(actor)
        self._state_label_actors.append(actor)

        # Highlight the ROI sphere in red
        if roi_idx < len(self._roi_actors):
            self._roi_actors[roi_idx].GetProperty().SetColor(*color)
            self._roi_actors[roi_idx].GetProperty().SetOpacity(1.0)

    def _add_propagation_line(self, source_idx: int, target_idx: int,
                                color=(1.0, 0.3, 0.3)):
        """Draw a line from source to target ROI during animated propagation."""
        p1 = self.centroids[source_idx]
        p2 = self.centroids[target_idx]

        pts = vtk.vtkPoints()
        pts.InsertNextPoint(float(p1[0]), float(p1[1]), float(p1[2]))
        pts.InsertNextPoint(float(p2[0]), float(p2[1]), float(p2[2]))

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(3.0)
        actor.GetProperty().SetOpacity(0.8)
        self.ren.AddActor(actor)
        self._propagation_line_actors.append(actor)

    def _select_roi(self, idx: int):
        """Select an ROI."""
        # Reset old selection
        if self.selected_roi is not None and self.selected_roi < len(self._roi_actors):
            self._roi_actors[self.selected_roi].GetProperty().SetColor(0.6, 0.6, 0.6)
            self._roi_actors[self.selected_roi].GetProperty().SetOpacity(0.7)

        # Reset all ROIs
        for act in self._roi_actors:
            act.GetProperty().SetColor(0.6, 0.6, 0.6)
            act.GetProperty().SetOpacity(0.7)

        self.selected_roi = idx
        if idx < len(self._roi_actors):
            self._roi_actors[idx].GetProperty().SetColor(1.0, 1.0, 0.0)
            self._roi_actors[idx].GetProperty().SetOpacity(1.0)

        self._show_connections(idx, depth=self.propagation_depth,
                               top_k=self.propagation_top_k)
        print(f"[select] ROI {idx}: {self.roi_names[idx]} "
              f"(depth={self.propagation_depth}, top_k={self.propagation_top_k})")

    def run(self):
        """Start the interactive visualizer."""
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(self.win)
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)

        def on_click(obj, ev):
            x, y = obj.GetEventPosition()
            if self.picker.Pick(x, y, 0, self.ren) <= 0:
                return
            # Find nearest ROI
            px, py, pz = self.picker.GetPickPosition()
            pos = np.array([px, py, pz], np.float32)
            dists = np.linalg.norm(self.centroids - pos, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < 10.0:  # threshold
                self._anim_playing = False
                self._summary_actor.VisibilityOff()
                self._story_actor.VisibilityOff()
                self._clear_state_labels()
                self._select_roi(nearest)

        def on_key(obj, ev):
            key = obj.GetKeySym()
            key_lower = key.lower() if key else ""
            shift = bool(obj.GetShiftKey())

            if key_lower == "escape":
                obj.TerminateApp()

            elif key_lower == "p" and not shift:
                # Perturb selected ROI (synchronous — VTK freezes during GPT)
                self._anim_playing = False
                if self.selected_roi is None:
                    print("[perturb] Select an ROI first (click).")
                    return
                name = self.roi_names[self.selected_roi]
                current_state = self.brain_state_db.get(name)

                self._sel_actor.SetInput(f"Generating perturbation options\nfor {name}...\n(window will freeze briefly)")
                self.win.Render()

                print(f"\n[perturb] Region: {name}")
                if current_state:
                    print(f"  Current state: {current_state}")
                print(f"  Fetching perturbation options from GPT...")
                sys.stdout.flush()

                try:
                    proposals = self.brain_state_db.propose_perturbations(name)
                except Exception as e:
                    print(f"[perturb] GPT proposal failed: {e}")
                    proposals = [
                        "Heightened activity in this region",
                        "Suppressed activity in this region",
                        "Shift to an alternative processing mode",
                        "Disrupted connectivity with downstream regions",
                    ]

                # Show options in VTK and console
                lines = [f"Perturbation options for:", name[:60], ""]
                for i, p in enumerate(proposals, 1):
                    lines.append(f"  {i}. {p[:55]}")
                lines.append(f"  5. (Custom — type in console)")
                lines.append("")
                lines.append("Press 1-5 in VTK window")
                self._sel_actor.SetInput("\n".join(lines))
                self.win.Render()

                print(f"\n  Perturbation options for {name}:")
                for i, p in enumerate(proposals, 1):
                    print(f"    {i}. {p}")
                print(f"    5. (Write your own)")
                print(f"  >>> Press 1-5 in the VTK window <<<\n")
                sys.stdout.flush()

                self._perturb_proposals = proposals
                self._perturb_region = name
                self._perturb_waiting = True

            elif key_lower in ("1", "2", "3", "4") and self._perturb_waiting:
                # User picked a perturbation option (synchronous)
                idx = int(key_lower) - 1
                desc = self._perturb_proposals[idx]
                self._perturb_waiting = False
                self._perturb_proposals = None
                name = self._perturb_region
                print(f"[perturb] Applying option {key_lower}: {desc}")
                self._sel_actor.SetInput(f"Applying perturbation\nto {name}...\n(window will freeze briefly)")
                self.win.Render()

                try:
                    new_state = self.brain_state_db.alter_region_state(
                        name, desc, skip_validation=True)
                    print(f"[perturb] {name} -> {new_state}")
                    self._sel_actor.SetInput(f"Perturbed: {name}\n{new_state[:80]}")
                except Exception as e:
                    print(f"[perturb] Error: {e}")
                    self._sel_actor.SetInput(f"Perturbation failed: {e}")
                self.win.Render()

            elif key_lower == "5" and self._perturb_waiting:
                # Custom perturbation — synchronous console input
                self._perturb_waiting = False
                self._perturb_proposals = None
                name = self._perturb_region
                self._sel_actor.SetInput(f"Type perturbation in console\nthen press Enter")
                self.win.Render()

                print(f"  >>> Type your custom perturbation in the console <<<")
                sys.stdout.flush()
                desc = input("  Enter your perturbation: ").strip()
                if not desc:
                    print("[perturb] Cancelled.")
                    return

                print(f"[perturb] Applying: {desc}")
                self._sel_actor.SetInput(f"Applying perturbation\nto {name}...")
                self.win.Render()

                try:
                    new_state = self.brain_state_db.alter_region_state(
                        name, desc, skip_validation=True)
                    print(f"[perturb] {name} -> {new_state}")
                    self._sel_actor.SetInput(f"Perturbed: {name}\n{new_state[:80]}")
                except Exception as e:
                    print(f"[perturb] Error: {e}")
                    self._sel_actor.SetInput(f"Perturbation failed: {e}")
                self.win.Render()

            elif key_lower == "p" and shift:
                # Propagate through network (synchronous)
                if self.selected_roi is None:
                    print("[propagate] Select an ROI first.")
                    return
                if not self.brain_state_db.has_states():
                    print("[propagate] Initialize states first (s).")
                    return

                source_name = self.roi_names[self.selected_roi]
                source_idx = self.selected_roi
                connections = get_strongest_connections(
                    self.A, source_idx,
                    top_k=self.propagation_top_k,
                    depth=self.propagation_depth
                )

                self._sel_actor.SetInput(f"Propagating from {source_name}...\n(window will freeze briefly)")
                self._clear_state_labels()
                self._summary_actor.VisibilityOff()
                self._story_actor.VisibilityOff()
                self.win.Render()

                print(f"[propagate] From {source_name} to "
                      f"{len(connections)} regions (graph-based)...")
                sys.stdout.flush()

                # Build name->index map and target->source map for visualization
                name_to_idx = {n: i for i, n in enumerate(self.roi_names)}
                # Map target_name -> source_idx from the connections list
                target_source_map = {}
                for c in connections:
                    tname = self.roi_names[c["target_idx"]]
                    target_source_map[tname] = c["source_idx"]

                # Collect propagation steps for looped animation
                self._anim_steps = []  # list of (from_idx, to_idx, state_text, depth)

                def _render_update(msg):
                    """Called per-region during propagation."""
                    print(f"  {msg}")
                    sys.stdout.flush()
                    if ": " in msg:
                        parts = msg.strip().split(": ", 1)
                        target_name = parts[0].strip()
                        new_state = parts[1].strip()
                        ti = name_to_idx.get(target_name)
                        if ti is not None:
                            # Draw line from the actual source (not always the original)
                            from_idx = target_source_map.get(target_name, source_idx)
                            # Find depth for color
                            depth = 1
                            for c in connections:
                                if c["target_idx"] == ti:
                                    depth = c.get("depth", 1)
                                    break
                            color = (1.0, 0.3, 0.3) if depth == 1 else (1.0, 0.6, 0.2)
                            self._add_propagation_line(from_idx, ti, color=color)
                            self._add_state_label(ti, new_state, color=color)
                            self.win.Render()
                            # Store for animation loop
                            self._anim_steps.append((from_idx, ti, new_state, depth))

                try:
                    for c in connections:
                        c["target_name"] = self.roi_names[c["target_idx"]]

                    updates = self.brain_state_db.propagate_through_graph(
                        source_name, connections,
                        A=self.A, roi_names=self.roi_names,
                        callback=_render_update
                    )
                    if updates:
                        summary = self.brain_state_db.summarize_changes(
                            updates, source_name
                        )
                        story = self.brain_state_db.generate_flow_story(
                            updates, source_name, connections=connections
                        )
                        print(f"\n--- PROPAGATION SUMMARY ---")
                        print(summary)
                        print(f"\n--- INFORMATION FLOW STORY ---")
                        print(story)
                        print("--- END ---\n")

                        # Show summary in top-left
                        self._summary_actor.SetInput(
                            self._word_wrap(f"PROPAGATION SUMMARY\n{summary}", 45, 20))
                        self._summary_actor.VisibilityOn()

                        # Show story in top-right
                        self._story_actor.SetInput(
                            self._word_wrap(f"INFORMATION FLOW\n{story}", 45, 20))
                        self._story_actor.VisibilityOn()

                        # Clear yellow status text
                        self._sel_actor.SetInput("")

                        # Start looped animation
                        self._anim_source_idx = source_idx
                        self._anim_frame = 0
                        self._anim_playing = True
                    else:
                        print("[propagate] No changes.")
                except Exception as e:
                    print(f"[propagate] Error: {e}")
                    import traceback
                    traceback.print_exc()
                self.win.Render()

            elif key_lower == "s":
                # Initialize states (synchronous)
                region_names = self.roi_names
                self._sel_actor.SetInput("Enter brain state in console...")
                self.win.Render()
                print("[states] Enter global brain state (type in console):")
                sys.stdout.flush()

                gs = input("  > ").strip()
                self._sel_actor.SetInput(f"Initializing states...\n(window will freeze)")
                self.win.Render()

                try:
                    result = self.brain_state_db.initialize_from_global(
                        gs, region_names,
                        callback=lambda m: print(f"  {m}")
                    )
                    print(f"[states] {result}")
                    self._sel_actor.SetInput("States initialized.")
                except Exception as e:
                    print(f"[states] Error: {e}")
                    self._sel_actor.SetInput(f"State init failed: {e}")
                self.win.Render()

            elif key_lower == "r":
                self.brain_state_db.clear()
                print("[states] Cleared all states.")

            elif key_lower in ("plus", "equal"):
                self.connection_threshold = min(0.5, self.connection_threshold * 1.5)
                print(f"[threshold] {self.connection_threshold:.3f}")
                if self.selected_roi is not None:
                    self._show_connections(self.selected_roi,
                                           depth=self.propagation_depth,
                                           top_k=self.propagation_top_k)

            elif key_lower in ("minus", "underscore"):
                self.connection_threshold = max(0.001, self.connection_threshold / 1.5)
                print(f"[threshold] {self.connection_threshold:.3f}")
                if self.selected_roi is not None:
                    self._show_connections(self.selected_roi,
                                           depth=self.propagation_depth,
                                           top_k=self.propagation_top_k)

            elif key_lower == "bracketright":
                # ] = increase propagation depth
                self.propagation_depth = min(6, self.propagation_depth + 1)
                print(f"[depth] propagation depth = {self.propagation_depth}")
                self._sel_actor.SetInput(
                    f"Propagation depth: {self.propagation_depth}\n"
                    f"(top_k={self.propagation_top_k} per level)")
                if self.selected_roi is not None:
                    self._show_connections(self.selected_roi,
                                           depth=self.propagation_depth,
                                           top_k=self.propagation_top_k)
                self.win.Render()

            elif key_lower == "bracketleft":
                # [ = decrease propagation depth
                self.propagation_depth = max(1, self.propagation_depth - 1)
                print(f"[depth] propagation depth = {self.propagation_depth}")
                self._sel_actor.SetInput(
                    f"Propagation depth: {self.propagation_depth}\n"
                    f"(top_k={self.propagation_top_k} per level)")
                if self.selected_roi is not None:
                    self._show_connections(self.selected_roi,
                                           depth=self.propagation_depth,
                                           top_k=self.propagation_top_k)
                self.win.Render()

            elif key_lower == "period":
                # . = increase top_k (more connections per level)
                self.propagation_top_k = min(50, self.propagation_top_k + 5)
                print(f"[top_k] connections per level = {self.propagation_top_k}")
                if self.selected_roi is not None:
                    self._show_connections(self.selected_roi,
                                           depth=self.propagation_depth,
                                           top_k=self.propagation_top_k)
                self.win.Render()

            elif key_lower == "comma":
                # , = decrease top_k
                self.propagation_top_k = max(3, self.propagation_top_k - 5)
                print(f"[top_k] connections per level = {self.propagation_top_k}")
                if self.selected_roi is not None:
                    self._show_connections(self.selected_roi,
                                           depth=self.propagation_depth,
                                           top_k=self.propagation_top_k)
                self.win.Render()

        def on_anim_timer(_o, _e):
            """Looped propagation animation — show signal traveling step by step."""
            if not self._anim_playing or not self._anim_steps:
                return

            frame = self._anim_frame
            steps = self._anim_steps
            total_frames = len(steps) + 6  # steps + pause frames at end

            if frame == 0:
                # Clear previous animation visuals
                self._clear_state_labels()
                # Show source ROI highlighted
                if self._anim_source_idx < len(self._roi_actors):
                    self._roi_actors[self._anim_source_idx].GetProperty().SetColor(1.0, 1.0, 0.0)
                    self._roi_actors[self._anim_source_idx].GetProperty().SetOpacity(1.0)

            if frame < len(steps):
                from_idx, to_idx, state_text, depth = steps[frame]
                color = (1.0, 0.3, 0.3) if depth == 1 else (1.0, 0.6, 0.2)
                self._add_propagation_line(from_idx, to_idx, color=color)
                self._add_state_label(to_idx, state_text, color=color)
                # Highlight target ROI
                if to_idx < len(self._roi_actors):
                    self._roi_actors[to_idx].GetProperty().SetColor(*color)
                    self._roi_actors[to_idx].GetProperty().SetOpacity(1.0)

            self._anim_frame = frame + 1
            if self._anim_frame >= total_frames:
                # Reset ROI colors and loop
                for act in self._roi_actors:
                    act.GetProperty().SetColor(0.6, 0.6, 0.6)
                    act.GetProperty().SetOpacity(0.7)
                self._anim_frame = 0

            self.win.Render()

        iren.Initialize()
        iren.AddObserver("LeftButtonPressEvent", on_click)
        iren.AddObserver("KeyPressEvent", on_key)
        iren.AddObserver("TimerEvent", on_anim_timer)
        iren.CreateRepeatingTimer(500)  # animation tick every 500ms
        self.win.Render()

        print("\n=== rDCIM Propagation Visualizer ===")
        print(f"  ROIs: {self.R}")
        print(f"  Max |weight|: {self.max_weight:.4f}")
        print(f"  Click ROI to select | p perturb | Shift+P propagate")
        print(f"  s init states | r reset | +/- threshold")
        print(f"  [/] depth ({self.propagation_depth}) | ,/. top_k ({self.propagation_top_k}) | Esc quit")
        print()

        iren.Start()


def main():
    _ensure_ssl()
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser(description="rDCIM Propagation Visualizer")
    ap.add_argument("--rdcim", type=Path, default=DEFAULT_RDCIM,
                    help="rDCIM matrix .npy file")
    ap.add_argument("--centroids", type=Path, default=DEFAULT_CENTROIDS,
                    help="ROI centroid positions .npy file")
    ap.add_argument("--global-state", type=str, default="",
                    help="Initialize brain states with this global state")
    ap.add_argument("--perturb", type=int, default=None,
                    help="ROI index to perturb on startup")
    ap.add_argument("--perturbation", type=str, default="",
                    help="Perturbation description")
    ap.add_argument("--hq", action="store_true",
                    help="Use high-quality GPT model (gpt-5.4) instead of gpt-5.4-mini")
    ap.add_argument("--debug", action="store_true",
                    help="Print full LLM prompts to console before each call")
    ap.add_argument("--extra-parcellation", type=Path, default=None,
                    help="Path to NIfTI parcellation file for finer subregion labels")
    ap.add_argument("--extra-parcellation-labels", type=Path, default=None,
                    help="JSON label map for extra parcellation")
    args = ap.parse_args()

    model = "gpt-5.4" if args.hq else "gpt-5.4-mini"
    print(f"[config] LLM model: {model}")

    A = load_rdcim(args.rdcim)
    centroids = load_centroids(args.centroids)

    if A.shape[0] != centroids.shape[0]:
        print(f"[warning] matrix ({A.shape[0]}) != centroids ({centroids.shape[0]})")
        n = min(A.shape[0], centroids.shape[0])
        A = A[:n, :n]
        centroids = centroids[:n]

    roi_names = get_roi_names(A.shape[0])

    # Map ROI names to actual Allen atlas regions
    print("[init] Mapping ROIs to anatomical regions...")
    roi_names = map_rois_to_regions(centroids, roi_names)

    brain_db = BrainStateDB(DEFAULT_STATE_FILE, model=model, debug=args.debug)

    if args.global_state:
        print(f"[init] Initializing brain states: {args.global_state}")
        brain_db.initialize_from_global(args.global_state, roi_names)

    viz = RDCIMVisualizer(A, centroids, roi_names, brain_db)

    if args.perturb is not None and args.perturbation:
        if 0 <= args.perturb < len(roi_names):
            brain_db.alter_region_state(roi_names[args.perturb], args.perturbation)

    viz.run()


if __name__ == "__main__":
    main()
