#!/usr/bin/env python
"""ROI Flow Mode — manifold flow visualization with ROI activation panel.

Single-window visualization with two viewports:
  Left:  MDN particle flow through a learned neural manifold
  Right: ROI activation spheres showing corresponding brain region activity

Place a probe in the manifold (follows flow or manual path), freeze it,
and ask the LLM to interpret what the resulting ROI contribution shift means.
After LLM analysis, animated particles flow between donor/receiver ROIs.

Usage:
    python examples/roi_flow_mode.py [--hq] [--debug]

Controls:
    G then click — place probe in manifold (follows flow)
    M then click — manual path mode (click to extend path)
    Shift+G     — freeze probe, compute ROI delta, ask LLM
    V           — toggle ROI anim mode (path ↔ particles)
    1 / 2       — save current path as Path A / B
    D           — compare Path A vs Path B (cyan=A, magenta=B)
    C           — clear all
    +/-         — speed scale
    Q/Esc       — quit

Data:
    Run `python scripts/download_roi_flow_data.py` first to download all
    required data from HuggingFace.
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersGeneral import vtkSplineFilter
from vtkmodules.vtkFiltersCore import vtkTubeFilter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.field_loader import load_field, TriLinearSampler
from src.colormaps import turbo_rgb01
from src.roi_flow import ManifoldToROIKNN, ROIFlowAnalyzer, ROIFlowLLM
# Reuse visualization utilities from src/main.py
from src.main import build_cloud, add_window_legend, VtkTextOverlay, wrap_inside

# Default paths — all ROI flow data lives in data/roi_flow/
DATA_DIR = PROJECT_ROOT / "data"
ROI_FLOW_DIR = DATA_DIR / "roi_flow"

DEFAULT_META = ROI_FLOW_DIR / "mdn_universal_raw_grid64_meta.json"
DEFAULT_OOS = ROI_FLOW_DIR / "universal_soul_2sdm_rest_points.ply"
DEFAULT_PROBE_EMBED = ROI_FLOW_DIR / "probe_embed.npy"
DEFAULT_PROBE_ROI = ROI_FLOW_DIR / "probe_roi.npy"
DEFAULT_PROBE_ROI_CENTERS = ROI_FLOW_DIR / "probe_roi_centers.npy"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_points_any(path: Path) -> np.ndarray:
    """Load points from .npy, .ply, .obj, .stl files using VTK."""
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    readers = {
        ".ply": vtk.vtkPLYReader, ".obj": vtk.vtkOBJReader,
        ".stl": vtk.vtkSTLReader, ".vtk": vtk.vtkPolyDataReader,
    }
    reader_cls = readers.get(path.suffix)
    if reader_cls is None:
        raise ValueError(f"Unsupported format: {path.suffix}")
    reader = reader_cls()
    reader.SetFileName(str(path))
    reader.Update()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float32)


def _map_roi_to_brain_regions(roi_centers: np.ndarray) -> list[str]:
    """Map ROI centroids to Allen brain regions."""
    try:
        from src.mesh_overlay import FlowMeshOverlay
        alignment_file = DATA_DIR / "brain_alignment.json"
        mesh_dir = DATA_DIR / "meshes"
        if not alignment_file.exists() or not mesh_dir.exists():
            raise FileNotFoundError("Brain meshes not available")

        ren = vtk.vtkRenderer()
        win = vtk.vtkRenderWindow()
        win.SetOffScreenRendering(1)
        win.AddRenderer(ren)

        overlay = FlowMeshOverlay(ren=ren, win=win, mesh_dir=mesh_dir,
                                   alignment_file=alignment_file)
        grid_cache = DATA_DIR / "label_grid_cache.npz"
        if not overlay.load_label_grid(grid_cache):
            overlay.build_label_grid()
            overlay.save_label_grid(grid_cache)

        names = []
        for i, pos in enumerate(roi_centers):
            key = overlay.get_region_at_point(pos)
            if key is None:
                key = overlay.find_nearest_region(pos, search_radius=6)
            if key is not None:
                region_name = overlay.get_region_name(key)
                hemi = "left" if pos[0] < 0 else "right"
                ap = "anterior" if pos[1] > 0 else "posterior"
                names.append(f"ROI_{i} ({region_name}, {hemi}, {ap})")
            else:
                hemi = "left" if pos[0] < 0 else "right"
                names.append(f"ROI_{i} ({hemi})")
        mapped = sum(1 for n in names if ", " in n)
        print(f"[roi-map] Mapped {mapped}/{len(names)} ROIs to brain regions")
        return names
    except Exception as e:
        print(f"[roi-map] Could not map ROIs to brain regions: {e}")
        names = []
        for i, pos in enumerate(roi_centers):
            hemi = "left" if pos[0] < 0 else "right"
            ap = "anterior" if pos[1] > 0 else "posterior"
            si = "superior" if pos[2] > 0 else "inferior"
            names.append(f"ROI_{i} ({hemi}, {ap}, {si})")
        return names


def _build_spline_trail(pts_array: np.ndarray, diag: float):
    """Build a smooth yellow spline+tube actor from a sequence of points."""
    if pts_array is None or len(pts_array) < 2:
        return None
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(pts_array.astype(np.float32), deep=True))
    pl = vtk.vtkPolyLine()
    pl.GetPointIds().SetNumberOfIds(len(pts_array))
    for i in range(len(pts_array)):
        pl.GetPointIds().SetId(i, i)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(pl)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vpts)
    poly.SetLines(lines)

    spl = vtkSplineFilter()
    spl.SetInputData(poly)
    spl.SetSubdivideToLength()
    spl.SetLength(max(diag * 0.01, 1e-6))
    spl.Update()

    tube = vtkTubeFilter()
    tube.SetInputConnection(spl.GetOutputPort())
    tube.SetNumberOfSides(12)
    tube.SetRadius(diag * 0.004)
    tube.CappingOn()
    tube.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 0.0)
    actor.GetProperty().SetOpacity(0.85)
    actor.GetProperty().LightingOff()
    return actor


# ---------------------------------------------------------------------------
# ROI Panel (right side of split window)
# ---------------------------------------------------------------------------

class ROIPanel:
    """Renders ROI spheres colored/sized by activation (white-center bicolor)."""

    # Original bicolor scheme: white center, orange positive, blue negative
    POS_COLOR = np.array([255, 120, 0], np.float64)   # orange
    NEG_COLOR = np.array([0, 100, 255], np.float64)    # blue
    WHITE = np.array([255, 255, 255], np.float64)
    GRAY = np.array([200, 200, 200], np.float64) / 255.0

    def __init__(self, ren: vtk.vtkRenderer, centers: np.ndarray,
                 names: list[str]):
        self.ren = ren
        self.centers = centers.astype(np.float64)
        self.names = names
        self.n_rois = len(names)
        self.base_radius = 0.6
        self.radius_scale = 1.0
        self._spheres: list[vtk.vtkActor] = []
        self._setup()

    def _setup(self):
        for i in range(self.n_rois):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*self.centers[i])
            sphere.SetRadius(self.base_radius)
            sphere.SetPhiResolution(16)
            sphere.SetThetaResolution(16)
            sphere.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*self.GRAY)
            actor.GetProperty().SetOpacity(0.7)
            self.ren.AddActor(actor)
            self._spheres.append(actor)

    @staticmethod
    def _bicolor(value: float, abs_max: float):
        """White-center bicolor: white->orange (positive), white->blue (negative)."""
        if abs_max < 1e-12:
            return ROIPanel.GRAY
        t = np.clip(abs(value) / abs_max, 0.0, 1.0)
        if value >= 0:
            rgb = ROIPanel.WHITE * (1.0 - t) + ROIPanel.POS_COLOR * t
        else:
            rgb = ROIPanel.WHITE * (1.0 - t) + ROIPanel.NEG_COLOR * t
        return rgb / 255.0

    def update_values(self, values: np.ndarray):
        if values is None or len(values) != self.n_rois:
            return
        av = np.abs(values)
        abs_max = float(np.percentile(av, 97)) + 1e-8
        for i in range(self.n_rois):
            v = float(values[i])
            color = self._bicolor(v, abs_max)
            radius = self.base_radius + self.radius_scale * min(abs(v) / abs_max, 1.0)
            self._spheres[i].GetProperty().SetColor(*color)
            mapper = self._spheres[i].GetMapper()
            src = vtk.vtkSphereSource()
            src.SetCenter(*self.centers[i])
            src.SetRadius(radius)
            src.SetPhiResolution(16)
            src.SetThetaResolution(16)
            src.Update()
            mapper.SetInputData(src.GetOutput())
            mapper.Update()

    def dim_spheres(self, opacity: float = 0.15):
        for s in self._spheres:
            s.GetProperty().SetOpacity(opacity)

    def restore_spheres(self, opacity: float = 0.7):
        for s in self._spheres:
            s.GetProperty().SetOpacity(opacity)

    def reset_colors(self):
        for i in range(self.n_rois):
            self._spheres[i].GetProperty().SetColor(*self.GRAY)
            self._spheres[i].GetProperty().SetOpacity(0.7)


# ---------------------------------------------------------------------------
# ROI Flow Dots — animated particles flowing between donor/receiver ROIs
# ---------------------------------------------------------------------------

class ROIFlowDots:
    """Particles flowing from donor (negative delta) to receiver (positive delta) ROIs.

    Matches the original simulate_mdn_flow.py behavior: 120k particles,
    turbo colormap, endpoint fade, continuous emission from donor→receiver pairs.
    """

    def __init__(self, ren: vtk.vtkRenderer, roi_centers: np.ndarray,
                 max_particles: int = 120000):
        self.ren = ren
        self.C = roi_centers.astype(np.float32)
        self.Nmax = max_particles
        self.rng = np.random.default_rng(42)

        self.pos = np.zeros((self.Nmax, 3), np.float32)
        self.p0 = np.zeros((self.Nmax, 3), np.float32)
        self.dest = np.zeros((self.Nmax, 3), np.float32)
        self.alive = np.zeros(self.Nmax, bool)
        self.age = np.zeros(self.Nmax, np.float32)
        self.life = np.ones(self.Nmax, np.float32)
        self.speed = np.zeros(self.Nmax, np.float32)  # for colormap

        self.emitter_on = False
        self._neg_idx = np.array([], np.int32)
        self._pos_idx = np.array([], np.int32)
        self._pairs_probs = None
        self._emit_rate = 2000.0
        self._emit_accum = 0.0
        self._dt = 1.0 / 60.0
        # Endpoint fade parameters (matching original)
        self._fade_start = 0.08   # fade in over first 8%
        self._fade_end = 0.08     # fade out over last 8%
        # Burst mode (on by default — short sharp bursts with visible gap)
        self.burst_mode = True
        self.burst_period = 0.6       # seconds per cycle (fast)
        self.burst_emit_frac = 0.25   # emit during first 25% — short burst, long gap
        self._burst_timer = 0.0
        self.capture_accel = 2.5      # fast capture so particles clear out during gap
        # Mid-segment fade: make particles transparent in the middle of their journey
        # so you can clearly see source and destination endpoints
        self._mid_fade = True

        # Use build_cloud from src/main.py
        init_rgba = np.zeros((self.Nmax, 4), np.uint8)
        self._pts, self._colors, self._pd, self.actor = build_cloud(
            np.zeros((self.Nmax, 3), np.float32), rgba=init_rgba)
        self.actor.GetProperty().SetPointSize(1.2)
        self.actor.VisibilityOff()
        self.ren.AddActor(self.actor)

    def start_from_delta(self, delta: np.ndarray, top_frac: float = 0.15,
                         emit_rate: float = 2000.0):
        v = delta.astype(np.float64)
        k = max(1, int(len(v) * top_frac))
        idx_strong = np.argsort(-np.abs(v))[:k]
        v_str = v[idx_strong]
        eps = 1e-8
        self._pos_idx = idx_strong[v_str > eps]
        self._neg_idx = idx_strong[v_str < -eps]

        if len(self._pos_idx) == 0 or len(self._neg_idx) == 0:
            return

        p_don = np.abs(v[self._neg_idx])
        p_don /= p_don.sum()
        p_recv = np.abs(v[self._pos_idx])
        p_recv /= p_recv.sum()
        pairs = np.outer(p_don, p_recv).ravel()
        pairs /= pairs.sum()
        self._pairs_probs = pairs

        # Scale emit rate by delta magnitude (like original)
        l1 = float(np.sum(np.abs(v)))
        self._emit_rate = max(100.0, min(emit_rate * max(l1, 0.1), 8000.0))

        self.alive[:] = False
        self._emit_accum = 0.0
        self.emitter_on = True
        self.actor.VisibilityOn()
        print(f"[roi-flow] Started: {len(self._neg_idx)} donors -> "
              f"{len(self._pos_idx)} receivers, rate={self._emit_rate:.0f}/s")

    def tick(self):
        if not self.emitter_on:
            return

        # Burst phase: emit only during first fraction of each cycle
        if self.burst_mode:
            self._burst_timer = (self._burst_timer + self._dt) % max(self.burst_period, 1e-6)
            emit_phase = (self._burst_timer / max(self.burst_period, 1e-6)) < self.burst_emit_frac
        else:
            emit_phase = True

        # Advance alive particles
        idx = np.nonzero(self.alive)[0]
        if len(idx) > 0:
            # Accelerate during capture phase of burst
            speed_mul = 1.0
            if self.burst_mode and not emit_phase:
                speed_mul = self.capture_accel
            self.age[idx] += self._dt * speed_mul
            t = np.clip(self.age[idx] / self.life[idx], 0.0, 1.0)
            te = t * t * (3.0 - 2.0 * t)  # smoothstep easing
            seg = self.dest[idx] - self.p0[idx]
            self.pos[idx] = self.p0[idx] + seg * te[:, None]
            self.alive[idx[t >= 1.0]] = False

        # Spawn new particles (only during emit phase)
        Nd, Nr = len(self._neg_idx), len(self._pos_idx)
        if Nd > 0 and Nr > 0:
            self._emit_accum += self._emit_rate * self._dt
            n_new = int(self._emit_accum)
            if emit_phase and n_new > 0:
                self._emit_accum -= n_new
                free = np.nonzero(~self.alive)[0]
                use = free[:n_new]
                if len(use) > 0:
                    choice = self.rng.choice(Nd * Nr, size=len(use),
                                             replace=True, p=self._pairs_probs)
                    src_roi = self._neg_idx[choice // Nr]
                    dst_roi = self._pos_idx[choice % Nr]
                    self.p0[use] = self.C[src_roi]
                    self.dest[use] = self.C[dst_roi]
                    self.life[use] = self.rng.uniform(0.8, 1.8, size=len(use)).astype(np.float32)
                    self.age[use] = 0.0
                    self.alive[use] = True
                    seg_len = np.linalg.norm(self.dest[use] - self.p0[use], axis=1)
                    self.speed[use] = seg_len / (self.life[use] + 1e-8)
            elif not emit_phase:
                # Still accumulate but don't spend — creates burst on next emit phase
                pass

        self._update_display()

    def _update_display(self):
        idx = np.nonzero(self.alive)[0]
        P = np.zeros((self.Nmax, 3), np.float32)
        rgba = np.zeros((self.Nmax, 4), np.uint8)

        if len(idx) > 0:
            P[idx] = self.pos[idx]
            # Color from precomputed speed via turbo colormap
            spd = self.speed[idx]
            s_max = float(np.percentile(spd, 97)) if len(spd) > 0 else 1.0
            t = np.clip(spd / max(s_max, 1e-8), 0.0, 1.0)
            rgb = turbo_rgb01(t)

            # Endpoint fade + mid-segment dip
            # Particles are opaque near source and destination but fade in the
            # middle of their journey so you can clearly see where they come
            # from and where they go.
            prog = np.clip(self.age[idx] / self.life[idx], 0.0, 1.0)
            fs, fe = self._fade_start, self._fade_end

            # Base endpoint fade
            alpha = np.where(prog < fs, prog / max(fs, 1e-6),
                             np.where(prog > (1.0 - fe),
                                      (1.0 - prog) / max(fe, 1e-6), 1.0))

            # Mid-segment transparency dip (U-shaped alpha along journey)
            if getattr(self, '_mid_fade', False):
                # mid_alpha: 1.0 at endpoints, dips to 0.3 at center
                mid = 0.5
                spread = 0.35  # width of the dip
                dist_from_mid = np.abs(prog - mid) / spread
                mid_alpha = np.clip(0.3 + 0.7 * dist_from_mid, 0.3, 1.0)
                alpha = alpha * mid_alpha

            alpha = (np.clip(alpha, 0.0, 1.0) * 220).astype(np.uint8)

            rgba[idx, :3] = rgb
            rgba[idx, 3] = alpha

        self._pts.SetData(numpy_to_vtk(P, deep=True))
        self._colors.DeepCopy(numpy_to_vtk(rgba, deep=True))
        self._colors.Modified()
        self._pd.Modified()

    def stop(self):
        self.emitter_on = False
        self.alive[:] = False
        self.actor.VisibilityOff()

    def is_active(self):
        return self.emitter_on


# ---------------------------------------------------------------------------
# ROI Path Animation — animate ROI activation along the probe path
# ---------------------------------------------------------------------------

class ROIPathAnimation:
    """Snap between START and END ROI activation states — no intermediates.

    Fast toggle between the brain state at the beginning vs end of the path.
    200 most dynamic ROIs are animated; the rest stay dim gray.
    Active ROIs are vivid turbo-colored and blow up to 3.5× base radius.
    Quiet ROIs shrink to near-invisible gray dots.
    """

    N_TOP = 200             # number of ROIs to animate
    TOGGLE_TICKS = 40       # ticks to hold each state before flipping
    RADIUS_MIN_FRAC = 0.3   # quiet ROI radius multiplier
    RADIUS_MAX_FRAC = 3.5   # loudest ROI radius multiplier

    def __init__(self, roi_panel: ROIPanel, knn: 'ManifoldToROIKNN'):
        self.panel = roi_panel
        self.knn = knn
        self.active = False
        self._start_vals: np.ndarray | None = None   # (n_rois,)
        self._end_vals: np.ndarray | None = None     # (n_rois,)
        self._top_mask: np.ndarray | None = None     # (n_rois,) bool
        self._showing_end = False
        self._tick_counter = 0
        self._abs_max_start = 1.0
        self._abs_max_end = 1.0

    # ----- resampling helper (also used by ROIPathCompare) -----
    @staticmethod
    def _resample_path(pts: np.ndarray, n: int) -> np.ndarray:
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = cum[-1]
        if total_len < 1e-9:
            return pts[:1]
        sample_dists = np.linspace(0, total_len, n)
        out = np.zeros((n, 3), np.float32)
        for i, sd in enumerate(sample_dists):
            idx = np.clip(np.searchsorted(cum, sd, side='right') - 1,
                          0, len(pts) - 2)
            seg_len = cum[idx + 1] - cum[idx]
            t = (sd - cum[idx]) / max(seg_len, 1e-9)
            out[i] = pts[idx] * (1 - t) + pts[idx + 1] * t
        return out

    def build_from_path(self, path_points: list[np.ndarray],
                        n_samples: int = 60):
        """Query ROI values at path start and end."""
        if len(path_points) < 2:
            return

        pts = np.array(path_points, dtype=np.float32)
        start_pt = pts[0]
        end_pt = pts[-1]

        start_vals = self.knn.query(start_pt)
        end_vals = self.knn.query(end_pt)

        # Select top N_TOP ROIs by absolute difference start→end
        diff = np.abs(end_vals - start_vals)
        n_top = min(self.N_TOP, self.panel.n_rois)
        top_indices = np.argsort(-diff)[:n_top]
        mask = np.zeros(self.panel.n_rois, bool)
        mask[top_indices] = True

        self._start_vals = start_vals
        self._end_vals = end_vals
        self._top_mask = mask
        self._abs_max_start = float(np.percentile(np.abs(start_vals[mask]), 97)) + 1e-8
        self._abs_max_end = float(np.percentile(np.abs(end_vals[mask]), 97)) + 1e-8
        self._showing_end = False
        self._tick_counter = 0
        self.active = True

        # Dim non-selected ROIs immediately
        for i in range(self.panel.n_rois):
            if not mask[i]:
                self.panel._spheres[i].GetProperty().SetColor(0.3, 0.3, 0.32)
                self.panel._spheres[i].GetProperty().SetOpacity(0.06)
                mapper = self.panel._spheres[i].GetMapper()
                src = vtk.vtkSphereSource()
                src.SetCenter(*self.panel.centers[i])
                src.SetRadius(self.panel.base_radius * 0.25)
                src.SetPhiResolution(6)
                src.SetThetaResolution(6)
                src.Update()
                mapper.SetInputData(src.GetOutput())
                mapper.Update()

        # Apply start state immediately
        self._apply_state(False)

        print(f"[roi-anim] Start/end toggle: {n_top} active ROIs, "
              f"flip every {self.TOGGLE_TICKS} ticks")

    def _apply_state(self, show_end: bool):
        """Snap all selected ROIs to start or end state."""
        vals = self._end_vals if show_end else self._start_vals
        abs_max = self._abs_max_end if show_end else self._abs_max_start

        for i in range(self.panel.n_rois):
            if not self._top_mask[i]:
                continue

            v = float(vals[i])
            intensity = min(abs(v) / abs_max, 1.0)

            # Pure turbo color — no gray blending for strong signals
            rgb_t = turbo_rgb01(np.array([intensity]))[0]
            tr, tg, tb = rgb_t[0] / 255.0, rgb_t[1] / 255.0, rgb_t[2] / 255.0

            # Blend: very low intensity stays slightly gray, rest is full turbo
            if intensity < 0.08:
                r, g, b = 0.45, 0.45, 0.47
            else:
                sat = min(intensity * 1.5, 1.0)
                r = 0.45 * (1 - sat) + tr * sat
                g = 0.45 * (1 - sat) + tg * sat
                b = 0.47 * (1 - sat) + tb * sat

            # Radius: massive difference between quiet and active
            radius = self.panel.base_radius * (
                self.RADIUS_MIN_FRAC
                + (self.RADIUS_MAX_FRAC - self.RADIUS_MIN_FRAC) * intensity
            )

            # Opacity
            opacity = 0.12 + 0.88 * intensity

            self.panel._spheres[i].GetProperty().SetColor(r, g, b)
            self.panel._spheres[i].GetProperty().SetOpacity(opacity)

            mapper = self.panel._spheres[i].GetMapper()
            src = vtk.vtkSphereSource()
            src.SetCenter(*self.panel.centers[i])
            src.SetRadius(radius)
            src.SetPhiResolution(16)
            src.SetThetaResolution(16)
            src.Update()
            mapper.SetInputData(src.GetOutput())
            mapper.Update()

        self._showing_end = show_end

    def tick(self):
        """Toggle between start and end states on a timer."""
        if not self.active:
            return

        self._tick_counter += 1
        if self._tick_counter >= self.TOGGLE_TICKS:
            self._tick_counter = 0
            self._apply_state(not self._showing_end)

    def stop(self):
        self.active = False
        self._start_vals = None
        self._end_vals = None
        self.panel.restore_spheres()
        self.panel.reset_colors()

    def is_active(self):
        return self.active


# ---------------------------------------------------------------------------
# ROI Path Comparison — animate difference between two paths
# ---------------------------------------------------------------------------

class ROIPathCompare:
    """Animate the difference between two saved paths.

    Shows how Path A vs Path B differ in their ROI activation trajectories.
    Color encodes which path activates each ROI more:
      - Cyan  = Path A dominant
      - Magenta = Path B dominant
      - Gray  = similar
    Sphere size pulses with magnitude of the difference.
    """

    N_TOP = 100
    RADIUS_MIN_FRAC = 0.6
    RADIUS_MAX_FRAC = 2.5

    def __init__(self, roi_panel: ROIPanel, knn: 'ManifoldToROIKNN'):
        self.panel = roi_panel
        self.knn = knn
        self.active = False
        self._diff_frames: np.ndarray | None = None  # (n, n_rois)
        self._top_mask: np.ndarray | None = None
        self._global_max: float = 1.0
        self._n_frames = 0
        self._cursor = 0.0
        self._direction = 1
        self._tick_accum = 0.0

    def build(self, path_a: list[np.ndarray], path_b: list[np.ndarray],
              n_samples: int = 60):
        """Build comparison animation from two paths."""
        if len(path_a) < 2 or len(path_b) < 2:
            print("[compare] Both paths need at least 2 points.")
            return

        pts_a = ROIPathAnimation._resample_path(
            np.array(path_a, np.float32), n_samples)
        pts_b = ROIPathAnimation._resample_path(
            np.array(path_b, np.float32), n_samples)

        frames_a = np.zeros((n_samples, self.panel.n_rois), np.float32)
        frames_b = np.zeros((n_samples, self.panel.n_rois), np.float32)
        for i in range(n_samples):
            frames_a[i] = self.knn.query(pts_a[i])
            frames_b[i] = self.knn.query(pts_b[i])

        # Difference: positive = A stronger, negative = B stronger
        diff = frames_a - frames_b

        # Top N most different ROIs
        max_diff = np.max(np.abs(diff), axis=0)
        n_top = min(self.N_TOP, self.panel.n_rois)
        top_idx = np.argsort(-max_diff)[:n_top]
        mask = np.zeros(self.panel.n_rois, bool)
        mask[top_idx] = True

        self._diff_frames = diff
        self._top_mask = mask
        self._global_max = float(np.percentile(max_diff[mask], 97)) + 1e-8
        self._n_frames = n_samples
        self._cursor = 0.0
        self._direction = 1
        self._tick_accum = 0.0
        self.active = True

        # Dim non-selected
        for i in range(self.panel.n_rois):
            if not mask[i]:
                self.panel._spheres[i].GetProperty().SetColor(0.3, 0.3, 0.3)
                self.panel._spheres[i].GetProperty().SetOpacity(0.06)
                mapper = self.panel._spheres[i].GetMapper()
                src = vtk.vtkSphereSource()
                src.SetCenter(*self.panel.centers[i])
                src.SetRadius(self.panel.base_radius * 0.35)
                src.SetPhiResolution(8)
                src.SetThetaResolution(8)
                src.Update()
                mapper.SetInputData(src.GetOutput())

        print(f"[compare] Built comparison: {n_samples} frames, "
              f"{n_top} active ROIs")

    def tick(self):
        if not self.active or self._diff_frames is None:
            return

        self._tick_accum += 0.5
        if self._tick_accum < 1.0:
            return
        self._tick_accum -= 1.0

        self._cursor += self._direction
        if self._cursor >= self._n_frames - 1:
            self._cursor = float(self._n_frames - 1)
            self._direction = -1
        elif self._cursor <= 0:
            self._cursor = 0.0
            self._direction = 1

        idx_lo = int(self._cursor)
        idx_hi = min(idx_lo + 1, self._n_frames - 1)
        frac = self._cursor - idx_lo
        cur = (self._diff_frames[idx_lo] * (1 - frac)
               + self._diff_frames[idx_hi] * frac)

        gm = self._global_max

        for i in range(self.panel.n_rois):
            if not self._top_mask[i]:
                continue
            d = float(cur[i])
            mag = min(abs(d) / gm, 1.0)

            # Cyan (A dominant) ↔ gray ↔ Magenta (B dominant)
            if d > 0:
                # Path A stronger → cyan
                r = 0.5 * (1 - mag)
                g = 0.5 + 0.5 * mag
                b = 0.5 + 0.5 * mag
            else:
                # Path B stronger → magenta
                r = 0.5 + 0.5 * mag
                g = 0.5 * (1 - mag)
                b = 0.5 + 0.5 * mag

            radius = self.panel.base_radius * (
                self.RADIUS_MIN_FRAC
                + (self.RADIUS_MAX_FRAC - self.RADIUS_MIN_FRAC) * mag)
            opacity = 0.2 + 0.8 * mag

            self.panel._spheres[i].GetProperty().SetColor(r, g, b)
            self.panel._spheres[i].GetProperty().SetOpacity(opacity)

            mapper = self.panel._spheres[i].GetMapper()
            src = vtk.vtkSphereSource()
            src.SetCenter(*self.panel.centers[i])
            src.SetRadius(radius)
            src.SetPhiResolution(16)
            src.SetThetaResolution(16)
            src.Update()
            mapper.SetInputData(src.GetOutput())
            mapper.Update()

    def stop(self):
        self.active = False
        self._diff_frames = None
        self.panel.restore_spheres()
        self.panel.reset_colors()

    def is_active(self):
        return self.active


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ROI Flow Mode — manifold + ROI visualization")

    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--oos", type=Path, default=DEFAULT_OOS)
    ap.add_argument("--probe-embed", type=Path, default=DEFAULT_PROBE_EMBED)
    ap.add_argument("--probe-roi", type=Path, default=DEFAULT_PROBE_ROI)
    ap.add_argument("--probe-roi-centers", type=Path, default=DEFAULT_PROBE_ROI_CENTERS)
    ap.add_argument("--roi-names", type=Path, default=None,
                    help="JSON list of R ROI names (auto-mapped to brain regions if not given)")
    ap.add_argument("--probe-k", type=int, default=256)
    ap.add_argument("--probe-sigma", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--max-step-frac", type=float, default=0.01)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--respawn-jitter", type=float, default=0.015)
    ap.add_argument("--window-size", type=int, nargs=2, default=[1600, 800])
    ap.add_argument("--hq", action="store_true", help="Use gpt-5.4 instead of gpt-5.4-mini")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    model = "gpt-5.4" if args.hq else "gpt-5.4-mini"
    print(f"[config] ROI Flow Mode | LLM model: {model}")

    # Check data exists
    for name, path in [("probe-embed", args.probe_embed),
                       ("probe-roi", args.probe_roi),
                       ("probe-roi-centers", args.probe_roi_centers)]:
        if not path.exists():
            print(f"\n[error] Missing: {path}")
            print(f"  Run: python scripts/download_roi_flow_data.py")
            sys.exit(1)

    # ---------- Load MDN field ----------
    print(f"[field] loading {args.meta} ...")
    if not args.meta.exists():
        print(f"\n[error] Missing: {args.meta}")
        print(f"  Run: python scripts/download_roi_flow_data.py")
        sys.exit(1)
    fld = load_field(args.meta)
    G = fld["G"]
    amin, amax = fld["amin"], fld["amax"]
    diag = float(np.linalg.norm(amax - amin))
    sampler = TriLinearSampler(fld["mean"], amin, amax)
    V_all = fld["mean"].reshape(-1, 3)
    vmax_mean = float(np.percentile(np.linalg.norm(V_all, axis=1), 99.5))
    target_step = args.max_step_frac * max(diag, 1e-9)
    print(f"[field] grid={G}, diag={diag:.4f}, vmax={vmax_mean:.6f}")

    # ---------- Load OOS points ----------
    oos_pts = _load_points_any(args.oos)
    print(f"[oos] {oos_pts.shape[0]} points")

    # Seed particles from OOS∩TRAIN overlap (like original script)
    from scipy.spatial import cKDTree
    train_pts = fld.get("TRAIN")
    if train_pts is not None and len(train_pts) > 0:
        overlap_radius = 0.01 * max(diag, 1e-9)
        tree = cKDTree(train_pts.astype(np.float32))
        dists, _ = tree.query(oos_pts.astype(np.float32), k=1)
        cand = oos_pts[dists <= overlap_radius]
        if len(cand) > 100:
            seed_pts = cand.astype(np.float32)
            print(f"[seed] {len(seed_pts)} OOS∩TRAIN overlap points")
        else:
            in_bounds = np.all((oos_pts >= amin) & (oos_pts <= amax), axis=1)
            seed_pts = oos_pts[in_bounds] if in_bounds.sum() > 100 else oos_pts
            print(f"[seed] {len(seed_pts)} OOS points (overlap too sparse)")
    else:
        in_bounds = np.all((oos_pts >= amin) & (oos_pts <= amax), axis=1)
        seed_pts = oos_pts[in_bounds] if in_bounds.sum() > 100 else oos_pts
        print(f"[seed] {len(seed_pts)} OOS points (no TRAIN)")

    # ---------- Load probe data ----------
    X_embed = np.load(str(args.probe_embed)).astype(np.float32)
    Y_roi = np.load(str(args.probe_roi)).astype(np.float32)
    C_centers = np.load(str(args.probe_roi_centers)).astype(np.float32)
    print(f"[probe] embed: {X_embed.shape}, roi: {Y_roi.shape}, centers: {C_centers.shape}")
    assert X_embed.shape[0] == Y_roi.shape[0]
    assert C_centers.shape[0] == Y_roi.shape[1]
    n_rois = Y_roi.shape[1]

    # ROI names
    if args.roi_names and args.roi_names.exists():
        roi_names = json.loads(args.roi_names.read_text(encoding="utf-8"))
    else:
        print("[roi-map] Mapping ROI centroids to brain regions...")
        roi_names = _map_roi_to_brain_regions(C_centers)
    assert len(roi_names) == n_rois

    knn = ManifoldToROIKNN(X_embed, Y_roi, k=args.probe_k, sigma=args.probe_sigma)
    analyzer = ROIFlowAnalyzer(roi_names, C_centers)
    llm = ROIFlowLLM(model=model, debug=args.debug)
    _sigma_str = "auto" if knn.sigma == 0 else f"{knn.sigma:.4f}"
    print(f"[knn] built with k={knn.k}, sigma={_sigma_str}")

    # ---------- VTK setup ----------
    win = vtk.vtkRenderWindow()
    win.SetSize(*args.window_size)
    win.SetWindowName("mindVisualizer — ROI Flow Mode")

    ren_manifold = vtk.vtkRenderer()
    ren_manifold.SetViewport(0.0, 0.0, 0.55, 1.0)
    ren_manifold.SetBackground(0.0, 0.0, 0.0)
    win.AddRenderer(ren_manifold)

    ren_roi = vtk.vtkRenderer()
    ren_roi.SetViewport(0.55, 0.0, 1.0, 1.0)
    ren_roi.SetBackground(0.02, 0.02, 0.04)
    win.AddRenderer(ren_roi)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    # ---------- OOS overlay (reuse build_cloud from src/main.py) ----------
    n_oos = len(oos_pts)
    oos_rgba = np.zeros((n_oos, 4), np.uint8)
    oos_rgba[:, 0] = 100; oos_rgba[:, 1] = 100; oos_rgba[:, 2] = 120; oos_rgba[:, 3] = 72
    _, oos_colors, oos_pd, oos_actor = build_cloud(oos_pts, point_size=1.5, rgba=oos_rgba)
    ren_manifold.AddActor(oos_actor)

    # ---------- MDN Particles ----------
    rng = np.random.default_rng(0)
    n_particles = min(len(seed_pts), 12000)
    sel = rng.choice(len(seed_pts), size=n_particles,
                     replace=len(seed_pts) < n_particles)
    P = seed_pts[sel].copy()
    overlap_sigma = args.respawn_jitter * max(diag, 1e-9)
    P += rng.standard_normal(P.shape).astype(np.float32) * overlap_sigma
    np.clip(P, amin, amax, out=P)

    ttl_lo, ttl_hi = 30, 120
    ttl = rng.integers(ttl_lo, ttl_hi + 1, size=n_particles, dtype=np.int32)
    ages = rng.integers(0, ttl_hi, size=n_particles, dtype=np.int32)

    # Initial color: white with low alpha (like original)
    init_rgba = np.tile(np.array([[255, 255, 255, 48]], np.uint8), (n_particles, 1))
    pts_vtk, colors_arr, p_pd, p_actor = build_cloud(P, point_size=2.0, rgba=init_rgba)
    ren_manifold.AddActor(p_actor)

    dt = [args.dt]
    speed_scale = [args.speed_scale]

    # ---------- ROI Panel ----------
    roi_panel = ROIPanel(ren_roi, C_centers, roi_names)

    # ---------- ROI Flow Dots (particle mode — alternative) ----------
    roi_flow_dots = ROIFlowDots(ren_roi, C_centers)

    # ---------- ROI Path Animation (default mode) ----------
    roi_path_anim = ROIPathAnimation(roi_panel, knn)
    roi_anim_mode = ["path"]  # "path" (default) or "particles"

    # ---------- ROI Path Comparison ----------
    roi_path_compare = ROIPathCompare(roi_panel, knn)
    saved_paths = {"A": None, "B": None}  # saved path point lists
    saved_trail_actors = {"A": None, "B": None}  # trail actors for saved paths

    # ---------- Probe state ----------
    probe_mode = [None]  # None, "flow", "manual"
    probe_pos = [None]
    probe_path = []
    probe_start_roi = [None]
    probe_frozen = [False]
    trail_actor_ref = [None]  # reference to current spline trail actor
    placement_mode = [None]  # None, "flow", "manual" — set by G/M key, consumed by click

    # Probe marker — proportional to manifold size
    probe_radius = diag * 0.008
    probe_sphere = vtk.vtkSphereSource()
    probe_sphere.SetRadius(probe_radius)
    probe_sphere.SetPhiResolution(16)
    probe_sphere.SetThetaResolution(16)
    probe_mapper = vtk.vtkPolyDataMapper()
    probe_mapper.SetInputConnection(probe_sphere.GetOutputPort())
    probe_actor = vtk.vtkActor()
    probe_actor.SetMapper(probe_mapper)
    probe_actor.GetProperty().SetColor(1.0, 1.0, 0.0)
    probe_actor.VisibilityOff()
    ren_manifold.AddActor(probe_actor)

    # ---------- Text overlays (reuse from src/main.py) ----------
    overlay = VtkTextOverlay(ren_manifold, max_lines=6)

    # Legend (top-left, white Courier)
    add_window_legend(ren_manifold, [
        "G then click  place probe (flow)",
        "M then click  manual path mode",
        "Shift+G       freeze & interpret",
        "V             toggle ROI anim mode",
        "1 / 2         save path A / B",
        "D             compare A vs B",
        "C             clear all",
        "+/-           speed  |  Q quit",
    ], font_px=12)

    gpt_pending = {"result": None}

    # ROI mode legend (top-left of ROI viewport)
    roi_legend_actor = vtk.vtkTextActor()
    roi_legend_actor.SetInput("")
    roi_leg_tp = roi_legend_actor.GetTextProperty()
    roi_leg_tp.SetFontFamilyToCourier()
    roi_leg_tp.SetFontSize(11)
    roi_leg_tp.SetColor(0.8, 0.8, 0.8)
    roi_leg_tp.SetOpacity(0.7)
    roi_leg_tp.SetJustificationToLeft()
    roi_leg_tp.SetVerticalJustificationToTop()
    roi_legend_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    roi_legend_actor.GetPositionCoordinate().SetValue(0.02, 0.97)
    ren_roi.AddActor(roi_legend_actor)

    # Static legend for ROI viewport (bottom-left)
    add_window_legend(ren_roi, [
        "V  switch view mode",
    ], font_px=11)

    def _update_roi_legend():
        if roi_flow_dots.is_active():
            roi_legend_actor.SetInput(
                "PARTICLE FLOW\n"
                "Particles: donor -> receiver\n"
                "Burst mode | V to switch")
        elif roi_path_anim.is_active():
            roi_legend_actor.SetInput(
                "PATH ANIMATION\n"
                "Flipping START <-> END\n"
                "V to switch view")
        elif roi_path_compare.is_active():
            roi_legend_actor.SetInput(
                "PATH COMPARISON\n"
                "Cyan=A  Magenta=B")
        else:
            roi_legend_actor.SetInput("")

    # LLM result text panel in the ROI viewport (bottom, green text)
    gpt_text_actor = vtk.vtkTextActor()
    gpt_text_actor.SetInput("")
    gpt_tp = gpt_text_actor.GetTextProperty()
    gpt_tp.SetFontFamilyToCourier()
    gpt_tp.SetFontSize(11)
    gpt_tp.SetColor(0.3, 0.9, 0.3)  # green
    gpt_tp.SetOpacity(0.9)
    gpt_tp.SetJustificationToLeft()
    gpt_tp.SetVerticalJustificationToBottom()
    gpt_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    gpt_text_actor.GetPositionCoordinate().SetValue(0.02, 0.02)
    gpt_text_actor.VisibilityOff()
    ren_roi.AddActor(gpt_text_actor)

    def _word_wrap(text: str, width: int = 50, max_lines: int = 18) -> str:
        """Word-wrap text to fit the ROI viewport."""
        import textwrap
        lines = []
        for paragraph in text.split('\n'):
            wrapped = textwrap.wrap(paragraph, width=width) or ['']
            lines.extend(wrapped)
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + ['...']
        return '\n'.join(lines)

    # ---------- Helper: remove trail ----------
    def _remove_trail():
        if trail_actor_ref[0] is not None:
            ren_manifold.RemoveActor(trail_actor_ref[0])
            trail_actor_ref[0] = None

    # ---------- Helper: rebuild trail from path ----------
    def _rebuild_trail():
        _remove_trail()
        if len(probe_path) >= 2:
            pts_arr = np.array(probe_path, dtype=np.float32)
            actor = _build_spline_trail(pts_arr, diag)
            if actor is not None:
                ren_manifold.AddActor(actor)
                trail_actor_ref[0] = actor

    # ---------- Helper: place probe ----------
    def _place_probe(pos, mode):
        pos = np.clip(pos, amin, amax)
        probe_mode[0] = mode
        probe_frozen[0] = False
        probe_pos[0] = pos.copy()
        probe_path.clear()
        probe_path.append(pos.copy())
        probe_start_roi[0] = knn.query(pos)

        roi_panel.update_values(probe_start_roi[0])
        roi_flow_dots.stop()
        roi_path_anim.stop()
        roi_path_compare.stop()
        _remove_trail()

        probe_sphere.SetCenter(*pos)
        probe_actor.VisibilityOn()

        overlay.clear_log()
        overlay.hide_gpt()
        gpt_pending["result"] = None

        mode_label = "following flow" if mode == "flow" else "manual (click to extend)"
        overlay.add_log(f"Probe placed [{mode_label}]")
        print(f"[probe] placed at ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) [{mode}]")

    # ---------- Helper: freeze and analyze ----------
    def _freeze_and_analyze():
        if probe_mode[0] is None or probe_frozen[0] or probe_pos[0] is None:
            return
        probe_frozen[0] = True

        end_roi = knn.query(probe_pos[0])
        delta = analyzer.compute_delta(probe_start_roi[0], end_roi)

        overlay.add_log("Probe frozen — asking LLM...")
        print("[probe] frozen, computing ROI delta...")

        context = analyzer.build_llm_context(delta)
        print(f"\n{context}\n")

        # Start ROI animation (path mode or particle mode)
        if roi_anim_mode[0] == "path":
            roi_flow_dots.stop()
            roi_path_anim.build_from_path(probe_path, n_samples=60)
        else:
            roi_path_anim.stop()
            roi_panel.update_values(delta)
            roi_flow_dots.start_from_delta(delta)
            roi_panel.dim_spheres(0.15)
        _update_roi_legend()

        # Rebuild trail as smooth spline now that path is complete
        _rebuild_trail()

        def _gpt_worker():
            gpt_pending["result"] = llm.interpret_roi_flow(context)

        t = threading.Thread(target=_gpt_worker, daemon=True)
        t.start()

    # ---------- Helper: reset probe (keep saved paths) ----------
    def _reset_probe():
        probe_mode[0] = None
        probe_frozen[0] = False
        probe_pos[0] = None
        probe_path.clear()
        probe_start_roi[0] = None
        probe_actor.VisibilityOff()
        _remove_trail()
        roi_panel.reset_colors()
        roi_panel.restore_spheres()
        roi_flow_dots.stop()
        roi_path_anim.stop()
        roi_path_compare.stop()
        roi_legend_actor.SetInput("")
        gpt_text_actor.SetInput("")
        gpt_text_actor.VisibilityOff()
        overlay.clear_log()
        overlay.hide_gpt()
        gpt_pending["result"] = None

    # ---------- Helper: clear ----------
    def _clear_all():
        probe_mode[0] = None
        probe_frozen[0] = False
        probe_pos[0] = None
        probe_path.clear()
        probe_start_roi[0] = None
        probe_actor.VisibilityOff()
        _remove_trail()
        roi_panel.reset_colors()
        roi_panel.restore_spheres()
        roi_flow_dots.stop()
        roi_path_anim.stop()
        roi_path_compare.stop()
        # Remove saved path trails
        for k in ("A", "B"):
            if saved_trail_actors[k] is not None:
                ren_manifold.RemoveActor(saved_trail_actors[k])
                saved_trail_actors[k] = None
        overlay.clear_log()
        overlay.hide_gpt()
        gpt_text_actor.SetInput("")
        gpt_text_actor.VisibilityOff()
        roi_legend_actor.SetInput("")
        gpt_pending["result"] = None

    # ---------- Callbacks ----------
    trail_rebuild_counter = [0]

    def on_key(obj, event):
        key = iren.GetKeySym()
        shift = bool(iren.GetShiftKey())

        if key == "g" and not shift:
            # Enter flow placement mode — next click places probe
            placement_mode[0] = "flow"
            overlay.add_log("Click to place probe (flow mode)")

        elif key == "m" and not shift:
            if probe_mode[0] == "manual" and not probe_frozen[0] and probe_pos[0] is not None:
                # Already in manual mode with active probe — extend path
                placement_mode[0] = "manual_extend"
                overlay.add_log("Click to add path point")
            else:
                # Start new manual path
                placement_mode[0] = "manual"
                overlay.add_log("Click to place probe (manual mode)")

        elif key == "G" or (key == "g" and shift):
            _freeze_and_analyze()

        elif key in ("c", "C"):
            placement_mode[0] = None
            _clear_all()

        elif key in ("plus", "equal"):
            speed_scale[0] *= 1.25
            overlay.add_log(f"Speed: {speed_scale[0]:.2f}")
        elif key == "minus":
            speed_scale[0] /= 1.25
            overlay.add_log(f"Speed: {speed_scale[0]:.2f}")

        elif key == "v" and not shift:
            # Toggle between path animation and particle flow mode
            if roi_anim_mode[0] == "path":
                roi_anim_mode[0] = "particles"
                overlay.add_log("ROI mode: particles")
            else:
                roi_anim_mode[0] = "path"
                overlay.add_log("ROI mode: path animation")
            # If frozen, restart with new mode
            if probe_frozen[0] and probe_pos[0] is not None:
                end_roi = knn.query(probe_pos[0])
                delta = analyzer.compute_delta(probe_start_roi[0], end_roi)
                if roi_anim_mode[0] == "path":
                    roi_flow_dots.stop()
                    roi_panel.restore_spheres()
                    roi_path_anim.build_from_path(probe_path, n_samples=60)
                else:
                    roi_path_anim.stop()
                    roi_panel.update_values(delta)
                    roi_flow_dots.start_from_delta(delta)
                    roi_panel.dim_spheres(0.15)
                _update_roi_legend()

        elif key == "1" and not shift:
            # Save current path as Path A, then reset probe for next path
            if len(probe_path) >= 2:
                saved_paths["A"] = [p.copy() for p in probe_path]
                overlay.add_log(f"Path A saved ({len(probe_path)} pts)")
                print(f"[compare] Path A saved: {len(probe_path)} points")
                if saved_trail_actors["A"] is not None:
                    ren_manifold.RemoveActor(saved_trail_actors["A"])
                ta = _build_spline_trail(np.array(probe_path, np.float32), diag)
                if ta is not None:
                    ta.GetProperty().SetColor(0.0, 0.9, 0.9)  # cyan
                    ta.GetProperty().SetOpacity(0.6)
                    ren_manifold.AddActor(ta)
                    saved_trail_actors["A"] = ta
                # Reset probe so next path starts fresh
                _reset_probe()
            else:
                overlay.add_log("Need a path first (place+freeze probe)")

        elif key == "2" and not shift:
            # Save current path as Path B, then reset probe for next path
            if len(probe_path) >= 2:
                saved_paths["B"] = [p.copy() for p in probe_path]
                overlay.add_log(f"Path B saved ({len(probe_path)} pts)")
                print(f"[compare] Path B saved: {len(probe_path)} points")
                if saved_trail_actors["B"] is not None:
                    ren_manifold.RemoveActor(saved_trail_actors["B"])
                tb = _build_spline_trail(np.array(probe_path, np.float32), diag)
                if tb is not None:
                    tb.GetProperty().SetColor(0.9, 0.0, 0.9)  # magenta
                    tb.GetProperty().SetOpacity(0.6)
                    ren_manifold.AddActor(tb)
                    saved_trail_actors["B"] = tb
                _reset_probe()
            else:
                overlay.add_log("Need a path first (place+freeze probe)")

        elif key == "d" and not shift:
            # Compare Path A vs Path B with LLM analysis
            if saved_paths["A"] is None or saved_paths["B"] is None:
                overlay.add_log("Save Path A (1) and Path B (2) first")
            else:
                roi_flow_dots.stop()
                roi_path_anim.stop()
                roi_panel.restore_spheres()
                roi_path_compare.build(saved_paths["A"], saved_paths["B"])
                overlay.add_log("Comparing Path A (cyan) vs B (magenta)...")
                print("[compare] Started path comparison animation")

                # Compute deltas and LLM context for both paths
                pa, pb = saved_paths["A"], saved_paths["B"]
                roi_start_a = knn.query(np.array(pa[0], np.float32))
                roi_end_a = knn.query(np.array(pa[-1], np.float32))
                delta_a = analyzer.compute_delta(roi_start_a, roi_end_a)
                ctx_a = analyzer.build_llm_context(delta_a)

                roi_start_b = knn.query(np.array(pb[0], np.float32))
                roi_end_b = knn.query(np.array(pb[-1], np.float32))
                delta_b = analyzer.compute_delta(roi_start_b, roi_end_b)
                ctx_b = analyzer.build_llm_context(delta_b)

                print(f"\n--- PATH A ---\n{ctx_a}\n--- PATH B ---\n{ctx_b}\n")

                # LLM comparison in background
                def _compare_worker():
                    gpt_pending["result"] = llm.compare_two_paths(ctx_a, ctx_b)

                t = threading.Thread(target=_compare_worker, daemon=True)
                t.start()
                overlay.add_log("Asking LLM to compare paths...")
                _update_roi_legend()

        elif key in ("q", "Escape"):
            iren.TerminateApp()

    iren.AddObserver("KeyPressEvent", on_key)

    def on_click(obj, event):
        if placement_mode[0] is None:
            return  # No placement pending — normal click behavior

        x, y = iren.GetEventPosition()
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(x, y, 0, ren_manifold)
        pos = np.array(picker.GetPickPosition(), dtype=np.float32)
        pos = np.clip(pos, amin, amax)

        mode = placement_mode[0]
        placement_mode[0] = None  # Consume the placement

        if mode == "flow":
            _place_probe(pos, "flow")
        elif mode == "manual":
            _place_probe(pos, "manual")
        elif mode == "manual_extend":
            probe_pos[0] = pos.copy()
            probe_path.append(pos.copy())
            probe_sphere.SetCenter(*pos)
            _rebuild_trail()
            roi_vec = knn.query(pos)
            roi_panel.update_values(roi_vec)
            overlay.add_log(f"Path point {len(probe_path)}")

    iren.AddObserver("LeftButtonPressEvent", on_click)

    # ---------- Timer loop ----------
    def on_timer(obj, event):
        nonlocal P, ages, ttl

        # Check for GPT result
        if gpt_pending["result"] is not None:
            text = gpt_pending["result"]
            gpt_pending["result"] = None
            overlay.show_gpt(text)
            overlay.add_log("LLM interpretation ready.")
            # Show in ROI viewport as well
            gpt_text_actor.SetInput(_word_wrap(text, width=45, max_lines=16))
            gpt_text_actor.VisibilityOn()
            print(f"\n--- ROI FLOW INTERPRETATION ---\n{text}\n--- END ---\n")

        # Advect particles
        V = sampler.sample_vec(P)
        step = dt[0] * speed_scale[0] * (target_step / max(vmax_mean, 1e-9))
        P[:] += V * step
        np.clip(P, amin, amax, out=P)

        # Death/respawn
        ages += 1
        dead = ages >= ttl
        if np.any(dead):
            n_dead = dead.sum()
            sel2 = rng.integers(0, len(seed_pts), size=n_dead)
            base = seed_pts[sel2]
            J = rng.standard_normal((n_dead, 3)).astype(np.float32) * overlap_sigma
            P[dead] = np.clip(base + J, amin, amax)
            ages[dead] = 0
            ttl[dead] = rng.integers(ttl_lo, ttl_hi + 1, size=n_dead, dtype=np.int32)

        # Update colors (vectorized turbo colormap)
        speeds = np.linalg.norm(V, axis=1)
        s_max = float(np.percentile(speeds, 97)) + 1e-8
        t_vals = np.clip(speeds / s_max, 0.0, 1.0)
        rgb = turbo_rgb01(t_vals)
        alpha = np.full(n_particles, 200, np.uint8)
        rgba = np.concatenate([rgb, alpha[:, None]], axis=1)

        pts_vtk.SetData(numpy_to_vtk(P, deep=True))
        colors_arr.DeepCopy(numpy_to_vtk(rgba, deep=True))
        colors_arr.Modified()
        p_pd.Modified()

        # Advect probe (flow mode)
        if probe_mode[0] == "flow" and not probe_frozen[0] and probe_pos[0] is not None:
            pos = probe_pos[0]
            v = sampler.sample_vec(pos.reshape(1, 3))[0]
            pos += v * step
            np.clip(pos, amin, amax, out=pos)
            probe_pos[0] = pos
            probe_sphere.SetCenter(*pos)
            probe_path.append(pos.copy())

            # Rebuild trail periodically (every 20 steps) for smooth spline
            trail_rebuild_counter[0] += 1
            if trail_rebuild_counter[0] >= 20:
                trail_rebuild_counter[0] = 0
                _rebuild_trail()

            if len(probe_path) % 5 == 0:
                roi_vec = knn.query(pos)
                roi_panel.update_values(roi_vec)

        # Tick ROI animation (path, particles, or compare — whichever is active)
        if roi_path_anim.is_active():
            roi_path_anim.tick()
        if roi_flow_dots.is_active():
            roi_flow_dots.tick()
        if roi_path_compare.is_active():
            roi_path_compare.tick()

        win.Render()

    iren.Initialize()
    iren.CreateRepeatingTimer(int(1000 / args.fps))
    iren.AddObserver("TimerEvent", on_timer)

    ren_manifold.ResetCamera()
    ren_roi.ResetCamera()

    print("\n=== ROI Flow Mode ===")
    print("  G then click  — place probe (follows flow)")
    print("  M then click  — manual path mode (click to extend)")
    print("  Shift+G       — freeze & interpret ROI flow")
    print("  V             — toggle ROI anim: path ↔ particles")
    print("  1 / 2         — save current path as A / B")
    print("  D             — compare Path A vs Path B")
    print("  C             — clear all")
    print("  +/-           — speed scale")
    print("  Q             — quit")
    print("=" * 40)

    win.Render()
    iren.Start()


if __name__ == "__main__":
    main()
