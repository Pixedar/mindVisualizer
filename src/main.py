#!/usr/bin/env python3
"""mindVisualizer — Brain flow particle visualizer with probe analysis.

Filtered and modular version of simulate_mdn_flow.py, focused on:
  - MDN particle flow with OOS seeding and far-from-OOS death acceleration
  - Flow mesh overlay (all available brain region meshes, key: q/w)
  - Probe system: single, multi-probe, and branching modes
  - GPT-powered region transition analysis (RAG or direct)
  - Brain region state database with perturbation propagation

Run with no arguments for defaults:
  python -m src.main

Key bindings:
  space       pause/resume
  f           cycle field (mean / gated / comp1..6)
  v           cycle color mode (SPEED / ENTROPY / DELTA_ENTROPY / DIR_DELTA_ENTROPY)
  m           cycle colormap (TURBO / RAINBOW)
  q / w       cycle flow mesh (prev / next)
  \\ (backslash) hide flow mesh
  9 / 0       mesh opacity -/+
  [ / ]       dt -/+
  + / -       speed scale +/-
  1 / 2       particle lifetime -/+
  7 / 8       particle opacity -/+
  y           toggle speed filter
  u           filter direction (top/bottom)
  z / x       filter fraction -/+
  o           toggle OOS overlay
  g           place probe (click in scene first)
  G (shift+g) analyze probe path with GPT (async)
  b           toggle branching mode
  n           cycle multi-probe count (1/4/8)
  s           initialize brain states (prompts in console)
  S (shift+s) propagate state through probe path
  c           clear probe
  Escape      quit
"""

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from scipy.spatial import cKDTree

from .field_loader import load_field, TriLinearSampler, load_points_any
from .colormaps import turbo_rgb01, rainbow_rgb01, bicolor_white_center
from .mesh_overlay import FlowMeshOverlay
from .probe import ProbeSystem
from .region_analyzer import (analyze_probe_path, analyze_with_gpt,
                               format_transitions_text)
from .brain_state import BrainStateDB

# ---------- paths (relative to project root) ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MDN_DIR = DATA_DIR / "mdn"
MESH_DIR = DATA_DIR / "meshes"
ALIGNMENT_FILE = DATA_DIR / "brain_alignment.json"
DEFAULT_META = MDN_DIR / "mdn_particles_rdcim_teacher_edge_hq_grid125_meta.json"
DEFAULT_OOS = MDN_DIR / "mdn_particles_rdcim_teacher_edge_hq_training_points.npy"
DEFAULT_EXTRA_PARCELLATION = DATA_DIR / "extra_parcellation" / "combined_atlas.nii.gz"
DEFAULT_EXTRA_LABELS = DATA_DIR / "extra_parcellation" / "combined_atlas_labels.json"


# ---------- utility functions ----------

def lattice_positions(n_per_axis, amin, amax, margin=0.02, jitter=0.012, seed=42):
    rng = np.random.default_rng(seed)
    fx = np.linspace(margin, 1.0 - margin, n_per_axis, dtype=np.float32)
    X, Y, Z = np.meshgrid(fx, fx, fx, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    if jitter > 0:
        P += (rng.random(P.shape, dtype=np.float32) - 0.5) * 2.0 * (jitter / max(n_per_axis - 1, 1))
        P = np.clip(P, 0, 1)
    span = (amax - amin).astype(np.float32)
    return (amin + P * span).astype(np.float32)


def wrap_inside(P, amin, amax, margin_frac=0.0):
    span = amax - amin
    offset = amin + margin_frac * span
    inner = span * (1.0 - 2.0 * margin_frac)
    D = P - offset
    D = (D % inner + inner) % inner
    return offset + D


def build_cloud(P_init, point_size=2.0, rgba=None, opacity=1.0):
    N = len(P_init)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(P_init, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_points)
    verts = vtk.vtkCellArray()
    verts.Allocate(N)
    verts.InsertNextCell(N)
    for i in range(N):
        verts.InsertCellPoint(i)
    pd.SetVerts(verts)

    colors = vtkUnsignedCharArray()
    if rgba is not None:
        C = rgba.astype(np.uint8, copy=False)
    else:
        C = np.tile(np.array([[255, 255, 255, 48]], np.uint8), (N, 1))
    colors.SetName("RGBA")
    colors.SetNumberOfComponents(4)
    colors.SetNumberOfTuples(N)
    colors.DeepCopy(numpy_to_vtk(C, deep=True))
    pd.GetPointData().SetScalars(colors)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetColorModeToDirectScalars()
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    actor.GetProperty().SetOpacity(float(opacity))
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetInterpolationToFlat()
    return vtk_points, colors, pd, actor


def compute_fraction_mask(vals, frac, top):
    N = len(vals)
    if N == 0:
        return np.zeros(0, dtype=bool)
    frac = float(np.clip(frac, 0.0, 1.0))
    k = int(np.ceil(N * frac))
    if k <= 0:
        return np.zeros(N, dtype=bool)
    if k >= N:
        return np.ones(N, dtype=bool)
    mask = np.zeros(N, dtype=bool)
    if top:
        idx = np.argpartition(vals, N - k)[N - k:]
    else:
        idx = np.argpartition(vals, k)[:k]
    mask[idx] = True
    return mask


def densify_oos_surface(base_pts, full_oos_pts, amin, amax, extra_count,
                        k=16, jitter_frac=0.4, power=2.0, seed=2025):
    if extra_count <= 0 or len(base_pts) == 0 or len(full_oos_pts) == 0:
        return np.zeros((0, 3), np.float32)
    cloud = np.asarray(full_oos_pts, dtype=np.float32)
    m_add = min(extra_count, 500_000)
    eps = 1e-9
    k_eff = int(max(3, min(k, len(cloud))))
    tree = cKDTree(cloud)
    dists, _ = tree.query(base_pts, k=k_eff)
    if dists.ndim == 1:
        dists = dists[:, None]
    r_k = dists[:, -1].astype(np.float32) + eps
    med = float(np.median(r_k)) + eps
    w = np.power(r_k / med, float(power)).astype(np.float64)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        return np.zeros((0, 3), np.float32)
    w /= sw
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(base_pts), size=m_add, replace=True, p=w)
    extras = np.zeros((m_add, 3), np.float32)
    for j, i in enumerate(sel):
        s_loc = float(r_k[i]) * float(jitter_frac)
        extras[j] = base_pts[i] + rng.standard_normal(3).astype(np.float32) * s_loc
    return np.clip(extras, amin, amax).astype(np.float32)


def add_window_legend(ren, lines, font_px=13):
    txt = "\n".join(lines)
    ta = vtk.vtkTextActor()
    ta.SetInput(txt)
    tp = ta.GetTextProperty()
    tp.SetColor(1, 1, 1)
    tp.SetFontSize(int(font_px))
    tp.SetOpacity(0.92)
    tp.SetFontFamilyToCourier()
    ta.SetPosition(10, 10)
    ren.AddActor(ta)
    return ta


class VtkTextOverlay:
    """Scrolling text log + GPT explanation panel in the VTK window."""

    def __init__(self, ren: vtk.vtkRenderer, max_lines: int = 8):
        self._ren = ren
        self._max_lines = max_lines
        self._log_lines: list[str] = []

        self._log_actor = vtk.vtkTextActor()
        self._log_actor.SetInput("")
        tp = self._log_actor.GetTextProperty()
        tp.SetColor(1.0, 0.9, 0.3)
        tp.SetFontSize(13)
        tp.SetOpacity(0.95)
        tp.SetFontFamilyToCourier()
        tp.SetJustificationToRight()
        tp.SetVerticalJustificationToTop()
        self._log_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self._log_actor.GetPositionCoordinate().SetValue(0.99, 0.97)
        ren.AddActor(self._log_actor)

        self._gpt_actor = vtk.vtkTextActor()
        self._gpt_actor.SetInput("")
        tp2 = self._gpt_actor.GetTextProperty()
        tp2.SetColor(0.8, 1.0, 0.8)
        tp2.SetFontSize(14)
        tp2.SetOpacity(0.95)
        tp2.SetFontFamilyToCourier()
        tp2.SetJustificationToRight()
        tp2.SetVerticalJustificationToTop()
        self._gpt_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        self._gpt_actor.GetPositionCoordinate().SetValue(0.99, 0.70)
        self._gpt_actor.VisibilityOff()
        ren.AddActor(self._gpt_actor)

    def add_log(self, text: str):
        self._log_lines.append(text)
        if len(self._log_lines) > self._max_lines:
            self._log_lines = self._log_lines[-self._max_lines:]
        self._log_actor.SetInput("\n".join(self._log_lines))

    def clear_log(self):
        self._log_lines.clear()
        self._log_actor.SetInput("")

    def show_gpt(self, text: str):
        wrapped = []
        for line in text.split("\n"):
            while len(line) > 60:
                brk = line.rfind(" ", 0, 60)
                if brk <= 0:
                    brk = 60
                wrapped.append(line[:brk])
                line = line[brk:].lstrip()
            wrapped.append(line)
        if len(wrapped) > 20:
            wrapped = wrapped[:20] + ["..."]
        self._gpt_actor.SetInput("\n".join(wrapped))
        self._gpt_actor.VisibilityOn()

    def hide_gpt(self):
        self._gpt_actor.SetInput("")
        self._gpt_actor.VisibilityOff()


# ---------- main ----------

def main():
    # Fix SSL cert path (Git on Windows sets bad path)
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
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser(description="mindVisualizer - Brain flow particle visualizer")
    ap.add_argument("--meta", type=Path, default=DEFAULT_META)
    ap.add_argument("--oos", type=Path, default=DEFAULT_OOS)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--max-step-frac", type=float, default=0.01)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--respawn-jitter", type=float, default=0.015)
    ap.add_argument("--oos-seed-limit", type=int, default=50000)
    ap.add_argument("--oos-fill-count", type=int, default=10000)
    ap.add_argument("--overlap-frac", type=float, default=0.01)
    ap.add_argument("--overlap-jitter", type=float, default=0.002)
    ap.add_argument("--overlap-limit", type=int, default=200000)
    ap.add_argument("--no-far-from-oos", action="store_true")
    ap.add_argument("--oos-dist-grid", type=int, default=96)
    ap.add_argument("--oos-dist-thresh-frac", type=float, default=0.009)
    ap.add_argument("--oos-dist-gamma", type=float, default=3.0)
    ap.add_argument("--oos-death-boost", type=float, default=8.0)
    ap.add_argument("--no-flow-mesh", action="store_true")
    ap.add_argument("--no-rag", action="store_true",
                    help="Skip RAG, use only GPT's own knowledge")
    ap.add_argument("--multi-probe", type=int, default=1,
                    help="Number of probes to spawn (1=single, 4=neighborhood)")
    ap.add_argument("--branching", action="store_true",
                    help="Enable MDN component branching")
    ap.add_argument("--window-size", type=int, nargs=2, default=[1200, 800])
    ap.add_argument("--hq", action="store_true",
                    help="Use high-quality GPT model (gpt-5.4) instead of gpt-5.4-mini")
    ap.add_argument("--debug", action="store_true",
                    help="Print full LLM prompts to console before each call")
    ap.add_argument("--extra-parcellation", type=Path, default=None,
                    help="Path to NIfTI parcellation file for finer subregion labels "
                         "(e.g., Brainnetome BN_Atlas_246_1mm.nii.gz)")
    ap.add_argument("--extra-parcellation-labels", type=Path, default=None,
                    help="JSON label map for extra parcellation (label_id -> name)")
    args = ap.parse_args()

    model = "gpt-5.4" if args.hq else "gpt-5.4-mini"
    print(f"[config] LLM model: {model}")

    far_from_oos = not args.no_far_from_oos
    flow_mesh_enabled = not args.no_flow_mesh
    use_rag = not args.no_rag
    debug_mode = args.debug

    # ---------- load field ----------
    print(f"[field] loading {args.meta} ...")
    fld = load_field(args.meta)
    G = fld["G"]
    amin, amax = fld["amin"], fld["amax"]
    V_mean = fld["mean"]
    MUS = fld["mus"]
    PI = fld["pi"]
    ENT = fld["ENT"]
    K = len(MUS)

    sampler_mean = TriLinearSampler(V_mean, amin, amax)
    diag = float(np.linalg.norm(amax - amin))
    target_step = args.max_step_frac * max(diag, 1e-6)

    def interior_percentile(V, q=100.0):
        mag = np.linalg.norm(V.reshape(-1, 3), axis=1)
        nz = mag[mag > 0]
        return float(np.percentile(nz, q)) if nz.size else 1.0

    vmax_mean = interior_percentile(V_mean, q=100.0)

    sampler_gated = None
    V_gated = None
    if PI is not None and PI.ndim == 4 and PI.shape[-1] == K and K > 0:
        idx = np.argmax(PI, axis=-1)
        V_gated = np.zeros_like(MUS[0], dtype=np.float32)
        for k_i in range(K):
            mask = (idx == k_i)[..., None].astype(np.float32)
            V_gated += MUS[k_i] * mask
        sampler_gated = TriLinearSampler(V_gated, amin, amax)

    samplers_comp = [TriLinearSampler(mu, amin, amax) for mu in MUS]

    field_modes = ["mean"]
    if sampler_gated is not None:
        field_modes.append("gated")
    field_modes += [f"comp{i + 1}" for i in range(K)]
    field_mode = [field_modes[0]]

    sampler_by_name = {"mean": sampler_mean}
    if sampler_gated is not None:
        sampler_by_name["gated"] = sampler_gated
    for i, s in enumerate(samplers_comp):
        sampler_by_name[f"comp{i + 1}"] = s

    vmax_by_name = {"mean": vmax_mean}
    if V_gated is not None:
        vmax_by_name["gated"] = interior_percentile(V_gated, q=100.0)
    for i, mu in enumerate(MUS):
        vmax_by_name[f"comp{i + 1}"] = interior_percentile(mu, q=100.0)

    def current_sampler():
        return sampler_by_name.get(field_mode[0], sampler_mean)
    def vmax_for_mode():
        return vmax_by_name.get(field_mode[0], vmax_mean)

    ent_min = ent_max = 0.0
    if ENT is not None:
        ent_min, ent_max = float(np.min(ENT)), float(np.max(ENT))

    # ---------- load OOS ----------
    oos_pts = None
    if args.oos is not None:
        try:
            raw = load_points_any(Path(args.oos))
            if raw.size > 0:
                oos_pts = raw.astype(np.float32)
                print(f"[oos] loaded {len(oos_pts)} points")
        except Exception as e:
            print("[oos] failed to load:", e)

    # ---------- seeding ----------
    rng = np.random.default_rng(2025)
    overlap_radius = float(args.overlap_frac) * max(diag, 1e-9)
    overlap_sigma = float(args.overlap_jitter) * max(diag, 1e-9)
    seed_mode = "grid"
    seed_points = None
    n_axis = None

    if fld["TRAIN"] is not None and oos_pts is not None and len(fld["TRAIN"]) > 0 and len(oos_pts) > 0:
        try:
            tree = cKDTree(fld["TRAIN"].astype(np.float32))
            dists, _ = tree.query(oos_pts.astype(np.float32), k=1)
            cand = oos_pts[dists <= overlap_radius]
            if cand.size > 0:
                seed_mode = "overlap"
                if args.overlap_limit and len(cand) > args.overlap_limit:
                    idx = rng.choice(len(cand), size=int(args.overlap_limit), replace=False)
                    cand = cand[idx]
                seed_points = cand.astype(np.float32)
                print(f"[seed] overlap: {len(seed_points)} points")
        except Exception as e:
            print("[seed] overlap failed:", e)

    if seed_mode == "overlap" and seed_points is not None:
        if args.oos_seed_limit > 0 and len(seed_points) > args.oos_seed_limit:
            prev = len(seed_points)
            idx_seed = rng.choice(prev, size=args.oos_seed_limit, replace=False)
            seed_points = seed_points[idx_seed]
            print(f"[seed] capped to {len(seed_points)} (from {prev})")

    if args.oos_fill_count > 0 and oos_pts is not None and seed_mode == "overlap" and seed_points is not None:
        extras = densify_oos_surface(base_pts=seed_points, full_oos_pts=oos_pts,
                                     amin=amin, amax=amax, extra_count=args.oos_fill_count)
        if extras is not None and len(extras) > 0:
            seed_points = np.concatenate([seed_points, extras], axis=0).astype(np.float32)
            print(f"[seed] oos-fill: added {len(extras)} -> total {len(seed_points)}")

    if seed_mode == "overlap" and seed_points is not None:
        if overlap_sigma > 0:
            J = rng.standard_normal(seed_points.shape).astype(np.float32) * overlap_sigma
            P0 = np.clip(seed_points + J, amin, amax)
        else:
            P0 = np.clip(seed_points, amin, amax)
    else:
        n_axis = max(2, G // max(1, args.stride))
        P0 = lattice_positions(n_axis, amin, amax, margin=0.02, jitter=args.respawn_jitter, seed=2025)

    P = P0.copy()
    Np = len(P)
    _base_ttl_lo, _base_ttl_hi = max(8, 60 // 2), 60
    ttl_state = {"scale": 1.0}

    def _sample_ttl(n):
        base = rng.integers(_base_ttl_lo, _base_ttl_hi + 1, size=int(n))
        return np.maximum(1, np.round(base * ttl_state["scale"])).astype(np.int32)

    ttl = _sample_ttl(Np)
    ages = rng.integers(0, np.maximum(1, ttl), size=Np, dtype=np.int32)

    ent_prev = sampler_mean.sample_scalar(ENT, P).astype(np.float32) if ENT is not None else None
    ent_d_ema = np.zeros(Np, np.float32) if ENT is not None else None
    ent_alpha = 0.2

    # ---------- VTK setup ----------
    init_rgba = np.tile(np.array([[255, 255, 255, 48]], np.uint8), (Np, 1))
    points, color_arr, poly, p_actor = build_cloud(P, point_size=2.0, rgba=init_rgba)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    ren.AddActor(p_actor)
    ren.ResetCamera()

    o_actor = None
    if oos_pts is not None and len(oos_pts) > 0:
        rgb_oos = np.tile(np.array([[255, 255, 255]], np.uint8), (len(oos_pts), 1))
        oos_rgba = np.concatenate([rgb_oos, np.full((len(oos_pts), 1), 72, np.uint8)], axis=1)
        _, _, _, o_actor = build_cloud(oos_pts, point_size=1.5, rgba=oos_rgba, opacity=72 / 255.0)
    oos_visible = [False]

    # OOS distance grid
    OOS_DIST = None
    sampler_oosdist = None
    far_oos_params = None
    if far_from_oos and oos_pts is not None and len(oos_pts) > 0:
        Ng = int(max(8, args.oos_dist_grid))
        print(f"[oos-dist] precomputing {Ng}^3 ...")
        tree_oos = cKDTree(oos_pts.astype(np.float32))
        gridP = lattice_positions(Ng, amin, amax, margin=0.0, jitter=0.0, seed=0)
        d, _ = tree_oos.query(gridP, k=1)
        OOS_DIST = d.reshape(Ng, Ng, Ng).astype(np.float32)
        sampler_oosdist = TriLinearSampler(np.zeros((Ng, Ng, Ng, 3), np.float32), amin, amax)
        far_oos_params = {
            "d0": float(args.oos_dist_thresh_frac) * diag,
            "gamma": float(args.oos_dist_gamma),
            "boost": float(args.oos_death_boost),
        }
        print(f"[oos-dist] ready: d0={far_oos_params['d0']:.4g}")

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetSize(*args.window_size)
    win.SetWindowName("mindVisualizer")

    # Flow mesh overlay
    flow_mesh = None
    if flow_mesh_enabled:
        try:
            flow_mesh = FlowMeshOverlay(ren=ren, win=win,
                                        mesh_dir=MESH_DIR, alignment_file=ALIGNMENT_FILE)
            n_regions = len(flow_mesh.get_all_region_keys())
            print(f"[flow mesh] ready ({n_regions} regions)")
            # Load or build voxel label grid for fast point-in-region queries
            grid_cache = DATA_DIR / "label_grid_cache.npz"
            if not flow_mesh.load_label_grid(grid_cache):
                flow_mesh.build_label_grid()
                flow_mesh.save_label_grid(grid_cache)
        except Exception as e:
            print("[flow mesh] disabled:", e)
            flow_mesh = None

    # Optional extra parcellation (auto-detect combined atlas if no explicit path)
    if flow_mesh:
        _ep_nifti = args.extra_parcellation
        _ep_labels = args.extra_parcellation_labels
        if _ep_nifti is None and DEFAULT_EXTRA_PARCELLATION.exists():
            _ep_nifti = DEFAULT_EXTRA_PARCELLATION
            if _ep_labels is None and DEFAULT_EXTRA_LABELS.exists():
                _ep_labels = DEFAULT_EXTRA_LABELS
        if _ep_nifti is not None:
            try:
                from .extra_parcellation import ExtraParcellation
                extra = ExtraParcellation(_ep_nifti, _ep_labels)
                if extra.load():
                    flow_mesh.set_extra_parcellation(extra)
            except Exception as e:
                print(f"[extra-parcellation] disabled: {e}")

    # Text overlay
    text_overlay = VtkTextOverlay(ren)

    # Brain state database
    brain_state_db = BrainStateDB(model=model, debug=args.debug)

    # Probe system
    probe_sys = ProbeSystem(ren, win, amin, amax)
    if flow_mesh is not None:
        probe_sys.set_mesh_overlay(flow_mesh)
    probe_sys.set_multi_count(args.multi_probe)
    if args.branching and PI is not None:
        probe_sys.set_branching(True, pi_field=PI, mus_samplers=samplers_comp)

    # Boundary constraint
    if sampler_oosdist is not None and OOS_DIST is not None and far_oos_params is not None:
        _boundary_d0 = far_oos_params["d0"] * 2.0

        def _probe_boundary_check(pos):
            d = float(sampler_oosdist.sample_scalar(OOS_DIST, pos[None, :])[0])
            return d < _boundary_d0

        probe_sys.set_boundary_check(_probe_boundary_check)

    # Region change callback
    def _on_region_change(entered, left, is_ghost=False, label=""):
        tag = " [branch]" if is_ghost else ""
        for name in left:
            text_overlay.add_log(f"LEFT{tag}: {name}")
        for name in entered:
            text_overlay.add_log(f"ENTERED{tag}: {name}")

    probe_sys.set_on_region_change(_on_region_change)

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.01)

    # ---------- state ----------
    available_colour_modes = (
        ["SPEED"]
        + (["ENTROPY"] if ENT is not None else [])
        + (["DELTA_ENTROPY"] if ENT is not None else [])
        + (["DIR_DELTA_ENTROPY"] if ENT is not None else [])
    )
    colour_idx = [0]
    cmap_mode = ["TURBO"]
    state = {"paused": False, "dt": float(args.dt), "scale": float(args.speed_scale)}
    filter_state = {"mode": "OFF", "dir_top": True, "frac": 1.0}
    clip_state = {"on": False, "frac": 1.0}
    mdn_alpha = [1.0]
    last_rgba = [init_rgba.copy()]
    probe_mode = [False]

    # Async GPT result holder
    gpt_pending = {"result": None, "running": False}

    def map_rgb_from_t(t01):
        return rainbow_rgb01(t01) if cmap_mode[0] == "RAINBOW" else turbo_rgb01(t01)

    def apply_colors(speed_vals, P_world, dent_vals=None, vis_mask=None):
        mode = available_colour_modes[colour_idx[0]]
        eps = 1e-12
        if mode == "ENTROPY" and ENT is not None:
            scal = sampler_mean.sample_scalar(ENT, P_world)
            t = (scal - ent_min) / max(ent_max - ent_min, eps)
            base_rgb = map_rgb_from_t(np.clip(t, 0, 1))
        elif mode == "DELTA_ENTROPY" and dent_vals is not None:
            a = np.abs(dent_vals)
            scale = float(np.percentile(a, 97)) if a.size else 1.0
            if not np.isfinite(scale) or scale <= eps:
                scale = 1.0
            base_rgb = map_rgb_from_t(np.clip(a / scale, 0, 1))
        elif mode == "DIR_DELTA_ENTROPY" and dent_vals is not None:
            base_rgb = bicolor_white_center(dent_vals)
        else:
            smin = float(np.min(speed_vals)) if speed_vals.size else 0.0
            smax = float(np.max(speed_vals)) if speed_vals.size else 1.0
            t = (speed_vals - smin) / max(smax - smin, eps)
            base_rgb = map_rgb_from_t(np.clip(t, 0, 1))

        a_val = int(np.clip(255.0 * mdn_alpha[0], 5, 255))
        a = np.full(len(base_rgb), a_val, np.uint8)
        if vis_mask is not None:
            a = a.copy()
            a[~vis_mask] = 0

        rgba = np.concatenate([base_rgb, a[:, None]], axis=1).astype(np.uint8)
        color_arr.DeepCopy(numpy_to_vtk(rgba, deep=True))
        color_arr.Modified()
        poly.Modified()
        last_rgba[0] = rgba

    # Warm-up
    for _ in range(12):
        V = sampler_mean.sample_vec(P)
        P = wrap_inside(P + V * (args.dt * args.speed_scale) * (target_step / max(vmax_mean, 1e-9)),
                        amin, amax, margin_frac=args.margin)
    if ENT is not None:
        ent_prev = sampler_mean.sample_scalar(ENT, P).astype(np.float32)
        ent_d_ema[:] = 0.0

    # ---------- timer callback ----------
    def on_timer(_o, _e):
        nonlocal P, ages, ttl, ent_prev, ent_d_ema

        # Check for async GPT result
        if gpt_pending["result"] is not None:
            explanation = gpt_pending["result"]
            gpt_pending["result"] = None
            gpt_pending["running"] = False
            print("\n--- GPT INTERPRETATION ---")
            print(explanation)
            print("--- END ---\n")
            text_overlay.show_gpt(explanation)
            text_overlay.add_log("GPT analysis complete")

        if state["paused"]:
            return

        samp = current_sampler()
        Vraw = samp.sample_vec(P)

        vmx = float(vmax_for_mode())
        if clip_state["on"]:
            thr = float(max(1e-9, clip_state["frac"] * vmx))
            s = np.linalg.norm(Vraw, axis=1)
            k = np.minimum(1.0, thr / (s + 1e-9)).astype(np.float32)
            Vstep = Vraw * k[:, None]
            eff_vmax = min(vmx, thr)
        else:
            Vstep = Vraw
            eff_vmax = vmx

        step = state["dt"] * state["scale"] * (target_step / max(eff_vmax, 1e-9))
        P = wrap_inside(P + Vstep * step, amin, amax, margin_frac=args.margin)

        dent = None
        if ENT is not None:
            ent_now = sampler_mean.sample_scalar(ENT, P).astype(np.float32)
            d = ent_now - ent_prev
            ent_d_ema = (1.0 - ent_alpha) * ent_d_ema + ent_alpha * d
            ent_prev = ent_now
            dent = ent_d_ema

        age_inc = np.ones(len(P), np.int32)
        if sampler_oosdist is not None and OOS_DIST is not None and far_oos_params is not None:
            d_oos = sampler_oosdist.sample_scalar(OOS_DIST, P).astype(np.float32)
            d0 = max(1e-9, far_oos_params["d0"])
            t = np.clip((d_oos - d0) / d0, 0.0, 1.0)
            gate = np.power(t, float(max(0.1, far_oos_params["gamma"]))).astype(np.float32)
            extra = np.floor(gate * float(max(0.0, far_oos_params["boost"])) + 1e-9).astype(np.int32)
            age_inc += extra
        ages += age_inc

        dead = ages >= ttl
        if np.any(dead):
            if seed_mode == "overlap" and seed_points is not None and len(seed_points) > 0:
                sel = rng.integers(0, len(seed_points), size=dead.sum())
                base = seed_points[sel]
                if overlap_sigma > 0:
                    J = rng.standard_normal((dead.sum(), 3)).astype(np.float32) * overlap_sigma
                    P[dead] = np.clip(base + J, amin, amax)
                else:
                    P[dead] = np.clip(base, amin, amax)
            else:
                jitter_world = (amax - amin) * (args.respawn_jitter / max((n_axis or 2) - 1, 1))
                J = (rng.random((dead.sum(), 3)).astype(np.float32) - 0.5) * 2.0 * jitter_world
                P[dead] = np.clip(P0[dead] + J, amin, amax)
            ages[dead] = 0
            ttl[dead] = _sample_ttl(dead.sum())
            if ENT is not None:
                ent_prev[dead] = sampler_mean.sample_scalar(ENT, P[dead]).astype(np.float32)
                ent_d_ema[dead] = 0.0

        # Probe step
        if probe_sys.active:
            probe_sys.step(samp, step)

        # Render
        speed = np.linalg.norm(Vstep, axis=1)
        if filter_state["mode"] == "OFF":
            vis_mask = np.ones(len(P), dtype=bool)
        else:
            vis_mask = compute_fraction_mask(speed, filter_state["frac"],
                                             top=filter_state["dir_top"])

        points.SetData(numpy_to_vtk(P, deep=True))
        points.Modified()
        apply_colors(speed, P, dent_vals=dent, vis_mask=vis_mask)
        win.Render()

    # ---------- click handler ----------
    def on_left_click(obj, ev):
        if not probe_mode[0]:
            return
        x, y = obj.GetEventPosition()
        if picker.Pick(x, y, 0, ren) <= 0:
            return
        px, py, pz = picker.GetPickPosition()
        if not np.isfinite([px, py, pz]).all():
            return
        pos = np.array([px, py, pz], np.float32)
        probe_sys.place(pos)
        probe_mode[0] = False
        n = len(probe_sys.probes)
        print(f"[probe] {n} probe(s) placed. Shift+G to analyze, C to clear.")

    # ---------- async GPT ----------
    def _run_gpt_full_async(probe_snapshots, use_rag_flag):
        """Run FULL analysis + GPT in background thread (no VTK calls)."""
        try:
            all_transitions = []
            branch_transitions = []
            for snap in probe_snapshots:
                path_arr = snap["path"]
                field_mags = snap["field_mags"]
                transitions = analyze_probe_path(
                    path_arr, flow_mesh, sample_every=5,
                    field_mags=field_mags if len(field_mags) == len(path_arr) else None,
                    entropy_sampler=sampler_mean if ENT is not None else None,
                    entropy_field=ENT,
                )
                if snap["ghost"]:
                    branch_transitions.extend(transitions)
                else:
                    all_transitions.extend(transitions)

            if not all_transitions:
                gpt_pending["result"] = "No brain regions detected along probe path."
                return

            # Log transitions (console only, no VTK)
            print("\n--- PROBE TRAJECTORY ANALYSIS ---")
            for i, t in enumerate(all_transitions, 1):
                print(f"  {i}. {t['region_name']} ({t['relative_position']}), "
                      f"steps {t['entry_idx']}-{t['exit_idx']}")
            if branch_transitions:
                print(f"  + {len(branch_transitions)} branch transitions")

            rag_label = "RAG" if use_rag_flag else "direct"
            print(f"\n[GPT] Sending to GPT ({rag_label})...")
            explanation = analyze_with_gpt(all_transitions, use_rag=use_rag_flag,
                                            model=model, debug=debug_mode)
            gpt_pending["result"] = explanation
        except Exception as e:
            gpt_pending["result"] = f"[GPT ERROR] {e}"

    # ---------- key handler ----------
    def on_keypress(obj, ev):
        nonlocal ttl, ages
        key = obj.GetKeySym()
        key_lower = key.lower() if key else ""
        shift = bool(obj.GetShiftKey())

        if key_lower == "escape":
            obj.TerminateApp()

        elif key_lower == "space":
            state["paused"] = not state["paused"]
            print(f"[{'PAUSED' if state['paused'] else 'RUNNING'}]")

        elif key_lower == "f":
            i = field_modes.index(field_mode[0])
            field_mode[0] = field_modes[(i + 1) % len(field_modes)]
            print(f"[field] {field_mode[0]}")

        elif key_lower == "v":
            if available_colour_modes:
                colour_idx[0] = (colour_idx[0] + 1) % len(available_colour_modes)
                print(f"[colour] {available_colour_modes[colour_idx[0]]}")

        elif key_lower == "m":
            cmap_mode[0] = "RAINBOW" if cmap_mode[0] == "TURBO" else "TURBO"
            print(f"[cmap] {cmap_mode[0]}")

        # Flow mesh
        elif key_lower == "q" and flow_mesh:
            flow_mesh.cycle(-1)
        elif key_lower == "w" and flow_mesh:
            flow_mesh.cycle(+1)
        elif key_lower == "backslash" and flow_mesh:
            flow_mesh.hide()
        elif key_lower == "9" and flow_mesh:
            flow_mesh.set_opacity(1.0 / 1.25)
        elif key_lower == "0" and flow_mesh:
            flow_mesh.set_opacity(1.25)

        # Sim controls
        elif key_lower == "bracketleft":
            state["dt"] /= 1.25
            print(f"[dt] {state['dt']:.4f}")
        elif key_lower == "bracketright":
            state["dt"] *= 1.25
            print(f"[dt] {state['dt']:.4f}")
        elif key_lower in ("plus", "equal"):
            state["scale"] *= 1.25
            print(f"[speed-scale] {state['scale']:.3f}")
        elif key_lower in ("minus", "underscore"):
            state["scale"] /= 1.25
            print(f"[speed-scale] {state['scale']:.3f}")

        elif key_lower == "1":
            ttl_state["scale"] = float(np.clip(ttl_state["scale"] / 1.25, 0.1, 100.0))
            ttl[:] = np.maximum(1, np.round(ttl * (1.0 / 1.25))).astype(np.int32)
            print(f"[lifetime] scale = {ttl_state['scale']:.3f}")
        elif key_lower == "2":
            ttl_state["scale"] = float(np.clip(ttl_state["scale"] * 1.25, 0.1, 100.0))
            ttl[:] = np.maximum(1, np.round(ttl * 1.25)).astype(np.int32)
            print(f"[lifetime] scale = {ttl_state['scale']:.3f}")

        elif key_lower == "7":
            mdn_alpha[0] = float(np.clip(mdn_alpha[0] / 1.25, 0.02, 1.0))
            print(f"[alpha] {mdn_alpha[0]:.2f}")
        elif key_lower == "8":
            mdn_alpha[0] = float(np.clip(mdn_alpha[0] * 1.25, 0.02, 1.0))
            print(f"[alpha] {mdn_alpha[0]:.2f}")

        elif key_lower == "y":
            filter_state["mode"] = "OFF" if filter_state["mode"] == "SPEED" else "SPEED"
            print(f"[filter] mode={filter_state['mode']}")
        elif key_lower == "u":
            filter_state["dir_top"] = not filter_state["dir_top"]
            print(f"[filter] dir={'TOP' if filter_state['dir_top'] else 'BOTTOM'}")
        elif key_lower == "z":
            filter_state["frac"] = float(np.clip(filter_state["frac"] / 1.25, 0.0, 1.0))
            print(f"[filter] frac={filter_state['frac'] * 100:.1f}%")
        elif key_lower == "x":
            filter_state["frac"] = float(np.clip(filter_state["frac"] * 1.25, 0.0, 1.0))
            print(f"[filter] frac={filter_state['frac'] * 100:.1f}%")

        elif key_lower == "j":
            clip_state["on"] = not clip_state["on"]
            print(f"[speed-clip] {'ON' if clip_state['on'] else 'OFF'}")
        elif key_lower == "a":
            clip_state["frac"] = float(np.clip(clip_state["frac"] / 1.25, 0.02, 10.0))
            print(f"[speed-clip] frac={clip_state['frac']:.3f}")
        elif key_lower == "d":
            clip_state["frac"] = float(np.clip(clip_state["frac"] * 1.25, 0.02, 10.0))
            print(f"[speed-clip] frac={clip_state['frac']:.3f}")

        elif key_lower == "o":
            if o_actor is not None:
                if oos_visible[0]:
                    ren.RemoveActor(o_actor)
                    oos_visible[0] = False
                else:
                    ren.AddActor(o_actor)
                    oos_visible[0] = True
                print(f"[oos overlay] visible = {oos_visible[0]}")
                win.Render()

        # ---------- probe controls ----------
        elif key_lower == "g" and not shift:
            probe_mode[0] = True
            print("[probe] Click in the scene to place probe...")

        elif key_lower == "g" and shift:
            if not probe_sys.active:
                print("[probe] No probe active. Press g then click.")
                text_overlay.add_log("No probe to analyze")
            elif flow_mesh is None:
                print("[probe] Flow mesh required.")
            elif gpt_pending["running"]:
                print("[probe] GPT analysis already running...")
            else:
                # Stop probe movement — we're analyzing the path so far
                probe_sys.freeze()
                print("[probe] Probe stopped. Analyzing trajectory...")
                text_overlay.add_log("Please wait — analyzing probe path...")
                win.Render()

                # Collect probe data snapshots (lightweight, no VTK calls)
                probe_snapshots = []
                for p in probe_sys.get_all_probes():
                    if not p.path or len(p.path) < 3:
                        continue
                    probe_snapshots.append({
                        "path": p.get_path_array().copy(),
                        "field_mags": p.get_field_mags_array().copy(),
                        "ghost": p.ghost,
                        "label": p.label,
                    })

                if not probe_snapshots:
                    print("[probe] No probe data.")
                    text_overlay.add_log("No probe data")
                else:
                    gpt_pending["running"] = True
                    t = threading.Thread(
                        target=_run_gpt_full_async,
                        args=(probe_snapshots, use_rag),
                        daemon=True
                    )
                    t.start()

        elif key_lower == "c":
            probe_sys.clear()
            probe_mode[0] = False
            text_overlay.clear_log()
            text_overlay.hide_gpt()

        # ---------- branching / multi-probe ----------
        elif key_lower == "b":
            new_state = not probe_sys._branching_enabled
            if PI is not None:
                probe_sys.set_branching(new_state, pi_field=PI, mus_samplers=samplers_comp)
                text_overlay.add_log(f"Branching {'ON' if new_state else 'OFF'}")
            else:
                print("[probe] No PI field - branching not available")

        elif key_lower == "n":
            counts = [1, 4, 8]
            cur = probe_sys._multi_count
            idx = counts.index(cur) if cur in counts else 0
            new_count = counts[(idx + 1) % len(counts)]
            probe_sys.set_multi_count(new_count)
            text_overlay.add_log(f"Multi-probe: {new_count}")

        # ---------- brain state ----------
        elif key_lower == "s" and not shift:
            # Initialize brain states
            if flow_mesh is None:
                print("[brain-state] Flow mesh required.")
                return
            region_names = []
            for k in flow_mesh.get_all_region_keys():
                name = flow_mesh.get_region_name(k)
                if hasattr(flow_mesh, '_hemispheres') and k in flow_mesh._hemispheres:
                    region_names.append(f"{name} (left)")
                    region_names.append(f"{name} (right)")
                else:
                    region_names.append(name)
            print("[brain-state] Enter global brain state (or press Enter for default):")
            print("  Example: 'someone thinking about loved ones'")

            def _init_states_async():
                try:
                    # Read from stdin (blocking)
                    import sys
                    global_state = input("  > ").strip()
                    result = brain_state_db.initialize_from_global(
                        global_state, region_names,
                        callback=lambda msg: print(f"  {msg}")
                    )
                    print(f"[brain-state] {result}")
                    text_overlay.add_log("Brain states initialized")
                except Exception as e:
                    print(f"[brain-state] Error: {e}")

            t = threading.Thread(target=_init_states_async, daemon=True)
            t.start()

        elif key_lower == "s" and shift:
            # Propagate state through probe path
            if not probe_sys.active or flow_mesh is None:
                print("[brain-state] Need active probe + flow mesh.")
                return
            if not brain_state_db.has_states():
                print("[brain-state] Initialize states first (press 's').")
                return

            # Snapshot probe data (lightweight)
            primary = probe_sys.get_primary_probe()
            if primary is None or len(primary.path) < 3:
                print("[brain-state] No probe path.")
                return
            prop_snapshot = {
                "path": primary.get_path_array().copy(),
                "field_mags": primary.get_field_mags_array().copy(),
            }

            def _propagate_async():
                try:
                    path_arr = prop_snapshot["path"]
                    field_mags = prop_snapshot["field_mags"]
                    transitions = analyze_probe_path(
                        path_arr, flow_mesh, sample_every=5,
                        field_mags=field_mags if len(field_mags) == len(path_arr) else None,
                    )
                    if not transitions:
                        print("[brain-state] No regions in path.")
                        return

                    def _hemi_name(t):
                        name = t["region_name"]
                        h = t.get("hemisphere")
                        return f"{name} ({h})" if h else name

                    source = _hemi_name(transitions[0])
                    affected = [_hemi_name(t) for t in transitions[1:]]
                    strengths = {_hemi_name(t): t.get("avg_flow_strength", 0.0) or 0.0
                                 for t in transitions}

                    print(f"[brain-state] Propagating from {source}...")
                    updates = brain_state_db.propagate_through_regions(
                        source, affected, flow_strengths=strengths,
                        callback=lambda m: print(f"  {m}")
                    )
                    if updates:
                        summary = brain_state_db.summarize_changes(updates, source)
                        story = brain_state_db.generate_flow_story(updates, source)
                        print(f"\n--- PROPAGATION SUMMARY ---")
                        print(summary)
                        print(f"\n--- INFORMATION FLOW STORY ---")
                        print(story)
                        print("--- END ---\n")
                        gpt_pending["result"] = f"{summary}\n\n{story}"
                    else:
                        print("[brain-state] No state changes.")
                except Exception as e:
                    print(f"[brain-state] Error: {e}")

            t = threading.Thread(target=_propagate_async, daemon=True)
            t.start()

    # ---------- legend ----------
    add_window_legend(ren, [
        "mindVisualizer",
        "space pause | f field | v colour | m cmap | q/w mesh | \\ hide",
        "[ ] dt | +/- speed | 1/2 lifetime | 7/8 alpha | y/u/z/x filter",
        "o OOS | j clip | a/d frac | g probe | G(shift) analyze(async)",
        "b branching | n multi(1/4/8) | s states | S propagate",
        "c clear | Esc quit",
    ], font_px=12)

    # ---------- interactor ----------
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    iren.Initialize()
    iren.AddObserver("TimerEvent", on_timer)
    iren.AddObserver("KeyPressEvent", on_keypress)
    iren.AddObserver("LeftButtonPressEvent", on_left_click)
    iren.CreateRepeatingTimer(max(1, int(1000 / max(1, args.fps))))
    win.Render()

    rag_status = "RAG" if use_rag else "direct (no RAG)"
    print("\n=== mindVisualizer ready ===")
    print(f"  Particles: {Np} | Seed: {seed_mode} | Field: {field_mode[0]}")
    print(f"  Colours: {', '.join(available_colour_modes)}")
    print(f"  Meshes: {len(flow_mesh.get_all_region_keys()) if flow_mesh else 0} regions")
    print(f"  GPT: {rag_status} | Multi-probe: {probe_sys._multi_count} | "
          f"Branching: {'ON' if probe_sys._branching_enabled else 'OFF'}")
    print(f"  Brain states: {'loaded' if brain_state_db.has_states() else 'not initialized (press s)'}")
    print(f"  Press 'g' then click to place probe. Shift+G to analyze.")
    print()
    iren.Start()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
