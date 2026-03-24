"""Flow probe system: single probe, multi-probe, and branching support.

The probe is advected by the same velocity field as the particles. Its trajectory
is recorded and can be analyzed for brain region transitions.

Features:
  - Single probe: click to place, follows mean flow
  - Multi-probe: initialize N probes in a local neighborhood
  - Branching: when MDN components are highly uncertain (50/50 split), spawn
    a ghost probe that follows the dominant alternative component
  - Live region highlighting in RED (ghost highlights more transparent)
"""

import numpy as np
import vtk


class FlowProbe:
    """A single probe that follows the flow field."""

    def __init__(self, ren: vtk.vtkRenderer, amin: np.ndarray, amax: np.ndarray,
                 color=(0.0, 1.0, 0.3), opacity=0.9, ghost=False, label=""):
        self.ren = ren
        self.amin = amin.astype(np.float32)
        self.amax = amax.astype(np.float32)
        self.active = False
        self.position = None
        self.path: list[np.ndarray] = []
        self.speeds: list[float] = []
        self.raw_field_mags: list[float] = []  # field magnitude (independent of speed scale)
        self._max_path_len = 50000
        self._mesh_overlay = None
        self._highlighted_regions: set[str] = set()
        self._highlight_actors: dict[str, vtk.vtkActor] = {}
        self._check_interval = 30
        self._step_counter = 0
        self.current_regions: set[str] = set()
        self._boundary_check = None
        self._on_region_change = None
        self.ghost = ghost
        self.label = label
        self._color = color
        self.steps_alive = 0
        self._stuck_counter = 0
        self._stuck_threshold = 150  # steps with near-zero movement before warning
        self._stuck_warned = False
        self._stuck_eps = 1e-6  # minimum displacement per step
        # Debounce: require N consecutive detections before entering, N misses before leaving
        self._debounce_enter = 3   # consecutive checks to confirm entry
        self._debounce_leave = 5   # consecutive misses to confirm exit
        self._region_hit_count: dict[str, int] = {}   # key -> consecutive hit count
        self._region_miss_count: dict[str, int] = {}  # key -> consecutive miss count
        self._nearest_max_dist_mm = 5.0  # hard limit for nearest-region snapping (mm)

        diag = float(np.linalg.norm(amax - amin))
        marker_alpha = 0.4 if ghost else 1.0
        trail_alpha = 0.35 if ghost else 0.9

        # Probe marker (sphere)
        self._sphere = vtk.vtkSphereSource()
        self._sphere.SetRadius(0.012 * diag if not ghost else 0.008 * diag)
        self._sphere.SetThetaResolution(16)
        self._sphere.SetPhiResolution(16)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self._sphere.GetOutputPort())
        self.marker_actor = vtk.vtkActor()
        self.marker_actor.SetMapper(mapper)
        self.marker_actor.GetProperty().SetColor(*color)
        self.marker_actor.GetProperty().SetOpacity(marker_alpha)
        self.marker_actor.GetProperty().LightingOff()
        self.marker_actor.VisibilityOff()
        ren.AddActor(self.marker_actor)

        # Trail line
        self._trail_points = vtk.vtkPoints()
        self._trail_cells = vtk.vtkCellArray()
        self._trail_pd = vtk.vtkPolyData()
        self._trail_pd.SetPoints(self._trail_points)
        self._trail_pd.SetLines(self._trail_cells)
        trail_mapper = vtk.vtkPolyDataMapper()
        trail_mapper.SetInputData(self._trail_pd)
        self.trail_actor = vtk.vtkActor()
        self.trail_actor.SetMapper(trail_mapper)
        self.trail_actor.GetProperty().SetColor(*color)
        self.trail_actor.GetProperty().SetLineWidth(3.0 if not ghost else 2.0)
        self.trail_actor.GetProperty().SetOpacity(trail_alpha)
        self.trail_actor.GetProperty().LightingOff()
        self.trail_actor.VisibilityOff()
        ren.AddActor(self.trail_actor)

    def set_mesh_overlay(self, mesh_overlay):
        self._mesh_overlay = mesh_overlay

    def set_boundary_check(self, fn):
        self._boundary_check = fn

    def set_on_region_change(self, fn):
        self._on_region_change = fn

    def place(self, position: np.ndarray):
        """Place probe at position and start recording."""
        self.position = np.array(position, dtype=np.float32).ravel()[:3]

        # Boundary validation
        if self._boundary_check is not None and not self._boundary_check(self.position):
            found = False
            diag = float(np.linalg.norm(self.amax - self.amin))
            for scale in [0.01, 0.02, 0.05, 0.1, 0.2]:
                for _ in range(50):
                    offset = np.random.randn(3).astype(np.float32) * scale * diag
                    candidate = np.clip(self.position + offset, self.amin, self.amax)
                    if self._boundary_check(candidate):
                        self.position = candidate
                        found = True
                        break
                if found:
                    break
            if not found:
                self.position = (self.amin + self.amax) / 2.0
                print(f"[probe{self.label}] Could not find valid position, using domain center")

        self.path = [self.position.copy()]
        self.speeds = [0.0]
        self.raw_field_mags = [0.0]
        self.active = True
        self._step_counter = 0
        self.steps_alive = 0
        self._stuck_counter = 0
        self._stuck_warned = False

        # Reset trail
        self._trail_points = vtk.vtkPoints()
        self._trail_cells = vtk.vtkCellArray()
        self._trail_points.InsertNextPoint(*self.position.tolist())
        self._trail_pd.SetPoints(self._trail_points)
        self._trail_pd.SetLines(self._trail_cells)
        self._trail_pd.Modified()

        self.marker_actor.VisibilityOn()
        self.marker_actor.SetPosition(*self.position.tolist())
        self.trail_actor.VisibilityOn()

        self._update_region_highlights()
        tag = " (ghost)" if self.ghost else ""
        print(f"[probe{self.label}{tag}] placed at "
              f"({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")

    def step(self, sampler, dt_step: float):
        """Advect one step using sampler."""
        if not self.active or self.position is None:
            return
        V = sampler.sample_vec(self.position[None, :])
        velocity = V[0]
        raw_mag = float(np.linalg.norm(velocity))

        new_pos = self.position + velocity * dt_step
        new_pos = np.clip(new_pos, self.amin, self.amax)

        # Boundary constraint
        if self._boundary_check is not None and not self._boundary_check(new_pos):
            half_pos = self.position + velocity * dt_step * 0.5
            half_pos = np.clip(half_pos, self.amin, self.amax)
            if self._boundary_check(half_pos):
                new_pos = half_pos
            else:
                return  # hit boundary

        self.position = new_pos.astype(np.float32)
        speed = float(np.linalg.norm(self.position - self.path[-1])) if self.path else 0.0

        # Stuck / weak flow detection
        if speed < self._stuck_eps and raw_mag < self._stuck_eps:
            self._stuck_counter += 1
            if self._stuck_counter >= self._stuck_threshold and not self._stuck_warned:
                self._stuck_warned = True
                tag = f" (ghost)" if self.ghost else ""
                print(f"\n[probe{self.label}{tag}] Flow is very weak or has ended here. "
                      f"The probe is stuck at ({self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}).")
                print(f"[probe{self.label}{tag}] Try placing the probe deeper in the brain "
                      f"where flow is stronger (press 'c' to clear, then 'g' + click).\n")
        else:
            self._stuck_counter = 0

        if len(self.path) < self._max_path_len:
            self.path.append(self.position.copy())
            self.speeds.append(speed)
            self.raw_field_mags.append(raw_mag)

        self.marker_actor.SetPosition(*self.position.tolist())

        # Append to trail
        idx = self._trail_points.InsertNextPoint(*self.position.tolist())
        if idx > 0:
            self._trail_cells.InsertNextCell(2)
            self._trail_cells.InsertCellPoint(idx - 1)
            self._trail_cells.InsertCellPoint(idx)
        self._trail_points.Modified()
        self._trail_cells.Modified()
        self._trail_pd.Modified()

        # Periodic region check
        self._step_counter += 1
        self.steps_alive += 1
        if self._step_counter % self._check_interval == 0:
            self._update_region_highlights()

    def _point_in_bbox(self, point, bounds):
        """Fast bounding-box containment check. bounds is (xmin,xmax,ymin,ymax,zmin,zmax)."""
        return (bounds[0] <= point[0] <= bounds[1] and
                bounds[2] <= point[1] <= bounds[3] and
                bounds[4] <= point[2] <= bounds[5])

    def _update_region_highlights(self):
        if self._mesh_overlay is None or self.position is None:
            return

        # --- Raw detection (what region key is at the probe right now?) ---
        raw_key = None
        if hasattr(self._mesh_overlay, 'get_region_at_point'):
            raw_key = self._mesh_overlay.get_region_at_point(self.position)
            if raw_key is None and hasattr(self._mesh_overlay, 'find_nearest_region'):
                raw_key = self._mesh_overlay.find_nearest_region(
                    self.position, search_radius=2,
                    max_distance_mm=self._nearest_max_dist_mm)
        else:
            for key in self._mesh_overlay.get_all_region_keys():
                if hasattr(self._mesh_overlay, 'fast_point_in_mesh'):
                    if self._mesh_overlay.fast_point_in_mesh(self.position, key):
                        raw_key = key
                        break
                elif self._mesh_overlay.point_in_mesh(self.position, key):
                    raw_key = key
                    break

        raw_detected = {raw_key} if raw_key else set()

        # --- Debounce: require consecutive detections to enter, consecutive misses to leave ---
        # Update hit/miss counters for detected key
        for key in raw_detected:
            self._region_hit_count[key] = self._region_hit_count.get(key, 0) + 1
            self._region_miss_count.pop(key, None)

        # Update miss counters for keys that were NOT detected this tick
        for key in list(self._region_hit_count.keys()):
            if key not in raw_detected:
                self._region_miss_count[key] = self._region_miss_count.get(key, 0) + 1
                self._region_hit_count[key] = 0

        # Determine stable set: regions that passed the entry threshold
        # and have not yet exceeded the leave threshold
        new_regions = set()
        for key in set(list(self._region_hit_count.keys()) +
                       list(self._highlighted_regions)):
            hits = self._region_hit_count.get(key, 0)
            misses = self._region_miss_count.get(key, 0)
            if key in self._highlighted_regions:
                # Already highlighted — keep it unless misses exceed threshold
                if misses < self._debounce_leave:
                    new_regions.add(key)
            else:
                # Not yet highlighted — add if hits exceed entry threshold
                if hits >= self._debounce_enter:
                    new_regions.add(key)

        # Clean up stale counters
        for key in list(self._region_miss_count.keys()):
            if self._region_miss_count[key] > self._debounce_leave + 2:
                self._region_miss_count.pop(key, None)
                self._region_hit_count.pop(key, None)

        # --- Apply changes (enter/leave) ---
        left = self._highlighted_regions - new_regions
        left_names = []
        for key in left:
            if key in self._highlight_actors:
                try:
                    self.ren.RemoveActor(self._highlight_actors[key])
                except Exception:
                    pass
                del self._highlight_actors[key]
            name = self._mesh_overlay.get_region_name(key)
            if hasattr(self._mesh_overlay, 'get_hemisphere_label'):
                hemi = self._mesh_overlay.get_hemisphere_label(key, self.position)
                if hemi:
                    name = f"{name} ({hemi})"
            left_names.append(name)
            tag = " [branch]" if self.ghost else ""
            print(f"[probe{self.label}] LEFT: {name}{tag}")

        entered = new_regions - self._highlighted_regions
        entered_names = []
        highlight_opacity = 0.12 if self.ghost else 0.25
        for key in entered:
            poly = None
            if hasattr(self._mesh_overlay, 'get_hemisphere_polydata'):
                poly = self._mesh_overlay.get_hemisphere_polydata(key, self.position)
            if poly is None:
                poly = self._mesh_overlay.get_polydata(key)
            if poly is not None:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.0)  # orange (all regions)
                actor.GetProperty().SetOpacity(highlight_opacity)
                actor.GetProperty().LightingOff()
                self.ren.AddActor(actor)
                self._highlight_actors[key] = actor
            name = self._mesh_overlay.get_region_name(key)
            if hasattr(self._mesh_overlay, 'get_hemisphere_label'):
                hemi = self._mesh_overlay.get_hemisphere_label(key, self.position)
                if hemi:
                    name = f"{name} ({hemi})"
            entered_names.append(name)
            tag = " [branch]" if self.ghost else ""
            # Detailed entry log with position context
            pos_detail = ""
            if self.position is not None and self._mesh_overlay is not None:
                try:
                    center = self._mesh_overlay.get_mesh_center(key)
                    bounds = self._mesh_overlay.get_mesh_bounds(key)
                    if center is not None and bounds is not None:
                        extent = [bounds[1]-bounds[0], bounds[3]-bounds[2],
                                  bounds[5]-bounds[4]]
                        char_size = sum(extent) / 3.0
                        dist = float(np.linalg.norm(self.position - center))
                        depth = max(0.0, 1.0 - min(dist / (char_size * 0.5), 1.0))
                        pos_parts = []
                        diff = self.position - center
                        if abs(diff[2]) > extent[2] * 0.15:
                            pos_parts.append("dorsal" if diff[2] > 0 else "ventral")
                        if abs(diff[0]) > extent[0] * 0.15:
                            pos_parts.append("lateral-R" if diff[0] > 0 else "lateral-L")
                        if abs(diff[1]) > extent[1] * 0.15:
                            pos_parts.append("anterior" if diff[1] > 0 else "posterior")
                        pos_str = "-".join(pos_parts) if pos_parts else "central"
                        pos_detail = f"  [{pos_str}, depth={depth:.0%}]"
                except Exception:
                    pass
            print(f"[probe{self.label}{tag}] ENTERED: {name}{pos_detail}")

        self._highlighted_regions = new_regions
        self.current_regions = new_regions

        # --- Extra parcellation subregion highlighting ---
        # Extra parcellation subregions also get orange, same as main regions.
        if (self._mesh_overlay is not None and
                hasattr(self._mesh_overlay, '_extra') and
                self._mesh_overlay._extra is not None and
                self.position is not None and new_regions):
            try:
                hier = self._mesh_overlay.get_hierarchical_regions_at_point(self.position)
                sub = hier.get("subregion")
                sub_mesh = hier.get("subregion_mesh")
                sub_key = f"_extra_{sub['label_id']}" if sub else None

                # Remove old subregion highlight if changed
                old_sub_key = getattr(self, '_current_subregion_key', None)
                if old_sub_key and old_sub_key != sub_key:
                    if old_sub_key in self._highlight_actors:
                        try:
                            self.ren.RemoveActor(self._highlight_actors[old_sub_key])
                        except Exception:
                            pass
                        del self._highlight_actors[old_sub_key]

                # Add new subregion highlight (orange, same as all regions)
                if sub_key and sub_mesh and sub_key not in self._highlight_actors:
                    display_mesh = sub_mesh
                    try:
                        bounds = [0.0] * 6
                        sub_mesh.GetBounds(bounds)
                        x_extent = bounds[1] - bounds[0]
                        if x_extent > 10.0:
                            x_mid = (bounds[0] + bounds[1]) / 2.0
                            plane = vtk.vtkPlane()
                            plane.SetOrigin(x_mid, 0, 0)
                            if self.position[0] >= x_mid:
                                plane.SetNormal(1, 0, 0)
                            else:
                                plane.SetNormal(-1, 0, 0)
                            clipper = vtk.vtkClipPolyData()
                            clipper.SetInputData(sub_mesh)
                            clipper.SetClipFunction(plane)
                            clipper.SetInsideOut(False)
                            clipper.Update()
                            clipped = clipper.GetOutput()
                            if clipped and clipped.GetNumberOfPoints() > 0:
                                display_mesh = clipped
                    except Exception:
                        pass

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(display_mesh)
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(1.0, 0.6, 0.0)  # orange
                    actor.GetProperty().SetOpacity(0.3)
                    actor.GetProperty().LightingOff()
                    self.ren.AddActor(actor)
                    self._highlight_actors[sub_key] = actor
                    tag = " [branch]" if self.ghost else ""
                    hemi_str = ""
                    if self.position is not None and self.position[0] >= 0:
                        hemi_str = " (right hemisphere)"
                    elif self.position is not None:
                        hemi_str = " (left hemisphere)"
                    print(f"[probe{self.label}{tag}] SUBREGION: {sub['name']}{hemi_str}")

                # Hide coarser main-atlas regions when we have a finer subregion
                if sub_key and sub_key in self._highlight_actors:
                    hidden = set()
                    for rkey in new_regions:
                        if rkey in self._highlight_actors and not rkey.startswith("_extra_"):
                            self._highlight_actors[rkey].VisibilityOff()
                            hidden.add(rkey)
                    self._hidden_orange_keys = hidden
                elif not sub_key:
                    for rkey in getattr(self, '_hidden_orange_keys', set()):
                        if rkey in self._highlight_actors:
                            self._highlight_actors[rkey].VisibilityOn()
                    self._hidden_orange_keys = set()

                self._current_subregion_key = sub_key
            except Exception:
                pass

        # --- Red hotspot: clip mesh near probe position ---
        # Instead of coloring an entire region red, clip the mesh surface
        # within a sphere around the probe and show that patch in red.
        self._update_hotspot()

        if self._on_region_change and (entered_names or left_names):
            self._on_region_change(entered_names, left_names,
                                   is_ghost=self.ghost, label=self.label)

    def _update_hotspot(self):
        """Color highlighted region meshes with a distance-based heatmap.

        Vertices near the probe are red, fading smoothly to orange further away.
        Uses per-vertex RGBA scalars — no clipping, no extra actors.
        """
        from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

        if self.position is None or not self._highlight_actors:
            return

        probe_pos = self.position.astype(np.float64)
        fade_radius = 15.0  # mm — distance over which red fades to orange

        for key, actor in self._highlight_actors.items():
            mapper = actor.GetMapper()
            if mapper is None:
                continue
            poly = mapper.GetInput()
            if poly is None or poly.GetNumberOfPoints() < 3:
                continue

            pts_vtk = poly.GetPoints()
            if pts_vtk is None:
                continue
            verts = vtk_to_numpy(pts_vtk.GetData()).astype(np.float64)

            # Distance from each vertex to probe
            dists = np.linalg.norm(verts - probe_pos, axis=1)
            t = np.clip(dists / fade_radius, 0.0, 1.0)  # 0=at probe, 1=far

            # Color gradient: red (1,0,0) at probe → orange (1,0.6,0) far
            r = np.full(len(t), 255, np.uint8)
            g = (t * 0.6 * 255).astype(np.uint8)
            b = np.zeros(len(t), np.uint8)

            # Alpha: brighter near probe, dimmer far away
            base_opacity = 0.45 if not self.ghost else 0.2
            far_opacity = 0.2 if not self.ghost else 0.08
            alpha_f = far_opacity + (base_opacity - far_opacity) * (1.0 - t)
            a = (np.clip(alpha_f, 0.0, 1.0) * 255).astype(np.uint8)

            rgba = np.column_stack([r, g, b, a])
            scalars = numpy_to_vtk(rgba, deep=True)
            scalars.SetNumberOfComponents(4)
            scalars.SetName("HeatmapColors")
            poly.GetPointData().SetScalars(scalars)

            mapper.SetColorModeToDirectScalars()
            mapper.SetScalarModeToUsePointData()
            mapper.ScalarVisibilityOn()
            actor.GetProperty().LightingOff()
            # Override the flat color — let scalars drive everything
            actor.GetProperty().SetOpacity(1.0)
            poly.Modified()

    def clear(self):
        self.active = False
        self.position = None
        self.path = []
        self.speeds = []
        self.raw_field_mags = []
        self._step_counter = 0
        self.steps_alive = 0
        self.marker_actor.VisibilityOff()
        self.trail_actor.VisibilityOff()
        self._trail_points = vtk.vtkPoints()
        self._trail_cells = vtk.vtkCellArray()
        self._trail_pd.SetPoints(self._trail_points)
        self._trail_pd.SetLines(self._trail_cells)
        self._trail_pd.Modified()
        for key, actor in self._highlight_actors.items():
            try:
                self.ren.RemoveActor(actor)
            except Exception:
                pass
        self._highlight_actors.clear()
        self._highlighted_regions.clear()
        self.current_regions.clear()

    def destroy(self):
        """Remove all actors from renderer."""
        self.clear()
        try:
            self.ren.RemoveActor(self.marker_actor)
        except Exception:
            pass
        try:
            self.ren.RemoveActor(self.trail_actor)
        except Exception:
            pass

    def get_path_array(self) -> np.ndarray:
        if not self.path:
            return np.zeros((0, 3), np.float32)
        return np.array(self.path, dtype=np.float32)

    def get_speeds_array(self) -> np.ndarray:
        if not self.speeds:
            return np.zeros(0, np.float32)
        return np.array(self.speeds, dtype=np.float32)

    def get_field_mags_array(self) -> np.ndarray:
        if not self.raw_field_mags:
            return np.zeros(0, np.float32)
        return np.array(self.raw_field_mags, dtype=np.float32)


class ProbeSystem:
    """Manages single/multi probes and branching behavior.

    Modes:
      - single: one probe following mean flow
      - multi: N probes in local neighborhood, all following mean flow
      - branching: when PI uncertainty is high, spawn ghost probes

    For state-propagation mode, defaults to multi(4) + branching.
    """

    def __init__(self, ren: vtk.vtkRenderer, win: vtk.vtkRenderWindow,
                 amin: np.ndarray, amax: np.ndarray):
        self.ren = ren
        self.win = win
        self.amin = amin.astype(np.float32)
        self.amax = amax.astype(np.float32)
        self.probes: list[FlowProbe] = []
        self.ghost_probes: list[FlowProbe] = []
        self._mesh_overlay = None
        self._boundary_check = None
        self._on_region_change = None
        self._branching_enabled = False
        self._multi_count = 1
        self._branch_threshold = 0.35  # max ratio between top 2 PI components
        self._branch_min_pi = 0.25     # min weight of 2nd component
        self._branch_check_interval = 50
        self._branch_step_counter = 0
        self._ghost_color = (0.4, 0.7, 1.0)  # pale blue for ghosts
        self._pi_field = None
        self._mus_samplers = None

    def set_mesh_overlay(self, overlay):
        self._mesh_overlay = overlay

    def set_boundary_check(self, fn):
        self._boundary_check = fn

    def set_on_region_change(self, fn):
        self._on_region_change = fn

    def set_branching(self, enabled: bool, pi_field=None, mus_samplers=None):
        """Enable/disable branching.

        Args:
            enabled: toggle branching
            pi_field: the PI weight grid (G,G,G,K) numpy array
            mus_samplers: list of TriLinearSampler for each component
        """
        self._branching_enabled = enabled
        self._pi_field = pi_field
        self._mus_samplers = mus_samplers
        print(f"[probe-system] branching {'ON' if enabled else 'OFF'}")

    def set_multi_count(self, n: int):
        self._multi_count = max(1, n)
        print(f"[probe-system] multi-probe count: {self._multi_count}")

    @property
    def active(self) -> bool:
        return any(p.active for p in self.probes)

    @property
    def path(self):
        """Return path of first probe (for backward compat)."""
        return self.probes[0].path if self.probes else []

    def place(self, position: np.ndarray):
        """Place probe(s) at position."""
        self.clear()
        diag = float(np.linalg.norm(self.amax - self.amin))
        jitter_scale = 0.04 * diag

        for i in range(self._multi_count):
            if i == 0:
                pos = position.copy()
                label = "" if self._multi_count == 1 else f"#{i+1}"
            else:
                offset = np.random.randn(3).astype(np.float32) * jitter_scale
                pos = np.clip(position + offset, self.amin, self.amax)
                label = f"#{i+1}"

            probe = FlowProbe(self.ren, self.amin, self.amax,
                              color=(0.0, 1.0, 0.3), ghost=False, label=label)
            if self._mesh_overlay:
                probe.set_mesh_overlay(self._mesh_overlay)
            if self._boundary_check:
                probe.set_boundary_check(self._boundary_check)
            if self._on_region_change:
                probe.set_on_region_change(self._on_region_change)
            probe.place(pos)
            self.probes.append(probe)

    def step(self, sampler, dt_step: float, pi_sampler=None):
        """Step all probes."""
        for p in self.probes:
            if p.active:
                p.step(sampler, dt_step)

        for g in self.ghost_probes:
            if g.active:
                # Ghost probes follow their specific component sampler
                comp_sampler = getattr(g, '_comp_sampler', sampler)
                g.step(comp_sampler, dt_step)

        # Prune ghost probes that have been alive > 200 steps but did not diverge
        diag = float(np.linalg.norm(self.amax - self.amin))
        prune_dist = 0.03 * diag
        to_remove = []
        for g in self.ghost_probes:
            if not g.active or g.position is None:
                continue
            if g.steps_alive > 200:
                for p in self.probes:
                    if not p.active or p.position is None:
                        continue
                    dist = float(np.linalg.norm(g.position - p.position))
                    if dist < prune_dist:
                        print("[branch] pruned ghost - did not diverge")
                        to_remove.append(g)
                        break
        for g in to_remove:
            g.destroy()
            self.ghost_probes.remove(g)

        # Check for branching
        if self._branching_enabled:
            self._branch_step_counter += 1
            if self._branch_step_counter % self._branch_check_interval == 0:
                self._check_branching(sampler, dt_step)

    def _check_branching(self, mean_sampler, dt_step):
        """Check if any active probe is at a high-uncertainty location and branch."""
        if self._pi_field is None or self._mus_samplers is None:
            return
        if not self.probes:
            return

        from .field_loader import TriLinearSampler

        for p in self.probes:
            if not p.active or p.position is None:
                continue
            # Sample PI weights at probe position
            pos = p.position[None, :]
            pi_vals = mean_sampler.sample_vec(pos)  # dummy, we need PI
            # Actually sample PI field directly
            try:
                K = self._pi_field.shape[-1]
                weights = np.zeros(K, np.float32)
                for k in range(K):
                    pi_k = self._pi_field[..., k:k+1]
                    # Use sampler to interpolate
                    w = mean_sampler.sample_scalar(
                        self._pi_field[..., k],
                        p.position[None, :]
                    )
                    weights[k] = float(w[0])
            except Exception:
                continue

            # Normalize
            ws = weights.sum()
            if ws <= 0:
                continue
            weights /= ws

            # Sort to find top 2
            sorted_idx = np.argsort(weights)[::-1]
            w1 = weights[sorted_idx[0]]
            w2 = weights[sorted_idx[1]] if len(sorted_idx) > 1 else 0.0

            # Check if uncertain enough
            if w2 < self._branch_min_pi:
                continue
            ratio = w2 / max(w1, 1e-9)
            if ratio < self._branch_threshold:
                continue

            # Check we haven't already branched from near this position
            already_branched = False
            for g in self.ghost_probes:
                if g.active and g.path:
                    d = np.linalg.norm(g.path[0] - p.position)
                    if d < 0.02 * float(np.linalg.norm(self.amax - self.amin)):
                        already_branched = True
                        break
            if already_branched:
                continue

            # Branch! Create ghost probe following 2nd component
            comp_idx = int(sorted_idx[1])
            if comp_idx >= len(self._mus_samplers):
                continue

            ghost = FlowProbe(self.ren, self.amin, self.amax,
                              color=self._ghost_color, ghost=True,
                              label=f"~branch(comp{comp_idx+1})")
            if self._mesh_overlay:
                ghost.set_mesh_overlay(self._mesh_overlay)
            if self._boundary_check:
                ghost.set_boundary_check(self._boundary_check)
            if self._on_region_change:
                ghost.set_on_region_change(self._on_region_change)
            ghost._comp_sampler = self._mus_samplers[comp_idx]
            ghost.place(p.position.copy())
            self.ghost_probes.append(ghost)

            print(f"[branch] SPLIT at ({p.position[0]:.2f}, {p.position[1]:.2f}, "
                  f"{p.position[2]:.2f}): comp{sorted_idx[0]+1}={w1:.2f} vs "
                  f"comp{comp_idx+1}={w2:.2f}")
            if self._on_region_change:
                self._on_region_change(
                    [f"BRANCH: comp{comp_idx+1} (weight={w2:.2f})"], [],
                    is_ghost=True, label="branch"
                )

    def clear(self):
        """Clear all probes."""
        for p in self.probes:
            p.destroy()
        for g in self.ghost_probes:
            g.destroy()
        self.probes.clear()
        self.ghost_probes.clear()
        self._branch_step_counter = 0

    def freeze(self):
        """Stop all probes from moving (keep path and highlights intact)."""
        for p in self.probes:
            p.active = False
        for g in self.ghost_probes:
            g.active = False
        print("[probe-system] probes frozen")

    def get_all_probes(self) -> list[FlowProbe]:
        """Return all probes (main + ghost) for analysis."""
        return self.probes + self.ghost_probes

    def get_primary_probe(self) -> FlowProbe | None:
        """Return the first active probe."""
        for p in self.probes:
            if p.active:
                return p
        return None

    def get_all_paths(self) -> list[tuple[np.ndarray, bool, str]]:
        """Return [(path_array, is_ghost, label), ...] for all probes."""
        result = []
        for p in self.probes:
            if p.path:
                result.append((p.get_path_array(), False, p.label))
        for g in self.ghost_probes:
            if g.path:
                result.append((g.get_path_array(), True, g.label))
        return result
