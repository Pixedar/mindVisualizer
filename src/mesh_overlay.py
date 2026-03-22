"""Brain-region mesh overlay for VTK renderers.

Loads OBJ meshes from data/meshes_obj/, applies um->mm scaling + alignment matrix,
and provides cycling/toggling/opacity controls.  Loads anatomical names from Allen
Human Brain Atlas structures.json.

Hemisphere detection: symmetric meshes (two connected components split along X)
are stored as "{key}_left" and "{key}_right" for per-hemisphere queries.

Region lookup: a 3D voxel label grid is pre-computed at startup so that
point-in-region queries are O(1) array lookups — fast and reliable regardless
of mesh quality (holes, non-watertight surfaces, etc.).
"""

import json
from pathlib import Path

import numpy as np
import vtk

DEFAULT_MESH_SCALE = 0.001  # um -> mm
LABEL_GRID_RESOLUTION = 100  # voxels per axis for region lookup grid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_alignment(path: Path):
    """Load 4x4 alignment matrix from JSON."""
    try:
        M = json.loads(Path(path).read_text())["matrix"]
        mx = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                mx.SetElement(i, j, float(M[i][j]))
        return mx
    except Exception:
        print("[mesh] no/invalid alignment json; showing raw OBJ coords")
        return None


def _load_structures(data_dir: Path) -> dict[str, str]:
    """Load region ID -> name mapping from Allen atlas structures.json."""
    mapping: dict[str, str] = {}
    structs_path = data_dir / "structures.json"
    if structs_path.exists():
        try:
            structs = json.loads(structs_path.read_text(encoding="utf-8"))
            for s in structs:
                sid = str(s["id"])
                name = s.get("name", sid)
                acronym = s.get("acronym", "")
                if acronym and acronym != "root":
                    mapping[sid] = f"{name} ({acronym})"
                else:
                    mapping[sid] = name
        except Exception as e:
            print(f"[mesh] warning: could not parse structures.json: {e}")
    mapping.setdefault("brain_outline", "Whole Brain Outline")
    return mapping


def _polydata_center_x(poly: vtk.vtkPolyData) -> float:
    """Return the X coordinate of the center of mass of *poly*."""
    com = vtk.vtkCenterOfMass()
    com.SetInputData(poly)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    return com.GetCenter()[0]


def _polydata_bounds(poly: vtk.vtkPolyData) -> tuple[float, ...]:
    """Return (xmin, xmax, ymin, ymax, zmin, zmax)."""
    return poly.GetBounds()


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class FlowMeshOverlay:
    """Brain-region mesh overlay for the main flow renderer.

    Loads OBJ meshes from ``mesh_dir/../meshes_obj``, applies alignment
    matrix (um->mm scale + JSON alignment), and optionally splits symmetric
    meshes into left/right hemispheres.
    """

    def __init__(self, ren: vtk.vtkRenderer, win: vtk.vtkRenderWindow,
                 mesh_dir: Path, alignment_file: Path,
                 data_dir: Path | None = None):
        self.ren = ren
        self.win = win
        self.dir = Path(mesh_dir)
        self._actors: dict[str, vtk.vtkActor] = {}
        self._polydata: dict[str, vtk.vtkPolyData] = {}
        self._visible: str | None = None
        self._opacity = 0.35
        self._align = _load_alignment(alignment_file)

        # Pre-computed bounding boxes: key -> (xmin, xmax, ymin, ymax, zmin, zmax)
        self._bboxes: dict[str, tuple[float, ...]] = {}

        # Hemisphere splits: key -> {"left": vtkPolyData, "right": vtkPolyData}
        self._hemispheres: dict[str, dict[str, vtk.vtkPolyData]] = {}

        # Load region names from atlas
        if data_dir is None:
            data_dir = self.dir.parent
        self._names = _load_structures(data_dir)

        # ----- discover OBJ meshes only -----
        obj_dir = self.dir.parent / "meshes_obj"
        self.keys: dict[str, Path] = {}
        if obj_dir.exists():
            for obj in sorted(obj_dir.glob("*.obj")):
                self.keys[obj.stem] = obj
        else:
            print(f"[mesh] warning: OBJ directory not found: {obj_dir}")

        # Skip overly broad structural IDs
        _skip = {"10153", "10154", "10155", "10156", "10157", "10158",
                 "10557", "10668"}
        self._order = [k for k in self.keys
                       if k != "brain_outline" and k not in _skip]

        # Compute bounding-box volumes and filter oversized meshes
        self._bb_volumes: dict[str, float] = {}
        self._compute_volumes_and_filter()

        # Optional extra parcellation (set via set_extra_parcellation)
        self._extra = None

        print(f"[mesh] discovered {len(self._order)} region meshes "
              f"({len(self.keys)} OBJ)")

    # ------------------------------------------------------------------
    # extra parcellation
    # ------------------------------------------------------------------

    def set_extra_parcellation(self, extra):
        """Attach an optional ExtraParcellation for finer subregion queries.

        Args:
            extra: ExtraParcellation instance (already loaded)
        """
        self._extra = extra

    def get_hierarchical_regions_at_point(self, point: np.ndarray) -> dict:
        """Return both primary (Allen) and subregion (extra parcellation) at a point.

        Returns:
            {
                "primary": str | None,         # Allen atlas region key
                "primary_name": str | None,    # Allen atlas region name
                "subregion": dict | None,      # {"label_id", "name", "volume_mm3"} from extra
                "subregion_mesh": vtkPolyData | None  # mesh if available
            }
        """
        result = {
            "primary": None,
            "primary_name": None,
            "subregion": None,
            "subregion_mesh": None,
        }

        # Primary lookup (Allen atlas)
        primary_key = self.get_region_at_point(point)
        if primary_key is None:
            primary_key = self.find_nearest_region(point, search_radius=2,
                                                    max_distance_mm=5.0)
        if primary_key:
            result["primary"] = primary_key
            result["primary_name"] = self.get_region_name(primary_key)

        # Extra parcellation lookup
        if self._extra is not None and self._extra.is_loaded():
            sub = self._extra.get_region_at_point(point)
            if sub is None:
                sub = self._extra.get_nearby_region(point, search_radius=2)
            if sub is not None:
                result["subregion"] = sub
                # Try to get mesh for highlighting
                mesh = self._extra.get_region_mesh(sub["label_id"])
                if mesh is not None:
                    result["subregion_mesh"] = mesh

        return result

    # ------------------------------------------------------------------
    # volume filtering
    # ------------------------------------------------------------------

    def _compute_volumes_and_filter(self):
        """Compute bounding-box volumes and exclude overly large meshes."""
        vols: dict[str, float] = {}
        for key in list(self._order):
            poly = self.get_polydata(key)
            if poly is not None and poly.GetNumberOfPoints() > 0:
                b = poly.GetBounds()
                vol = (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])
                vols[key] = vol
        self._bb_volumes = vols

        if not vols:
            return

        max_vol = max(vols.values())
        threshold = max_vol * 0.25
        oversized = [k for k, v in vols.items() if v > threshold]
        if oversized:
            names = [self.get_region_name(k) for k in oversized]
            print(f"[mesh] excluding {len(oversized)} oversized regions: "
                  f"{', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
            self._order = [k for k in self._order if k not in oversized]

    # ------------------------------------------------------------------
    # voxel label grid (pre-computed for fast point-in-region queries)
    # ------------------------------------------------------------------

    def build_label_grid(self):
        """Pre-compute a 3D voxel grid mapping each voxel to a region key.

        After this call, point_in_region() / get_region_at_point() use fast
        O(1) array lookups instead of expensive vtkSelectEnclosedPoints.
        """
        # Determine brain bounds from brain outline or all meshes
        brain_poly = self.get_polydata("10155")  # brain outline
        if brain_poly is not None:
            bb = brain_poly.GetBounds()
        else:
            # Fallback: union of all mesh bounding boxes
            all_bounds = [self.get_polydata(k).GetBounds()
                          for k in self._order if self.get_polydata(k) is not None]
            if not all_bounds:
                print("[mesh] no meshes loaded, cannot build label grid")
                return
            bb = (min(b[0] for b in all_bounds), max(b[1] for b in all_bounds),
                  min(b[2] for b in all_bounds), max(b[3] for b in all_bounds),
                  min(b[4] for b in all_bounds), max(b[5] for b in all_bounds))

        margin = 2.0
        self._grid_origin = np.array([bb[0] - margin, bb[2] - margin,
                                       bb[4] - margin], dtype=np.float32)
        self._grid_max = np.array([bb[1] + margin, bb[3] + margin,
                                    bb[5] + margin], dtype=np.float32)
        N = LABEL_GRID_RESOLUTION
        self._grid_spacing = (self._grid_max - self._grid_origin) / N
        self._grid_N = N

        # key_index: 0 = no region, 1..len = region keys
        self._grid_keys = [""]  # index 0 = empty
        key_to_idx = {}
        for ki, key in enumerate(self._order):
            self._grid_keys.append(key)
            key_to_idx[key] = ki + 1

        # Initialize grid to 0 (no region)
        self._label_grid = np.zeros((N, N, N), dtype=np.int16)

        # Build all voxel centers
        print(f"[mesh] building {N}^3 label grid for {len(self._order)} regions...")
        total = N * N * N
        all_pts = vtk.vtkPoints()
        all_pts.SetNumberOfPoints(total)
        idx = 0
        for iz in range(N):
            z = self._grid_origin[2] + (iz + 0.5) * self._grid_spacing[2]
            for iy in range(N):
                y = self._grid_origin[1] + (iy + 0.5) * self._grid_spacing[1]
                for ix in range(N):
                    x = self._grid_origin[0] + (ix + 0.5) * self._grid_spacing[0]
                    all_pts.SetPoint(idx, x, y, z)
                    idx += 1

        all_pd = vtk.vtkPolyData()
        all_pd.SetPoints(all_pts)

        # For each mesh, batch test all voxels
        # Process smallest meshes first (they are most specific / finest granularity)
        ordered_keys = sorted(self._order,
                              key=lambda k: self._bb_volumes.get(k, 0))

        for mi, key in enumerate(ordered_keys):
            poly = self.get_polydata(key)
            if poly is None or poly.GetNumberOfCells() < 10:
                continue

            mesh_bb = poly.GetBounds()

            # Index range that overlaps this mesh bbox
            ix_lo = max(0, int((mesh_bb[0] - self._grid_origin[0]) /
                               self._grid_spacing[0] - 1))
            ix_hi = min(N, int((mesh_bb[1] - self._grid_origin[0]) /
                               self._grid_spacing[0] + 2))
            iy_lo = max(0, int((mesh_bb[2] - self._grid_origin[1]) /
                               self._grid_spacing[1] - 1))
            iy_hi = min(N, int((mesh_bb[3] - self._grid_origin[1]) /
                               self._grid_spacing[1] + 2))
            iz_lo = max(0, int((mesh_bb[4] - self._grid_origin[2]) /
                               self._grid_spacing[2] - 1))
            iz_hi = min(N, int((mesh_bb[5] - self._grid_origin[2]) /
                               self._grid_spacing[2] + 2))

            # Collect candidate flat indices
            candidates = []
            for iz in range(iz_lo, iz_hi):
                for iy in range(iy_lo, iy_hi):
                    for ix in range(ix_lo, ix_hi):
                        candidates.append((ix, iy, iz,
                                           iz * N * N + iy * N + ix))

            if not candidates:
                continue

            # Build test polydata with only candidate points
            test_pts = vtk.vtkPoints()
            test_pts.SetNumberOfPoints(len(candidates))
            for ci, (_, _, _, flat) in enumerate(candidates):
                test_pts.SetPoint(ci, all_pts.GetPoint(flat))
            test_pd = vtk.vtkPolyData()
            test_pd.SetPoints(test_pts)

            enc = vtk.vtkSelectEnclosedPoints()
            enc.SetSurfaceData(poly)
            enc.SetTolerance(0.001)
            enc.SetInputData(test_pd)
            enc.Update()

            region_idx = key_to_idx[key]
            count = 0
            for ci, (ix, iy, iz, _) in enumerate(candidates):
                if enc.IsInside(ci):
                    self._label_grid[ix, iy, iz] = region_idx
                    count += 1

            enc.Complete()

            if (mi + 1) % 20 == 0 or count > 0:
                name = self.get_region_name(key)
                filled = int(np.count_nonzero(self._label_grid))
                print(f"  [{mi+1}/{len(ordered_keys)}] {name}: "
                      f"{count} voxels, total filled: {filled}")

        total_filled = int(np.count_nonzero(self._label_grid))
        print(f"[mesh] label grid complete: {total_filled}/{total} voxels "
              f"filled ({total_filled/total*100:.1f}%)")
        self._label_grid_ready = True

    def save_label_grid(self, path: Path):
        """Save the pre-computed label grid to disk for fast startup."""
        if not getattr(self, '_label_grid_ready', False):
            print("[mesh] no label grid to save")
            return
        path = Path(path)
        data = {
            "grid": self._label_grid,
            "origin": self._grid_origin,
            "max": self._grid_max,
            "spacing": self._grid_spacing,
            "N": self._grid_N,
            "keys": self._grid_keys,
        }
        np.savez_compressed(str(path), **{
            "grid": self._label_grid,
            "origin": self._grid_origin,
            "grid_max": self._grid_max,
            "spacing": self._grid_spacing,
        })
        # Save keys separately as JSON (string list)
        keys_path = path.with_suffix('.keys.json')
        keys_path.write_text(json.dumps(self._grid_keys), encoding='utf-8')
        print(f"[mesh] saved label grid to {path}")

    def load_label_grid(self, path: Path) -> bool:
        """Load a pre-computed label grid from disk.

        Returns True if loaded successfully, False otherwise.
        """
        path = Path(path)
        keys_path = path.with_suffix('.keys.json')
        if not path.exists() or not keys_path.exists():
            return False
        try:
            d = np.load(str(path))
            self._label_grid = d["grid"]
            self._grid_origin = d["origin"].astype(np.float32)
            self._grid_max = d["grid_max"].astype(np.float32)
            self._grid_spacing = d["spacing"].astype(np.float32)
            self._grid_N = int(self._label_grid.shape[0])
            self._grid_keys = json.loads(keys_path.read_text(encoding='utf-8'))

            # Validate keys match loaded meshes
            loaded_mesh_keys = set(self._order)
            grid_keys_set = set(self._grid_keys[1:])  # skip index 0 (empty)
            if not grid_keys_set.issubset(loaded_mesh_keys):
                missing = grid_keys_set - loaded_mesh_keys
                print(f"[mesh] cached grid has {len(missing)} keys not in current meshes, rebuilding...")
                return False

            total_filled = int(np.count_nonzero(self._label_grid))
            total = self._grid_N ** 3
            print(f"[mesh] loaded cached label grid: {total_filled}/{total} voxels "
                  f"filled ({total_filled/total*100:.1f}%)")
            self._label_grid_ready = True
            return True
        except Exception as e:
            print(f"[mesh] failed to load cached grid: {e}")
            return False

    def get_region_at_point(self, point: np.ndarray) -> str | None:
        """Return the region key at a 3D point using the label grid.

        Returns None if no region, or the key string (e.g. "12114").
        """
        if not getattr(self, '_label_grid_ready', False):
            return None
        p = np.asarray(point, dtype=np.float32).ravel()[:3]
        idx = ((p - self._grid_origin) / self._grid_spacing).astype(int)
        N = self._grid_N
        if np.any(idx < 0) or np.any(idx >= N):
            return None
        val = self._label_grid[idx[0], idx[1], idx[2]]
        if val == 0:
            return None
        return self._grid_keys[val]

    def get_all_regions_at_point(self, point: np.ndarray) -> list[str]:
        """Return all region keys near a 3D point.

        Checks the point's voxel and its 26 neighbors to catch points near
        region boundaries.
        """
        if not getattr(self, '_label_grid_ready', False):
            return []
        p = np.asarray(point, dtype=np.float32).ravel()[:3]
        idx = ((p - self._grid_origin) / self._grid_spacing).astype(int)
        N = self._grid_N
        if np.any(idx < 0) or np.any(idx >= N):
            return []

        found = set()
        # Check center voxel first
        val = self._label_grid[idx[0], idx[1], idx[2]]
        if val > 0:
            found.add(self._grid_keys[val])

        return list(found)

    def find_nearest_region(self, point: np.ndarray,
                            search_radius: int = 3,
                            max_distance_mm: float = 0.0) -> str | None:
        """Find nearest region within search_radius voxels of point.

        Useful when a point falls between regions (in a tiny gap).

        Args:
            point: 3D position in mm.
            search_radius: max voxels to search outward.
            max_distance_mm: if > 0, reject matches farther than this (in mm).
                             Distance is measured to the nearest labelled voxel
                             center, which approximates distance to surface for
                             the grid resolution we use.
        """
        if not getattr(self, '_label_grid_ready', False):
            return None
        p = np.asarray(point, dtype=np.float32).ravel()[:3]
        idx = ((p - self._grid_origin) / self._grid_spacing).astype(int)
        N = self._grid_N

        # Spiral outward from center
        for r in range(0, search_radius + 1):
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != r:
                            continue  # only check shell at distance r
                        ix = idx[0] + dx
                        iy = idx[1] + dy
                        iz = idx[2] + dz
                        if 0 <= ix < N and 0 <= iy < N and 0 <= iz < N:
                            val = self._label_grid[ix, iy, iz]
                            if val > 0:
                                if max_distance_mm > 0:
                                    voxel_center = self._grid_origin + \
                                        (np.array([ix, iy, iz]) + 0.5) * self._grid_spacing
                                    dist = float(np.linalg.norm(p - voxel_center))
                                    if dist > max_distance_mm:
                                        continue
                                return self._grid_keys[val]
        return None

    # ------------------------------------------------------------------
    # mesh loading
    # ------------------------------------------------------------------

    def _load_mesh(self, filepath: Path) -> vtk.vtkPolyData | None:
        """Load an OBJ mesh file."""
        if filepath.suffix.lower() != ".obj":
            return None
        reader = vtk.vtkOBJReader()
        reader.SetFileName(str(filepath))
        reader.Update()
        return reader.GetOutput()

    def _transform_polydata(self, poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        """Apply um->mm scaling and alignment matrix to *poly*."""
        tf_scale = vtk.vtkTransform()
        tf_scale.Scale(DEFAULT_MESH_SCALE, DEFAULT_MESH_SCALE, DEFAULT_MESH_SCALE)
        f1 = vtk.vtkTransformPolyDataFilter()
        f1.SetInputData(poly)
        f1.SetTransform(tf_scale)
        f1.Update()
        out = f1.GetOutput()

        if self._align is not None:
            tf = vtk.vtkTransform()
            tf.SetMatrix(self._align)
            f2 = vtk.vtkTransformPolyDataFilter()
            f2.SetInputData(out)
            f2.SetTransform(tf)
            f2.Update()
            out = f2.GetOutput()

        return out

    # ------------------------------------------------------------------
    # hemisphere detection
    # ------------------------------------------------------------------

    def _detect_hemispheres(self, key: str, poly: vtk.vtkPolyData):
        """If *poly* has exactly two connected regions split along X, store
        them as left / right hemispheres for *key*."""
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputData(poly)
        conn.SetExtractionModeToAllRegions()
        conn.ColorRegionsOn()
        conn.Update()

        n_regions = conn.GetNumberOfExtractedRegions()
        if n_regions != 2:
            return

        # Extract each region separately
        regions: list[vtk.vtkPolyData] = []
        for rid in range(2):
            extract = vtk.vtkConnectivityFilter()
            extract.SetInputData(poly)
            extract.SetExtractionModeToSpecifiedRegions()
            extract.AddSpecifiedRegion(rid)
            extract.Update()
            regions.append(extract.GetOutput())

        cx0 = _polydata_center_x(regions[0])
        cx1 = _polydata_center_x(regions[1])

        # Overall midpoint along X from the full mesh bounds
        bounds = poly.GetBounds()
        mid_x = (bounds[0] + bounds[1]) / 2.0

        # Check that the two regions sit on opposite sides of the midpoint
        if (cx0 < mid_x and cx1 > mid_x):
            self._hemispheres[key] = {"left": regions[0], "right": regions[1]}
        elif (cx1 < mid_x and cx0 > mid_x):
            self._hemispheres[key] = {"left": regions[1], "right": regions[0]}
        # else: not a clean left/right split -- skip

    # ------------------------------------------------------------------
    # actor creation
    # ------------------------------------------------------------------

    def _actor_for(self, key: str) -> vtk.vtkActor | None:
        if key in self._actors:
            return self._actors[key]
        if key not in self.keys:
            return None
        filepath = self.keys[key]
        if not filepath.exists():
            print(f"[flow mesh] missing file: {filepath}")
            return None

        poly = self._load_mesh(filepath)
        if poly is None or poly.GetNumberOfPoints() == 0:
            return None

        out = self._transform_polydata(poly)

        # Cache bounding box
        self._bboxes[key] = _polydata_bounds(out)

        # Hemisphere detection
        self._detect_hemispheres(key, out)

        # Build actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(out)
        act = vtk.vtkActor()
        act.SetMapper(mapper)
        if key == "brain_outline":
            act.GetProperty().SetColor(0.85, 0.85, 0.85)
            act.GetProperty().SetOpacity(max(0.15, self._opacity * 0.7))
        else:
            act.GetProperty().SetColor(0.92, 0.92, 0.92)
            act.GetProperty().SetOpacity(self._opacity)
        act.GetProperty().LightingOff()

        self._actors[key] = act
        self._polydata[key] = out
        return act

    # ------------------------------------------------------------------
    # public: polydata / region info
    # ------------------------------------------------------------------

    def get_polydata(self, key: str) -> vtk.vtkPolyData | None:
        if key not in self._polydata:
            self._actor_for(key)
        return self._polydata.get(key)

    def get_region_name(self, key: str) -> str:
        return self._names.get(key, f"Region {key}")

    def get_all_region_keys(self) -> list[str]:
        return list(self._order)

    def get_visible_key(self) -> str | None:
        return self._visible

    # ------------------------------------------------------------------
    # public: hemisphere queries
    # ------------------------------------------------------------------

    def get_hemisphere_polydata(self, key: str,
                                point: np.ndarray) -> vtk.vtkPolyData | None:
        """Return the hemisphere polydata (left or right) that contains *point*.

        If the mesh was cleanly split into 2 connected components, returns the
        matching half. Otherwise, if the mesh spans both hemispheres (wide X
        extent), clips it to the probe's side using a plane at the midpoint.
        """
        # Ensure mesh is loaded (triggers hemisphere detection)
        full = self.get_polydata(key)
        hemi = self._hemispheres.get(key)
        if hemi is not None:
            label = self._classify_hemisphere(key, point)
            return hemi.get(label, full)

        # Fallback: clip by X-plane if mesh spans both hemispheres
        if full is None:
            return full
        bounds = self._bboxes.get(key)
        if bounds is None:
            return full
        x_extent = bounds[1] - bounds[0]
        y_extent = bounds[3] - bounds[2]
        # Only clip if X extent is large relative to Y (indicates bilateral mesh)
        if x_extent > y_extent * 0.6:
            mid_x = (bounds[0] + bounds[1]) / 2.0
            plane = vtk.vtkPlane()
            plane.SetOrigin(mid_x, 0, 0)
            # Keep the side where the probe is:
            # vtkClipPolyData keeps points where implicit function >= 0 (default)
            # Plane function: dot(point - origin, normal)
            # If probe is at x < mid_x, we want to keep x < mid_x
            #   normal = (-1,0,0): function = -(x - mid_x) = mid_x - x, positive for x < mid_x => kept
            if float(point[0]) < mid_x:
                plane.SetNormal(-1, 0, 0)  # keep x < mid (probe's side)
            else:
                plane.SetNormal(1, 0, 0)   # keep x > mid (probe's side)
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(full)
            clipper.SetClipFunction(plane)
            clipper.Update()
            clipped = clipper.GetOutput()
            if clipped.GetNumberOfCells() > 0:
                return clipped

        return full

    def get_hemisphere_label(self, key: str, point: np.ndarray) -> str | None:
        """Return ``"left"`` or ``"right"`` for *point* relative to *key*.

        Returns ``None`` if the mesh has no hemisphere split.
        """
        self.get_polydata(key)  # ensure loaded
        if key not in self._hemispheres:
            return None
        return self._classify_hemisphere(key, point)

    def _classify_hemisphere(self, key: str, point: np.ndarray) -> str:
        """Decide whether *point* falls in the left or right hemisphere of
        *key* by comparing its X coordinate to the mesh midpoint."""
        bounds = self._bboxes.get(key)
        if bounds is None:
            bounds = _polydata_bounds(self._polydata[key])
        mid_x = (bounds[0] + bounds[1]) / 2.0
        # After alignment transform, determine which side is anatomical left/right
        # by checking probe X relative to mesh midpoint
        return "left" if float(point[0]) < mid_x else "right"

    # ------------------------------------------------------------------
    # public: point-in-mesh with fast rejection
    # ------------------------------------------------------------------

    def fast_point_in_mesh(self, point: np.ndarray, key: str) -> bool:
        """Check if *point* is inside the mesh identified by *key*.

        Performs a cheap bounding-box test first; only runs
        ``vtkSelectEnclosedPoints`` if the point falls within the box.
        """
        poly = self.get_polydata(key)
        if poly is None or poly.GetNumberOfCells() == 0:
            return False

        # --- fast bounding-box rejection ---
        bb = self._bboxes.get(key)
        if bb is None:
            bb = _polydata_bounds(poly)
            self._bboxes[key] = bb
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        if (px < bb[0] or px > bb[1] or
                py < bb[2] or py > bb[3] or
                pz < bb[4] or pz > bb[5]):
            return False

        # --- expensive enclosed-point test ---
        return self._enclosed_test(poly, point)

    def point_in_mesh(self, point: np.ndarray, key: str) -> bool:
        """Check if a 3D point is inside a mesh (no fast rejection)."""
        poly = self.get_polydata(key)
        if poly is None or poly.GetNumberOfCells() == 0:
            return False
        return self._enclosed_test(poly, point)

    @staticmethod
    def _enclosed_test(poly: vtk.vtkPolyData, point: np.ndarray) -> bool:
        pts = vtk.vtkPoints()
        pts.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))
        test_pd = vtk.vtkPolyData()
        test_pd.SetPoints(pts)

        enc = vtk.vtkSelectEnclosedPoints()
        enc.SetInputData(test_pd)
        enc.SetSurfaceData(poly)
        enc.SetTolerance(0.0001)
        enc.Update()
        return bool(enc.IsInside(0))

    # ------------------------------------------------------------------
    # public: display
    # ------------------------------------------------------------------

    def show(self, key: str):
        if key == self._visible:
            return
        self.hide()
        act = self._actor_for(key)
        if act is not None:
            self.ren.AddActor(act)
            self._visible = key
            name = self.get_region_name(key)
            print(f"[flow mesh] showing: {name} ({key})")
            if self.win is not None:
                self.win.Render()

    def hide(self):
        if self._visible and self._visible in self._actors:
            try:
                self.ren.RemoveActor(self._actors[self._visible])
            except Exception:
                pass
        self._visible = None
        if self.win is not None:
            self.win.Render()

    def cycle(self, step: int = +1):
        if not self._order:
            return
        if self._visible not in self._order:
            idx = 0 if step >= 0 else len(self._order) - 1
        else:
            idx = (self._order.index(self._visible) + step) % len(self._order)
        self.show(self._order[idx])

    def set_opacity(self, mul: float):
        self._opacity = float(np.clip(self._opacity * float(mul), 0.05, 1.0))
        for k, a in self._actors.items():
            if k == "brain_outline":
                a.GetProperty().SetOpacity(max(0.1, self._opacity * 0.7))
            else:
                a.GetProperty().SetOpacity(self._opacity)
        if self.win is not None:
            self.win.Render()

    # ------------------------------------------------------------------
    # public: geometry queries
    # ------------------------------------------------------------------

    def get_mesh_center(self, key: str) -> np.ndarray | None:
        poly = self.get_polydata(key)
        if poly is None:
            return None
        com = vtk.vtkCenterOfMass()
        com.SetInputData(poly)
        com.SetUseScalarsAsWeights(False)
        com.Update()
        return np.array(com.GetCenter(), dtype=np.float32)

    def get_mesh_bounds(self, key: str) -> np.ndarray | None:
        poly = self.get_polydata(key)
        if poly is None:
            return None
        return np.array(poly.GetBounds(), dtype=np.float32)

    def get_bb_volume(self, key: str) -> float:
        """Return cached bounding-box volume (or inf if unknown)."""
        return self._bb_volumes.get(key, float('inf'))
