"""Optional extra brain parcellation support.

Loads volumetric NIfTI parcellation atlases (e.g., Brainnetome, AAL3, Schaefer)
and provides point-in-region queries + on-demand mesh generation via marching cubes.

Dependencies: nibabel, scikit-image (optional — install with:
    pip install nibabel scikit-image
)

Usage:
    from src.extra_parcellation import ExtraParcellation

    extra = ExtraParcellation(
        nifti_path="data/extra_parcellation/BN_Atlas_246_1mm.nii.gz",
        label_map_path="data/extra_parcellation/BN_Atlas_labels.json"  # optional
    )
    extra.load()

    # Query a point in MNI space
    result = extra.get_region_at_point(np.array([-30.0, 15.0, 50.0]))
    # -> {"label_id": 42, "name": "SFG_R_7_2", "volume_mm3": 1523.0}

    # Get VTK mesh for visualization
    polydata = extra.get_region_mesh(42)
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


class ExtraParcellation:
    """Volumetric NIfTI parcellation atlas for additional region granularity.

    Designed to work alongside the primary Allen atlas meshes. When both
    parcellations cover a point, the caller (FlowMeshOverlay) can decide
    which to use (typically the smaller/more specific one).
    """

    def __init__(self, nifti_path: Path | str,
                 label_map_path: Path | str | None = None):
        """
        Args:
            nifti_path: Path to NIfTI volume (.nii or .nii.gz) with integer labels.
            label_map_path: Optional JSON mapping label IDs to human-readable names.
                            Format: {"1": "Region A", "2": "Region B", ...}
                            If omitted, labels will be "Region_<id>".
        """
        self.nifti_path = Path(nifti_path)
        self.label_map_path = Path(label_map_path) if label_map_path else None

        # Set after load()
        self._data: Optional[np.ndarray] = None        # (X,Y,Z) int label volume
        self._affine: Optional[np.ndarray] = None       # 4x4 voxel-to-world
        self._inv_affine: Optional[np.ndarray] = None   # 4x4 world-to-voxel
        self._voxel_vol: float = 1.0                    # mm^3 per voxel
        self._label_names: dict[int, str] = {}          # label_id -> name
        self._label_volumes: dict[int, float] = {}      # label_id -> volume in mm^3
        self._mesh_cache: dict[int, object] = {}        # label_id -> vtkPolyData
        self._unique_labels: set[int] = set()

    def load(self) -> bool:
        """Load the NIfTI volume and label map.

        Returns True on success, False on failure (with printed warning).
        """
        if not HAS_NIBABEL:
            print("[extra-parcellation] nibabel not installed. "
                  "Install with: pip install nibabel")
            return False

        if not self.nifti_path.exists():
            print(f"[extra-parcellation] File not found: {self.nifti_path}")
            return False

        try:
            img = nib.load(str(self.nifti_path))
            self._data = np.asarray(img.dataobj, dtype=np.int32)
            self._affine = img.affine.copy()
            self._inv_affine = np.linalg.inv(self._affine)

            # Compute voxel volume from affine
            vox_sizes = np.sqrt(np.sum(self._affine[:3, :3] ** 2, axis=0))
            self._voxel_vol = float(np.prod(vox_sizes))

            # Find all unique labels (excluding 0 = background)
            self._unique_labels = set(np.unique(self._data)) - {0}

            # Pre-compute volumes
            for label_id in self._unique_labels:
                voxel_count = int(np.sum(self._data == label_id))
                self._label_volumes[label_id] = voxel_count * self._voxel_vol

            # Load label names
            self._label_names = {}
            if self.label_map_path and self.label_map_path.exists():
                try:
                    raw = json.loads(self.label_map_path.read_text(encoding="utf-8"))
                    for k, v in raw.items():
                        self._label_names[int(k)] = str(v)
                except Exception as e:
                    print(f"[extra-parcellation] Warning: could not load label map: {e}")

            # Fill in missing names
            for label_id in self._unique_labels:
                if label_id not in self._label_names:
                    self._label_names[label_id] = f"Region_{label_id}"

            print(f"[extra-parcellation] Loaded {self.nifti_path.name}: "
                  f"{len(self._unique_labels)} regions, "
                  f"voxel size {vox_sizes[0]:.1f}x{vox_sizes[1]:.1f}x{vox_sizes[2]:.1f} mm, "
                  f"volume shape {self._data.shape}")
            return True

        except Exception as e:
            print(f"[extra-parcellation] Failed to load {self.nifti_path}: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._data is not None

    def _world_to_voxel(self, point: np.ndarray) -> np.ndarray:
        """Convert world (MNI) coordinates to voxel indices."""
        pt4 = np.array([point[0], point[1], point[2], 1.0])
        vox = self._inv_affine @ pt4
        return vox[:3]

    def get_region_at_point(self, point: np.ndarray) -> Optional[dict]:
        """Look up which region a world-space point falls in.

        Args:
            point: (3,) array in world/MNI coordinates (mm)

        Returns:
            dict with {"label_id": int, "name": str, "volume_mm3": float}
            or None if point is outside any labeled region.
        """
        if self._data is None:
            return None

        vox = self._world_to_voxel(point)
        ix, iy, iz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))

        # Bounds check
        if (ix < 0 or iy < 0 or iz < 0 or
                ix >= self._data.shape[0] or
                iy >= self._data.shape[1] or
                iz >= self._data.shape[2]):
            return None

        label = int(self._data[ix, iy, iz])
        if label == 0:
            return None

        return {
            "label_id": label,
            "name": self._label_names.get(label, f"Region_{label}"),
            "volume_mm3": self._label_volumes.get(label, 0.0),
        }

    def get_nearby_region(self, point: np.ndarray, search_radius: int = 2) -> Optional[dict]:
        """Search nearby voxels if exact point has no label.

        Args:
            point: (3,) world coordinates
            search_radius: search cube half-size in voxels

        Returns:
            dict or None
        """
        if self._data is None:
            return None

        vox = self._world_to_voxel(point)
        cx, cy, cz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))

        best_label = None
        best_dist = float('inf')

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    ix, iy, iz = cx + dx, cy + dy, cz + dz
                    if (ix < 0 or iy < 0 or iz < 0 or
                            ix >= self._data.shape[0] or
                            iy >= self._data.shape[1] or
                            iz >= self._data.shape[2]):
                        continue
                    label = int(self._data[ix, iy, iz])
                    if label == 0:
                        continue
                    dist = dx*dx + dy*dy + dz*dz
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label

        if best_label is None:
            return None

        return {
            "label_id": best_label,
            "name": self._label_names.get(best_label, f"Region_{best_label}"),
            "volume_mm3": self._label_volumes.get(best_label, 0.0),
        }

    def get_region_name(self, label_id: int) -> str:
        return self._label_names.get(label_id, f"Region_{label_id}")

    def get_region_volume(self, label_id: int) -> float:
        return self._label_volumes.get(label_id, 0.0)

    def get_all_labels(self) -> list[int]:
        return sorted(self._unique_labels)

    def get_region_mesh(self, label_id: int):
        """Generate a VTK polydata mesh for a region via marching cubes.

        Meshes are cached after first generation. Returns None if
        scikit-image is not installed or the region has too few voxels.
        """
        if label_id in self._mesh_cache:
            return self._mesh_cache[label_id]

        if not HAS_SKIMAGE:
            print("[extra-parcellation] scikit-image not installed for mesh generation. "
                  "Install with: pip install scikit-image")
            return None

        if not HAS_VTK:
            return None

        if self._data is None:
            return None

        # Create binary mask
        mask = (self._data == label_id).astype(np.float32)
        if mask.sum() < 10:  # too small for meaningful mesh
            return None

        # Pad to avoid edge artifacts
        padded = np.pad(mask, 1, mode='constant', constant_values=0)

        try:
            verts, faces, normals, _ = marching_cubes(padded, level=0.5)
            # Remove padding offset
            verts = verts - 1.0

            # Transform vertices from voxel to world space
            ones = np.ones((len(verts), 1), dtype=np.float64)
            verts_h = np.hstack([verts, ones])
            world_verts = (self._affine @ verts_h.T).T[:, :3]

            # Build VTK polydata
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(world_verts.astype(np.float64), deep=True))

            cells = vtk.vtkCellArray()
            for face in faces:
                cells.InsertNextCell(3)
                for idx in face:
                    cells.InsertCellPoint(int(idx))

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(cells)

            # Compute normals for smooth rendering
            normal_gen = vtk.vtkPolyDataNormals()
            normal_gen.SetInputData(polydata)
            normal_gen.ComputePointNormalsOn()
            normal_gen.Update()
            polydata = normal_gen.GetOutput()

            self._mesh_cache[label_id] = polydata
            return polydata

        except Exception as e:
            print(f"[extra-parcellation] Mesh generation failed for label {label_id}: {e}")
            return None

    def get_region_center(self, label_id: int) -> Optional[np.ndarray]:
        """Get the center of mass of a region in world coordinates."""
        if self._data is None:
            return None

        coords = np.argwhere(self._data == label_id)
        if len(coords) == 0:
            return None

        center_vox = coords.mean(axis=0)
        center_world = self._affine @ np.array([*center_vox, 1.0])
        return center_world[:3]
