import trimesh
import numpy as np
from scipy.ndimage import binary_erosion

# 1. Load your ship hull mesh (STL, OBJ, etc.)
mesh = trimesh.load('geometry/hull.stl')

# 2. Voxelize the mesh at a chosen resolution (pitch).
#    A smaller pitch -> finer resolution but more memory usage.
pitch = 0.05  # Adjust as needed
voxel_grid = mesh.voxelized(pitch=pitch)

# 3. Extract the dense 3D occupancy matrix (boolean array).
#    'matrix' is an X x Y x Z array of booleans indicating filled voxels.
dense = voxel_grid.matrix.copy()

# 4. Morphological erosion using SciPy
#    This will remove a 'erosion_iterations' voxel "shells" from the object,
#    stripping away small or thin features.
#    You can customize the 'structure' (structuring element) and iterations.
erosion_iterations = 1  # Adjust this value to control erosion amount
structure = np.ones((3, 3, 3), dtype=bool)  # 3x3x3 neighborhood
eroded_dense = binary_erosion(dense, structure=structure, iterations=erosion_iterations)

# 5. Create a new VoxelGrid from the eroded boolean array
eroded_voxel_grid = trimesh.voxel.VoxelGrid(
    encoding=eroded_dense,
    transform=voxel_grid.transform
)

# 6. Convert eroded voxel grid back to a mesh using marching cubes
eroded_mesh = eroded_voxel_grid.marching_cubes

# 7. Compute bounding boxes for both original and eroded meshes
bbox_original = mesh.bounding_box_oriented
bbox_eroded = eroded_mesh.bounding_box_oriented

extents_original = bbox_original.extents
extents_eroded = bbox_eroded.extents

# Print both bounding box results with 3 decimal places
print("Original bounding box (oriented) extents:", np.round(extents_original, 3))
print("Eroded bounding box (oriented) extents:", np.round(extents_eroded, 3))

# Save the eroded mesh for inspection
eroded_mesh.export('geometry/hull_eroded.stl')
