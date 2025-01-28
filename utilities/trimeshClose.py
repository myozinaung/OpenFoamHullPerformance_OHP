import trimesh
import numpy as np

def extrude_edges_to_xz(mesh):
    # Find open edges (edges that are not shared between two faces)
    edges = mesh.edges
    edges_sorted = np.sort(edges, axis=1)  # Ensure consistent ordering
    edge_counts = trimesh.grouping.group_rows(edges_sorted, require_count=1)

    open_edges = edges[edge_counts]  # Only edges appearing once are boundaries
    print(f"Detected {len(open_edges)} open edges.")

    # If no open edges, return the original mesh
    if len(open_edges) == 0:
        print("No open edges detected. Returning the original mesh.")
        return mesh

    # Collect new vertices and faces
    new_vertices = []
    new_faces = []

    for edge in open_edges:
        v1, v2 = mesh.vertices[edge]

        # Create new vertices projected onto the XZ plane
        v1_proj = [v1[0], 0, v1[2]]
        v2_proj = [v2[0], 0, v2[2]]

        # Add the new vertices
        base_index = len(mesh.vertices) + len(new_vertices)
        new_vertices.extend([v1_proj, v2_proj])

        # Add faces to connect original and projected vertices
        new_faces.extend([
            [edge[0], edge[1], base_index],       # Original edge to first new vertex
            [base_index, base_index + 1, edge[1]]  # New edge connecting projection
        ])

    # Add new vertices and faces to the mesh
    new_vertices = np.array(new_vertices)
    new_faces = np.array(new_faces)

    combined_mesh = trimesh.Trimesh(
        vertices=np.vstack([mesh.vertices, new_vertices]),
        faces=np.vstack([mesh.faces, new_faces]),
        process=False
    )
    return combined_mesh

# Load your STL file
input_stl = "jbc_surf.stl"
output_stl = "output_file.stl"

mesh = trimesh.load_mesh(input_stl)
print(f"Is the mesh watertight? {mesh.is_watertight}")

if not mesh.is_watertight:
    print("The mesh has open edges. Processing...")

    modified_mesh = extrude_edges_to_xz(mesh)

    # Save the modified STL file
    modified_mesh.export(output_stl)
    print(f"Modified STL saved to {output_stl}")
else:
    print("The mesh is already watertight. No extrusion needed.")
