import trimesh
import numpy as np
import sys

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

# Modified section for command line arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python trimeshClose.py <input_stl_file>")
        sys.exit(1)

    input_stl = sys.argv[1]
    output_stl = input_stl.rsplit('.', 1)[0] + '_closed.stl'

    mesh = trimesh.load_mesh(input_stl)
    print(f"Is the mesh watertight? {mesh.is_watertight}")
    print("\nOriginal mesh properties:")
    print(f"Volume: {mesh.volume:.2f} cubic units")
    print(f"Surface area: {mesh.area:.2f} square units")
    print(f"Center of mass: {mesh.center_mass}")
    print(f"Bounding box dimensions: {mesh.extents} (width, height, depth)")
    print(f"Bounding box bounds: min {mesh.bounds[0]}, max {mesh.bounds[1]}")

    if not mesh.is_watertight:
        print("\nThe mesh has open edges. Processing...")
        
        # First: identify broken faces
        broken = trimesh.repair.broken_faces(mesh)
        print(f"Found {len(broken)} broken faces")

        # First attempt: extrude edges to XZ plane
        modified_mesh = extrude_edges_to_xz(mesh)
        
        if not modified_mesh.is_watertight:
            print("First attempt unsuccessful. Trying repair sequence...")
            
            # Second attempt: comprehensive repair sequence
            # trimesh.repair.fix_normals(modified_mesh, multibody=True)
            # trimesh.repair.fix_inversion(modified_mesh, multibody=True)
            # trimesh.repair.fix_winding(modified_mesh)
            trimesh.repair.fill_holes(modified_mesh)
            modified_mesh.process()

        print("\nModified mesh properties:")
        print(f"Volume: {modified_mesh.volume:.2f} cubic units")
        print(f"Surface area: {modified_mesh.area:.2f} square units")
        print(f"Center of mass: {modified_mesh.center_mass}")
        print(f"Bounding box dimensions: {modified_mesh.extents} (width, height, depth)")
        print(f"Bounding box bounds: min {modified_mesh.bounds[0]}, max {modified_mesh.bounds[1]}")
        
        modified_mesh.export(output_stl)
        print(f"\nModified STL saved to {output_stl}")
        
        if not modified_mesh.is_watertight:
            print("\nWarning: The modified mesh is still not watertight.")
            print("The repair attempts were only partially successful.")
            sys.exit(1)
        else:
            print("\nSuccessfully created a watertight mesh!")
    else:
        print("The mesh is already watertight. No extrusion needed.")
