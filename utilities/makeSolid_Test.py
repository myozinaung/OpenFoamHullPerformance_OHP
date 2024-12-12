import trimesh

# Load the STL file
mesh = trimesh.load("hull_surface.stl")

# Check if the mesh is already watertight
if not mesh.is_watertight:
    # Attempt to fill holes automatically
    # This tries to create faces in place of large gaps.
    # Note: Large or complex openings may not be perfectly corrected automatically.
    mesh = mesh.fill_holes()

    # After filling holes, it might help to do some cleanup
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    
    # Check again if watertight after attempt
    if not mesh.is_watertight:
        print("Mesh is still not watertight, consider manual repair steps or more advanced methods.")
    else:
        print("Mesh is now watertight.")

# Export the fixed mesh
mesh.export("watertight_ship_hull.stl")
