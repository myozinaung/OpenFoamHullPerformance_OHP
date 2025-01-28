# trimesh lib output file size is significantly smaller than gmsh lib output file size
import gmsh
import numpy as np

def extrude_edges_to_xz_gmsh():
    """
    1. Reads an STL file into Gmsh as a discrete surface.
    2. Finds edges that belong to only one triangle (boundary edges).
    3. For each boundary edge, creates two new nodes projected onto the XZ plane (y=0).
    4. Creates two new triangular faces bridging each boundary edge to its projection.
    5. Exports the result as a new STL file.
    """

    # ------------------------------------------------------------------------------
    # User Inputs
    # ------------------------------------------------------------------------------
    input_stl  = "hull_surface.stl"
    output_stl = "output_file.stl"

    # ------------------------------------------------------------------------------
    # Initialize Gmsh and read the STL
    # ------------------------------------------------------------------------------
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("extrude_example")

    # Merge the STL file into the current model
    gmsh.merge(input_stl)

    # At this point, Gmsh has loaded the STL as a discrete surface. 
    # We can retrieve its mesh data directly.
    #
    # Note: If your STL has multiple surfaces, you may need to loop over them, or
    # combine them. This example assumes a single surface for simplicity.

    # ------------------------------------------------------------------------------
    # Retrieve all nodes (coordinates) in the mesh
    # ------------------------------------------------------------------------------
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    # nodeTags is an array of unique node IDs (e.g. [1,2,3,...])
    # nodeCoords is a flat [x1, y1, z1, x2, y2, z2, ...] array.

    # Build a quick map from nodeTag -> (x, y, z) and nodeTag -> localIndex
    # This will help in dealing with coordinates and adjacency.
    tag_to_local = {}
    coords_array = []
    for i, tag in enumerate(nodeTags):
        tag_to_local[tag] = i
        x = nodeCoords[3*i + 0]
        y = nodeCoords[3*i + 1]
        z = nodeCoords[3*i + 2]
        coords_array.append((x, y, z))
    coords_array = np.array(coords_array)  # shape: (numNodes, 3)

    # ------------------------------------------------------------------------------
    # Retrieve 2D elements (the triangles in the STL surface)
    # ------------------------------------------------------------------------------
    # getElements(dim=2) returns three lists of lists:
    #   eTypes   = [element_type_1, element_type_2, ...]
    #   eIds     = [ [elemTag1, elemTag2, ...], [ ... ] ... ]
    #   eNodes   = [ [n1, n2, n3, ...],         [ ... ] ... ]
    # Usually for an STL, there's only one element type (2 = triangle).
    # But handle the general case by looping over them.
    eTypes, eIds, eNodes = gmsh.model.mesh.getElements(dim=2)

    all_triangles = []
    for etype, ids, nodes in zip(eTypes, eIds, eNodes):
        # Gmsh's type "2" means 3-node triangle
        if etype != 2:
            # If your STL has only triangles, you might skip or raise an error otherwise
            print(f"Skipping element type {etype} (not a 3-node triangle).")
            continue

        # 'nodes' is a flattened list of nodeTags for each triangle.
        # Each triangle has 3 node tags. So we group them in threes.
        for i in range(len(ids)):
            # Triangle index in this chunk
            base = 3 * i
            n1 = nodes[base + 0]
            n2 = nodes[base + 1]
            n3 = nodes[base + 2]
            all_triangles.append((n1, n2, n3))

    all_triangles = np.array(all_triangles, dtype=int)  # shape: (numTriangles, 3)

    # ------------------------------------------------------------------------------
    # Identify boundary edges (edges that occur in exactly one triangle)
    # ------------------------------------------------------------------------------
    # We'll collect edges in a dictionary with sorted node pairs as keys
    # and count how many times they appear.
    edge_dict = {}
    for tri in all_triangles:
        # For each triangle, get its three edges
        edges_in_tri = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for e in edges_in_tri:
            if e not in edge_dict:
                edge_dict[e] = 1
            else:
                edge_dict[e] += 1

    # Boundary edges are those that appear exactly once
    boundary_edges = [e for e, count in edge_dict.items() if count == 1]
    print(f"Detected {len(boundary_edges)} open edges.")

    if len(boundary_edges) == 0:
        print("Mesh is already watertight. No extrusion needed.")
        # Just write out the original model or do nothing
        gmsh.write(output_stl)
        gmsh.finalize()
        return

    # ------------------------------------------------------------------------------
    # Prepare arrays for new nodes and new triangles
    # ------------------------------------------------------------------------------
    new_node_coords   = []
    new_node_tags     = []
    new_triangles     = []  # store lists of [nodeTag1, nodeTag2, nodeTag3]

    # We will assign new node tags starting from max(nodeTags) + 1
    next_tag = max(nodeTags) + 1

    # For convenience, we’ll build a dictionary to store the new node tag
    # for the projected version of each old node (to avoid duplication).
    projected_node_map = {}

    # Helper function to get or create a projected node
    def get_projected_node(old_tag):
        """Return the new node tag for the old node's projection onto y=0."""
        if old_tag in projected_node_map:
            return projected_node_map[old_tag]

        nonlocal next_tag
        old_coords = coords_array[tag_to_local[old_tag]]
        # Project onto XZ plane by setting y=0
        xz_coords = (old_coords[0], 0.0, old_coords[2])

        new_node_tags.append(next_tag)
        new_node_coords.append(xz_coords)
        projected_node_map[old_tag] = next_tag

        next_tag += 1
        return projected_node_map[old_tag]

    # ------------------------------------------------------------------------------
    # Build new triangles to connect each boundary edge to its projection
    # ------------------------------------------------------------------------------
    for edge in boundary_edges:
        v1, v2 = edge  # old node tags (two endpoints of boundary edge)

        # Retrieve projected node tags (will create them if not already created)
        pv1 = get_projected_node(v1)
        pv2 = get_projected_node(v2)

        # Now form the two new triangles for the 'side wall':
        #
        #  1) (v1, v2, pv1)
        #  2) (pv1, pv2, v2)
        #
        # Graphically:
        #
        #   v1 ______ v2
        #    | \      |
        #    |  \     |
        #    pv1 --- pv2   (all "pv?" are old vertex projected to y=0 plane)
        #
        new_triangles.append([v1, v2, pv1])
        new_triangles.append([pv1, v2, pv2])

    print(f"Creating {len(new_node_coords)} new nodes and {len(new_triangles)} new triangles.")

    # ------------------------------------------------------------------------------
    # Insert new nodes and new elements into the Gmsh model
    # ------------------------------------------------------------------------------
    if len(new_node_coords) > 0:
        # Add the new nodes
        # gmsh.model.mesh.addNodes(dimension, tag, nodeTags, coordinates)
        # dimension here is typically 2 or 3 in Gmsh geometry, but because 
        # this is a "discrete" mesh, dimension can be somewhat flexible. 
        # We'll assume dimension=0 or dimension=2 doesn't matter much for raw mesh additions.
        #
        # Let’s just pick dim=2 for consistency with surface mesh:
        gmsh.model.mesh.addNodes(
            dim=2,
            tag=1,  # We can reuse the same "surface tag" or just set 1 for discrete
            nodeTags=new_node_tags,
            coord=[c for xyz in new_node_coords for c in xyz]
        )

        # Add the new elements (triangles)
        # gmsh.model.mesh.addElements(dimension, tag, elementTypes, elementTags, nodeTags)
        # We must supply unique element IDs. Let's build them:
        _, existing_elem_ids, _ = gmsh.model.mesh.getElements(dim=2)
        # Flatten the existing elements' IDs to find the max:
        flattened_ids = np.concatenate(existing_elem_ids) if existing_elem_ids else []
        next_elem_id  = max(flattened_ids) + 1 if len(flattened_ids) > 0 else 1

        new_elem_tags = []
        for _ in new_triangles:
            new_elem_tags.append(next_elem_id)
            next_elem_id += 1

        # Flatten the node connectivity
        new_elem_node_tags = [node for tri in new_triangles for node in tri]

        gmsh.model.mesh.addElements(
            dim=2,
            tag=1,
            elementTypes=[2],          # "2" = 3-node triangle
            elementTags=[new_elem_tags],
            nodeTags=[new_elem_node_tags]
        )

    # ------------------------------------------------------------------------------
    # Write out the modified mesh as an STL
    # ------------------------------------------------------------------------------
    gmsh.write(output_stl)
    print(f"Modified STL saved to {output_stl}")

    gmsh.finalize()

# ------------------------------------------------------------------------------
# Run the extrusion process
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    extrude_edges_to_xz_gmsh()
