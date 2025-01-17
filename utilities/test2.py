import bpy

# Set this to the desired deck Z-level.
deck_z = 0.0  # Adjust this value as needed.

# Ensure we have the correct object selected
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    raise ValueError("No active mesh object selected.")

# Switch to object mode to safely manipulate geometry data
bpy.ops.object.mode_set(mode='OBJECT')

mesh = obj.data

# Deselect everything
for v in mesh.vertices:
    v.select = False
for e in mesh.edges:
    e.select = False
for p in mesh.polygons:
    p.select = False

# Identify edges that lie exactly (or very close) at the deck_z level.
# We consider an edge selected if both its vertices are at the specified Z level.
tolerance = 1e-6
for e in mesh.edges:
    v1 = mesh.vertices[e.vertices[0]].co
    v2 = mesh.vertices[e.vertices[1]].co
    if abs(v1.z - deck_z) < tolerance and abs(v2.z - deck_z) < tolerance:
        e.select = True

# Switch to Edit mode to apply mesh operations
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='EDGE')

# Fill the hole with a face
# This will create a face (or multiple faces, if needed) from the selected edge loop.
bpy.ops.mesh.fill()

# Optional: You might consider triangulating or otherwise processing the new face(s).
# bpy.ops.mesh.triangulate()

# Switch back to object mode if needed
bpy.ops.object.mode_set(mode='OBJECT')

print("Deck successfully closed.")
