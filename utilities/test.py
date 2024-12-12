
# pip install bpy
import bpy
import argparse

def calculate_hull_properties(stl_filepath):
    # Open a new blank file (without default cube and camera)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_filepath)
    obj = bpy.context.selected_objects[0]  # Reference to the imported object

    # Ensure the object is selected
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Get the bounding box of the hull
    bbox = obj.bound_box
    print(f"Bounding Box Min: {bbox[0][0]:.4f}, {bbox[0][1]:.4f}, {bbox[0][2]:.4f}")
    print(f"Bounding Box Max: {bbox[6][0]:.4f}, {bbox[6][1]:.4f}, {bbox[6][2]:.4f}")
    # Write bounds to file
    # with open('hullBounds.txt', 'w') as f:
    #     f.write(f"hullXmin  {bbox[0][0]:.4f};\n")
    #     f.write(f"hullXmax  {bbox[6][0]:.4f};\n")
    #     f.write(f"hullYmin  {bbox[0][1]:.4f};\n")
    #     f.write(f"hullYmax  {0.0};\n") # Assuming the hull is symmetric about Y-axis
    #     f.write(f"hullZmin  {bbox[0][2]:.4f};\n")
    #     f.write(f"hullZmax  {bbox[6][2]:.4f};\n")
    #     f.write(f"zWL       {draft:.4f};\n")
    # print(f"Results written to hullBounds.txt")
    z_clip = 100

    # Create a large cube that intersects the mesh at z_clip
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cube_add(size=10000, location=(0,-5000,z_clip-5000))
    cutter = bpy.context.active_object
    cutter.name = "CutterCube"

    # Move to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add a Boolean modifier to the object
    bool_mod = obj.modifiers.new(name="Boolean_Cut", type='BOOLEAN')
    bool_mod.object = cutter
    bool_mod.operation = 'INTERSECT'  # or 'DIFFERENCE' depending on what you want to keep

    # Apply the Boolean modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)

    # Delete the cutter object if desired
    bpy.data.objects.remove(cutter, do_unlink=True)

    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export the modified object as an STL file
    bpy.ops.wm.stl_export(filepath="hull_clipped_closed.stl") 


if __name__ == "__main__":
    # Argument parser for input arguments
    parser = argparse.ArgumentParser(description="Calculate hull properties")
    parser.add_argument("stl_filepath", type=str, help="Path to the STL file")

    args = parser.parse_args()

    calculate_hull_properties(args.stl_filepath)


# Usage: python3 test.py jbc.stl


