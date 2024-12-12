# pip install bpy
import bpy
import argparse

def make_manifold(stl_file):
    # Open a new blank file (without default cube and camera)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Ensure the 3D Print Toolbox is enabled (for making manifold)
    if not bpy.context.preferences.addons.get('object_print3d_utils'):
        bpy.ops.preferences.addon_enable(module='object_print3d_utils')

    # Import the STL file
    bpy.ops.import_mesh.stl(filepath=stl_file)

    ### Make manifold and calculate volume ###
    # Ensure we have the object selected
    obj = bpy.context.selected_objects[0]

    # Make Manifold using 3D Print Toolbox
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.print3d_clean_non_manifold()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Save the STL file with the manifold changes
    bpy.ops.export_mesh.stl(filepath="hull_solid.stl", check_existing=False, use_selection=True)
    print("Manifold hull saved to hull_solid.stl")

if __name__ == "__main__":
    # Argument parser for input arguments
    parser = argparse.ArgumentParser(description="STL Surface to Solid Converter")
    parser.add_argument("stl_file", type=str, help="Path to the STL file")

    args = parser.parse_args()

    make_manifold(args.stl_file)


# Usage: python3 STLSurfaceToSolid.py hull_surface.stl