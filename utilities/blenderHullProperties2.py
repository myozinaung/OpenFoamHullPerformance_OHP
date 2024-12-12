import bpy
import bmesh
import argparse
import math


def load_mesh_from_stl(stl_filepath):
    """Load a mesh from an STL file and return the object reference."""
    # Open a new blank file (without default cube, camera, etc.)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_filepath)
    obj = bpy.context.selected_objects[0]  # Reference to the imported object

    # Ensure the object is selected and active
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    return obj


def scale_object(obj, scale_factor):
    """Scale the object by scale_factor."""
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
    # Apply the scale so that subsequent calculations use the updated geometry
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.update()


def compute_hull_properties(obj, draft, rho_water):
    """
    Compute hull volume, mass, inertia, CoG, and CoB from an object.
    This function will also clip the hull underwater portion at the given draft.
    """
    ### Get Dimensions ###
    dim = obj.dimensions
    Length = dim[0] * 0.94  # Length LPP approximation
    Beam   = dim[1]         # Beam of the hull
    Depth  = dim[2]         # Depth of the hull
    VCG    = Depth * 0.65   # Approximate Vertical Center of Gravity (custom logic)

    ### Write bounding box before clipping ###
    bbox = obj.bound_box
    write_bounding_box(bbox, draft, filename='hullBounds_before_clipping.txt')

    ### Clip the hull at the draft ###
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.bisect(
        plane_co=(0, 0, draft),
        plane_no=(0, 0, 1),
        use_fill=True,
        clear_inner=False,  # Keep lower part
        clear_outer=True    # Remove upper part
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    # After clipping, apply transformations and recalculate volume
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.update()

    # Calculate volume using bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    vol = bm.calc_volume()
    bm.free()

    # Set Origin to Center of Volume to get CoB
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS')
    transform_x = obj.location.x
    transform_y = obj.location.y
    transform_z = obj.location.z

    # LCB ~ transform_x, set CoG (assuming symmetrical Y)
    # CoG differs from CoB by using VCG for vertical
    LCB = transform_x
    CoG = (LCB, 0.0, VCG)

    # Half hull mass
    mass = vol * rho_water / 2

    # Radii of gyration approximations
    kxx = 0.34 * Beam
    kyy = 0.25 * Length
    kzz = 0.26 * Length

    Ixx = mass * kxx**2
    Iyy = mass * kyy**2
    Izz = mass * kzz**2

    # Write results to file
    with open("hullMassInertiaCoG.txt", "w") as f:
        f.write(f"mass            {mass:.2f};\n")
        f.write(f"Ixx             {Ixx:.2f};\n")
        f.write(f"Iyy             {Iyy:.2f};\n")
        f.write(f"Izz             {Izz:.2f};\n")
        f.write(f"centreOfMass    ({CoG[0]:.6f} {CoG[1]:.6f} {CoG[2]:.6f});\n")

    return mass, Ixx, Iyy, Izz, CoG


def rotate_object(obj, trim_angle, center):
    """Rotate the object about the Y-axis by trim_angle degrees around a given center."""
    # Convert angle to radians
    trim_angle_rad = math.radians(trim_angle)
    bpy.ops.transform.rotate(value=trim_angle_rad, orient_axis='Y', orient_type='LOCAL', center_override=center)
    # Apply rotation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.update()


def translate_object(obj, sinkage):
    """Translate the object by sinkage along Z."""
    bpy.ops.transform.translate(value=(0, 0, sinkage))
    # Apply translation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.data.update()


def compute_bounding_box(obj):
    """Compute the bounding box of the object and return it."""
    return obj.bound_box


def write_bounding_box(bbox, draft, filename='hullBounds.txt'):
    """Write bounding box values to a file."""
    with open(filename, 'w') as f:
        f.write(f"hullXmin  {bbox[0][0]:.4f};\n")
        f.write(f"hullXmax  {bbox[6][0]:.4f};\n")
        f.write(f"hullYmin  {bbox[0][1]:.4f};\n")
        # Assuming the hull is symmetric about Y-axis
        f.write(f"hullYmax  {0.0};\n")
        f.write(f"hullZmin  {bbox[0][2]:.4f};\n")
        f.write(f"hullZmax  {bbox[6][2]:.4f};\n")
        f.write(f"zWL       {draft:.4f};\n")


def export_stl(obj, filename="hull_export.stl"):
    """Export the object as an STL file."""
    bpy.ops.wm.stl_export(filepath=filename)


if __name__ == "__main__":
    # Argument parser for input arguments
    parser = argparse.ArgumentParser(description="Calculate hull properties")
    parser.add_argument("stl_filepath", type=str, help="Path to the STL file")
    parser.add_argument("--draft", type=float, required=True, help="Draft of the hull")
    parser.add_argument("--rho_water", type=float, default=1000, help="Density of water in kg/m^3")
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for the hull")
    parser.add_argument("--trim_angle", type=float, default=0, help="Trim angle in degrees")
    parser.add_argument("--sinkage", type=float, default=0, help="Sinkage in meters")
    parser.add_argument("--CoG", type=float, default=None, nargs=3, help="Center of Gravity (CoG) in metre")

    args = parser.parse_args()

    # 1. Load
    obj = load_mesh_from_stl(args.stl_filepath)

    # 2. Scale
    scale_object(obj, args.scale_factor)

    # 3. Compute Mass, Inertia, CoG
    mass, Ixx, Iyy, Izz, CoG = compute_hull_properties(obj, args.draft, args.rho_water)
    if args.CoG is not None:
        CoG = tuple(args.CoG)

    # 4. Rotate
    rotate_object(obj, args.trim_angle, CoG)

    # 5. Translate
    translate_object(obj, args.sinkage)

    # 6. Bounding Box
    bbox = compute_bounding_box(obj)
    write_bounding_box(bbox, args.draft, filename='hullBounds.txt')

    # Export final STL
    export_stl(obj, filename="hull_export.stl")

    print("Results written to hullBounds.txt, hullMassInertiaCoG.txt, and hull.stl")