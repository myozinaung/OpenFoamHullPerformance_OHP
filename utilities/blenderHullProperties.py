# Import >> Scale >> Mass, Inertia (draft Model Scale) >> Rotate >> Translate >> Bounding Box >> Export
# pip install bpy
import bpy
import bmesh
import argparse

def move_origin(stl_filepath):
    # Open a new blank file (without default cube and camera)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_filepath)
    obj = bpy.context.selected_objects[0]  # Reference to the imported object

    # Ensure the object is selected
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Set the origin to the bottom aft center of the hull (min X, middle Y, min Z)
    bbox = obj.bound_box
    origin_new = (-bbox[0][0], -(bbox[0][1]+bbox[6][1])/2, -bbox[0][2])
    obj.location = origin_new

    # Export the modified object as an STL file
    hull_path_new = "geometry/hull_origin_moved.stl"
    bpy.ops.wm.stl_export(filepath=hull_path_new)
    print(f"Results written to hull_origin_moved.stl")
    return hull_path_new

def calculate_hull_properties(stl_filepath, draft, scale_factor, rho_water):
    ### 1. Import the STL file ###
    bpy.ops.wm.read_factory_settings(use_empty=True) # Open a new blank file (without default cube and camera)    
    bpy.ops.wm.stl_import(filepath=stl_filepath) # Import the STL file
    obj = bpy.context.selected_objects[0]  # Reference to the imported object
    
    bpy.context.view_layer.objects.active = obj # Ensure the object is selected
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    bbox = obj.bound_box # Get the bounding box of the hull before scaling
    print("Bounding Box Before Scaling:")
    print(f"Bounding Box Min: {bbox[0][0]:.4f}, {bbox[0][1]:.4f}, {bbox[0][2]:.4f}")
    print(f"Bounding Box Max: {bbox[6][0]:.4f}, {bbox[6][1]:.4f}, {bbox[6][2]:.4f}")

    ### 2. Scale the hull using scale factor ###
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) # Apply transformations to update the geometry
    obj.data.update() # Update the object's data # optional    

    bbox = obj.bound_box # Get the bounding box of the hull after scaling
    print("Bounding Box After Scaling:")
    print(f"Bounding Box Min: {bbox[0][0]:.4f}, {bbox[0][1]:.4f}, {bbox[0][2]:.4f}")
    print(f"Bounding Box Max: {bbox[6][0]:.4f}, {bbox[6][1]:.4f}, {bbox[6][2]:.4f}")

    ### 3. Mass, Inertia and CoG ###    
    bpy.ops.object.mode_set(mode='EDIT') # Switch to Edit Mode    
    bpy.ops.mesh.select_all(action='SELECT') # Select all geometry mesh
    
    # Clip the underwater portion of the hull at the draft
    draft_scaled = draft * scale_factor
    bpy.ops.mesh.bisect(
        plane_co=(0, 0, draft_scaled),    # Point on the Z-plane (origin)
        plane_no=(0, 0, 1),    # Normal vector of the plane (along Z-axis)
        use_fill=True,         # Fill the cut plane
        clear_inner=False,     # Keep the lower part
        clear_outer=True       # Remove the upper part
    ) # Clip the hull: Bisect the mesh at the draft
    
    bpy.ops.object.mode_set(mode='OBJECT') # Return to Object Mode    
    bpy.ops.wm.stl_export(filepath="geometry/hull_underwater.stl")

    # Calculate the underwater volume and mass of the object
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    vol = bm.calc_volume()
    bm.free()
    mass = vol * rho_water / 2  # mass of half hull (assuming symmetry)

    # Get the dimensions of the hull
    dim = obj.dimensions
    print(f"Dimensions: {dim}")
    Length = dim[0] * 0.94  # Length of the hull, LPP = 94% of the LOA for Inertia calculation
    Beam   = dim[1]  # Beam of the hull
    Depth  = dim[2]  # Depth of the hull
    VCG    = Depth * 0.65  # Vertical Center of Gravity (VCG), Use the formulae for different types of vessels

    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='BOUNDS') # Set Origin to Center of Mass (Volume) just get the CoB
    # Get the Transform X, Y, Z
    transform_x = LCB = obj.location.x # LCB or LCG
    transform_y = TCB = obj.location.y
    transform_z = VCB = obj.location.z

    CoB = (LCB, TCB, VCB)
    CoG = (LCB, 0, VCG)   

    kxx = 0.34 * Beam
    kyy = 0.25 * Length
    kzz = 0.26 * Length

    Ixx = mass * kxx**2
    Iyy = mass * kyy**2
    Izz = mass * kzz**2

    print(f"Mass: {mass:.4f}")
    print(f"Ixx: {Ixx:.4f}, Iyy: {Iyy:.4f}, Izz: {Izz:.4f}")
    print(f"CoB X: {LCB:.4f}, Y: {TCB:.4f}, Z: {VCB:.4f}")
    print(f"CoG X: {LCB:.4f}, Y: {0}, Z: {VCG:.4f}")
    # Write results to file
    with open("hullMassInertiaCoG.txt", "w") as f:
        f.write(f"mass            {mass:.4f};   // [kg]\n")
        f.write(f"Ixx             {Ixx:.4f};       // [kg.m^2] for Roll motion (not important here)\n")
        f.write(f"Iyy             {Iyy:.4f};      // [kg.m^2] for Pitch motion\n")
        f.write(f"Izz             {Izz:.4f};      // [kg.m^2] for Yaw motion (not important here)\n")
        f.write(f"centreOfMass    ({LCB:.4f} {0} {VCG:.4f}); // [m] (x y z), (LCB 0 VCG)\n")
    print(f"Results written to hullMassInertiaCoG.txt")

    return CoG

def transform(stl_filepath, draft, scale_factor, CoG, trim_angle, sinkage):
    # Open a new blank file (without default cube and camera)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import the STL file
    bpy.ops.wm.stl_import(filepath=stl_filepath)
    obj = bpy.context.selected_objects[0]  # Reference to the imported object

    # Ensure the object is selected
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Scale the hull using scale factor #
    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))

    # Rotate the hull by the trim angle [deg] about the Y-axis at the CoG
    trim_angle_rad = trim_angle * 3.14159 / 180  # Convert to radians
    bpy.ops.transform.rotate(value=trim_angle_rad, orient_axis='Y', orient_type='LOCAL', center_override=CoG)

    # Translate the hull by the sinkage
    bpy.ops.transform.translate(value=(0, 0, sinkage*scale_factor))

    # Apply transformations to update the geometry    
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) # Apply transformations to update the geometry
    obj.data.update() # Update the object's data # optional

    # Calculate the new bounding box of the transformed hull
    bbox = obj.bound_box
    print("Bounding Box after Rotation and Translation:")
    print(f"Bounding Box Min: {bbox[0][0]:.4f}, {bbox[0][1]:.4f}, {bbox[0][2]:.4f}")
    print(f"Bounding Box Max: {bbox[6][0]:.4f}, {bbox[6][1]:.4f}, {bbox[6][2]:.4f}")
    # Write bounds to file
    with open('hullBounds.txt', 'w') as f:
        f.write(f"hullXmin  {bbox[0][0]:.4f};\n")
        f.write(f"hullXmax  {bbox[6][0]:.4f};\n")
        f.write(f"hullYmin  {bbox[0][1]:.4f};\n")
        f.write(f"hullYmax  {0.0};\n") # Assuming the hull is symmetric about Y-axis
        f.write(f"hullZmin  {bbox[0][2]:.4f};\n")
        f.write(f"hullZmax  {bbox[6][2]:.4f};\n")
        f.write(f"zWL       {draft*scale_factor:.4f};\n")
    print(f"Results written to hullBounds.txt after rotation and translation")

    # Export the modified object as an STL file
    bpy.ops.wm.stl_export(filepath="geometry/hull.stl")
    print(f"Results written to hull.stl")

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
    parser.add_argument("--move_origin", action="store_true", help="Move the hull origin to a new position")    

    args = parser.parse_args()

    if args.move_origin:
        hull_path_new = move_origin(args.stl_filepath)
        stl_filepath = hull_path_new
    else:
        stl_filepath = args.stl_filepath
    CoG = calculate_hull_properties(stl_filepath, args.draft, args.scale_factor, args.rho_water)
    if args.CoG is not None:
        CoG = args.CoG
    transform(stl_filepath, args.draft, args.scale_factor, CoG, args.trim_angle, args.sinkage)

# Usage: python3 blenderHullProperties.py hullDTC.stl --draft 0.244 --rho_water 1000 --scale_factor 0.001 --trim_angle 0 --sinkage 0 --CoG 0.586 0 0.156


