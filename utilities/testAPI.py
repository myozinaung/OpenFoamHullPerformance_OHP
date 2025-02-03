from processGeometry import GeometryProcessor
from shutil import copyfile

# Create an instance
processor = GeometryProcessor()

input_file = "geometry/kcs.brep"
# input_file = "geometry/hull.stl"
base_STL = "geometry/hull.stl"
underwater_STL = "geometry/hullUnderwater.stl"

scale = 1/40
rotate = (0, 0, 0)
translate = (0, 0, 0)
draft = 0.27

# scale = 1/40
# rotate = (180, 0, 180)
# translate = (0, 0, 0)
# draft = 0.48

rho_water = 998

# Configuration flags
do_mirror = False
do_close_openings = True
do_transform = True

# Convert a file if input file is not STL    
if not input_file.lower().endswith('.stl'):
    success, message = processor.convert_to_stl(
        input_file=input_file,
        output_file=base_STL,
        scale=1.0, # scale is applied in transform_geometry
        mirror=False # gmesh mirror takes long time to run (due to Union?)
    )
    print("Convert to STL: ", message)
else:
    # If input is already STL, copy it to base_STL
    try:
        copyfile(input_file, base_STL)
        success, message = True, "File copied successfully"
    except Exception as e:
        success, message = False, f"Error copying file: {str(e)}"
    print("Copy STL file: ", message)

# Transform geometry
if do_transform:
    success, message = processor.transform_geometry(
        input_file=base_STL,
        output_file=base_STL,
        scale=scale,
        translate=translate,
        rotate=rotate # y-axis +ive bow down
    )
    print("Transform geometry: ", message)

# Mirror a mesh along the y-axis
if do_mirror:
    success, message = processor.mirror_geometry(
        input_file=base_STL,
        output_file=base_STL,
        mirror_axis='y',  # can be 'x', 'y', or 'z'
        origin=True       # True to mirror through origin, False to mirror through mesh center
    )
    print("Mirror geometry: ", message)

# Close openings in an STL file
if do_close_openings:
    success, message = processor.close_openings(
        input_file=base_STL,
        output_file=base_STL,
        method='trimesh' # gmsh, trimesh (faster)
    )
    print("Close openings: ", message)



# Write hull bounds and draft
success, message = processor.write_hull_bounds(
    input_file=base_STL,
    draft=draft,
    output_file="hullBounds.txt"
)
print("Write hull bounds: ", message)

# Cut by draft
success, message = processor.cut_by_draft(
    input_file=base_STL,
    draft=draft,
    output_file=underwater_STL,
    close_method="trimesh"
)
print("Cut by draft: ", message)

# Approximate mass properties
success, message = processor.approximate_mass_properties(
    original_stl=base_STL,
    clipped_stl=underwater_STL,
    rho_water=rho_water,  # seawater density
    output_file="hullMassInertiaCoG.txt"
)
print("Approximate mass properties: ", message)