from processGeometry import GeometryProcessor
from shutil import copyfile

# Create an instance
processor = GeometryProcessor()

input_file = "geometry/kcs.igs"
# input_file = "geometry/hullDTC.stl"
# input_file = "geometry/hull.stl"
base_STL = "geometry/hull.stl"
underwater_STL = "geometry/hullUnderwater.stl"

# scale = 0.001*1/59.4
scale = 0.001*1/40
rotate = (0, 0, 0)
translate = (0, 0, 0)
draft = 0.244*1000*40 # after scaling and transformation

# scale = 1/40
# rotate = (180, 0, 180)
# translate = (0, 0, 0)
# draft = 0.48

rho_water = 998

# Configuration flags
do_transform = False
do_mirror = False
do_close_openings = True

# Define color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Convert a file if input file is not STL    
if not input_file.lower().endswith('.stl'):
    success, message = processor.convert_to_stl(
        input_file=input_file,
        output_file=base_STL,
        scale=1.0, # scale is applied in transform_geometry
        mirror=True # gmesh mirror takes long time to run (due to Union?)
    )
    print(f"{BLUE}Convert to STL:{RESET}", message)
else:
    # If input is already STL, copy it to base_STL
    try:
        copyfile(input_file, base_STL)
        success, message = True, "File copied successfully"
    except Exception as e:
        success, message = False, f"Error copying file: {str(e)}"
    print(f"{BLUE}Copy STL file:{RESET}", message)

# Transform geometry
if do_transform:
    success, message = processor.transform_geometry(
        input_file=base_STL,
        output_file=base_STL,
        scale=scale,
        translate=translate,
        rotate=rotate # y-axis +ive bow down
    )
    print(f"{GREEN}Transform geometry:{RESET}", message)

# Mirror a mesh along the y-axis
if do_mirror:
    success, message = processor.mirror_geometry(
        input_file=base_STL,
        output_file=base_STL,
        mirror_axis='y',  # can be 'x', 'y', or 'z'
        origin=True       # True to mirror through origin, False to mirror through mesh center
    )
    print(f"{GREEN}Mirror geometry:{RESET}", message)

# Close openings in an STL file
if do_close_openings:
    success, message = processor.close_openings(
        input_file=base_STL,
        output_file=base_STL,
        method='trimesh' # gmsh, trimesh (faster)
    )
    print(f"{GREEN}Close openings:{RESET}", message)



# Write hull bounds and draft
success, message = processor.write_hull_bounds(
    input_file=base_STL,
    draft=draft,
    output_file="hullBounds.txt"
)
print(f"{BLUE}Write hull bounds:{RESET}", message)

# Cut by draft
success, message = processor.cut_by_draft(
    input_file=base_STL,
    draft=draft,
    output_file=underwater_STL,
    close_method="trimesh"
)
print(f"{BLUE}Cut by draft:{RESET}", message)

# Approximate mass properties
success, message = processor.approximate_mass_properties(
    original_stl=base_STL,
    clipped_stl=underwater_STL,
    rho_water=rho_water,  # seawater density
    output_file="hullMassInertiaCoG.txt"
)
print(f"{BLUE}Approximate mass properties:{RESET}", message)