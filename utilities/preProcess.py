from processGeometry import GeometryProcessor
from shutil import copyfile

# Create an instance
processor = GeometryProcessor()
base_STL = "geometry/hull.stl"
underwater_STL = "geometry/hullUnderwater.stl"
rho_water = 1025
bottom_keel_depth = 0.0

# input_file = "geometry/hull.stl"

# input_file = "geometry/hullDTC.stl"
# scale = 0.001*1/59.4


# input_file = "geometry/kcs.igs"
# scale = 0.001*1/40
# rotate = (10, -5, 0) # y-axis +ive bow down
# translate = (0, 0, 0)
# draft = 0.27 # after scaling and transformation
# half_domain = False

# input_file = "geometry/jbc.igs"
# scale = 1/40
# rotate = (180, 0, 180)
# translate = (0, 0, 0)
# draft = 0.48

# input_file = "geometry/imsyacht.igs"
# scale = 0.001*1/2.71
# rotate = (0, 0, 0) # y-axis +ive bow down, x-axis +ive incline to -ive y side
# translate = (0, 0, 0)
# draft = 0.0 # after scaling and transformation
# half_domain = False

# input_file = "geometry/ilca.stl"
# scale = 0.001
# rotate = (0, 0, 0) # y-axis +ive bow down, x-axis +ive incline to -ive y side
# translate = (0, 0, 0)
# draft = 0.25 # after scaling and transformation
# half_domain = False

input_file = "geometry/sailboat.step"
scale = 0.001*0.95
rotate = (0, -2, 0) # y-axis +ive bow down, x-axis +ive incline to -ive y side
translate = (0, 0, 0)
draft = 0.030761 # after scaling and transformation
half_domain = False
bottom_keel_depth = 0.7

# input_file = "geometry/ethan.stl"
# scale = 0.1
# rotate = (0, 0, 0) # y-axis +ive bow down, x-axis +ive incline to -ive y side
# translate = (0, 0, 0)
# draft = 0.05 # after scaling and transformation
# half_domain = False

# Configuration flags
do_transform = True
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
        mirror=False # gmesh mirror takes long time to run (due to Union?)
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



# Mirror a mesh along the y-axis
if do_mirror:
    success, message = processor.mirror_geometry(
        input_file=base_STL,
        output_file=base_STL,
        mirror_axis='y',  # can be 'x', 'y', or 'z'
        origin=True       # True to mirror through origin, False to mirror through mesh center
    )
    print(f"{GREEN}Mirror geometry:{RESET}", message)

# # Remesh surface
# success, message = processor.remesh_geometry(
#     input_file=base_STL,
#     output_file=base_STL,
#     max_hole_size=0.01,
#     target_edge_length=0.001
# )
# print(f"{GREEN}Remesh surface:{RESET}", message)

# Close openings in an STL file
if do_close_openings:
    success, message = processor.close_openings(
        input_file=base_STL,
        output_file=base_STL,
        method='trimesh' # gmsh, trimesh (faster)
    )
    print(f"{GREEN}Close openings:{RESET}", message)

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

# Write hull bounds and draft
success, message = processor.write_hull_bounds(
    input_file=base_STL,
    draft=draft,
    output_file="geometry/hullBounds.txt",
    half_domain=half_domain,
    bottom_keel_depth=bottom_keel_depth
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
    output_file="geometry/hullMassInertiaCoG.txt",
    half_domain=half_domain
)
print(f"{BLUE}Approximate mass properties:{RESET}", message)