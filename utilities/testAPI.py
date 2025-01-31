from processGeometry import GeometryProcessor

# Create an instance
processor = GeometryProcessor()

scale = 0.001/40.0

# Convert a file
success, message = processor.convert_to_stl(
    input_file="geometry/kcs.igs",
    output_file="geometry/hullSTL.stl",
    scale=scale,
    mirror=False # gmesh mirror takes long time to run (due to Union?)
)
print("Convert to STL: ", message)

# Mirror a mesh along the y-axis
success, message = processor.mirror_geometry(
    input_file="geometry/hullSTL.stl",
    output_file="geometry/hullSTLFull.stl",
    mirror_axis='y',  # can be 'x', 'y', or 'z'
    origin=True       # True to mirror through origin, False to mirror through mesh center
)
print("Mirror geometry: ", message)

# Close openings in an STL file
success, message = processor.close_openings(
    input_file="geometry/hullSTLFull.stl",
    output_file="geometry/hull.stl"
)
print("Close openings: ", message)

# Scale by 2, translate 10 units in x, and rotate 45 degrees around z
result = processor.transform_geometry(
    "geometry/hull.stl",
    "geometry/hullTransformed.stl",
    scale=2.0,
    translate=(10, 0, 0),
    rotate=(0, 0, 45)
)
print("Transform geometry: ", result)