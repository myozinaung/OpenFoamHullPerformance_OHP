# Get the bounds of the hull geometry
# Clip the hull STL file using a Z-plane at draft and save the clipped geometry as a new STL file
# need to use the python come with ParaView
# Usage: "c:/Program Files/ParaView 5.11.1/bin/pvpython.exe" automateParaview.py hull.stl 0.2264
# Input: STL file name (openfoam-ready stl and scaled), draft
# Output: hull_clipped.stl
import paraview.simple as pv
import argparse
import gzip
import shutil
import os

def decompress_gz_file(file_path):
    """Decompress a gzipped file."""
    decompressed_file_path = file_path[:-3]
    with gzip.open(file_path, 'rb') as f_in:
        with open(decompressed_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return decompressed_file_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Clip STL file using a Z-plane at draft and save the clipped geometry as a new STL file.')
    parser.add_argument('stl_file', type=str, help='Input STL file name (can be *.stl or *.stl.gz)')
    parser.add_argument('draft', type=float, help='Draft value for clipping')
    args = parser.parse_args()

    # Check if the file is compressed
    if args.stl_file.endswith('.gz'):
        stl_file = decompress_gz_file(args.stl_file)
        compressed = True
    else:
        stl_file = args.stl_file
        compressed = False

    # Load the STL file
    input_stl = pv.STLReader(FileNames=[stl_file])
    draft = args.draft

    # Get the bounding box of the geometry
    pv.UpdatePipeline()
    bounds = input_stl.GetDataInformation().GetBounds()
    print(f"Bounds of the geometry: {bounds}")

    # Write bounds to file
    with open('hullBounds.txt', 'w') as f:
        f.write(f"hullXmin  {bounds[0]:.4f};\n")
        f.write(f"hullXmax  {bounds[1]:.4f};\n")
        f.write(f"hullYmin  {bounds[2]:.4f};\n")
        f.write(f"hullYmax  {bounds[3]:.4f};\n")
        f.write(f"hullZmin  {bounds[4]:.4f};\n")
        f.write(f"hullZmax  {bounds[5]:.4f};\n")

    # Clip the geometry using a Z-plane
    clip = pv.Clip(Input=input_stl)
    clip.ClipType = 'Plane'
    clip.ClipType.Origin = [0, 0, draft]  # Replace 'draft' with the desired z-value
    clip.ClipType.Normal = [0, 0, 1]

    # Update the pipeline to apply the clip
    pv.UpdatePipeline()

    # Extract the surface
    surface = pv.ExtractSurface(Input=clip)

    # Save the clipped geometry as a new STL file
    output_file = 'hull_clipped.stl'
    pv.SaveData(output_file, proxy=surface)

    print(f"STL file has been clipped and saved as {output_file}")

    # Clean up decompressed file if necessary
    if compressed:
        os.remove(stl_file)

if __name__ == "__main__":
    main()
