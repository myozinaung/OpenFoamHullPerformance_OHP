# Description: This script reads an IGS file, meshes it, scales it, and saves it as an STL file.
import gmsh
import math
import argparse

def main(input_file, scale, mirror):
    # Initialize gmsh
    gmsh.initialize()

    # Get file extension
    file_extension = input_file.lower().split('.')[-1]
    
    # Check supported file types
    supported_formats = {'igs', 'iges', 'stp', 'step', 'brep'}
    if file_extension not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {supported_formats}")

    # Merge the input file: This loads the file into Gmsh.
    try:
        gmsh.merge(input_file)
    except Exception as e:
        raise Exception(f"Failed to load {input_file}: {str(e)}")

    # Classify Surfaces: This step is necessary to prepare the surfaces for meshing.
    angle = 20  # Angle in degrees, below which two surfaces are considered to be on the same plane
    forceParametrizablePatches = True
    includeBoundary = True
    curveAngle = 180

    gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary, forceParametrizablePatches, curveAngle * math.pi / 180.)

    # Create a volume from the classified surfaces
    gmsh.model.mesh.createGeometry()

    # Set the meshing algorithm
    # 2D mesh algorithm (1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 11: Quasi-structured Quad)
    # Default value: 6
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # Mesh the geometry
    gmsh.model.mesh.generate(3)

    # # Optimize the mesh
    # gmsh.model.mesh.optimize("Netgen") # Netgen, HighOrder

    # # Synchronize the internal CAD representation with the Gmsh model
    # gmsh.model.occ.synchronize()

    # Apply scaling transformation
    gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, scale, scale, scale)

    if mirror:
        # Duplicate the original entities
        entities = gmsh.model.getEntities()  # Get all entities
        gmsh.model.occ.copy(entities)

        # Mirror the duplicate in the XZ-plane (Y=0)
        mirrored_entities = gmsh.model.getEntities()[-len(entities):]  # Get the last added entities
        gmsh.model.occ.mirror(mirrored_entities, 0, 1, 0, 0)  # Mirror in the XZ-plane

        # Combine the original and mirrored copies into a single model
        gmsh.model.occ.fuse(entities, mirrored_entities)

    # Synchronize the internal CAD representation with the Gmsh model
    gmsh.model.occ.synchronize()

    # Save mesh to STL file (replace 'output_file.stl' with your desired output file path)
    gmsh.write('hull_surface.stl')

    # Finalize gmsh
    gmsh.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CAD file (IGS/STEP/BREP) and convert it to STL.")
    parser.add_argument("input_file", type=str, help="Path to the input CAD file (supported formats: .igs, .iges, .stp, .step, .brep)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor (default is 1.0)")
    parser.add_argument("--mirror", action="store_true", help="Mirror the geometry in the XZ-plane")

    args = parser.parse_args()

    main(args.input_file, args.scale, args.mirror)

# Run the script with the following command:
# python3 IGStoSTL.py model.igs --scale 0.001
# or
# python3 IGStoSTL.py model.step --scale 0.001