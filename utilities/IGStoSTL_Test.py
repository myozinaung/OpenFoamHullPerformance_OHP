import gmsh
import math
import argparse

def main(igs_file, scale, mirror):
    # Initialize gmsh
    gmsh.initialize()

    # Merge the IGS file into Gmsh
    gmsh.merge(igs_file)

    # Classify surfaces: This step prepares surfaces for meshing by identifying patches.
    angle = 20  # Angle in degrees below which surfaces are merged
    forceParametrizablePatches = True
    includeBoundary = True
    curveAngle = 180

    gmsh.model.mesh.classifySurfaces(
        angle * math.pi / 180.0,
        includeBoundary,
        forceParametrizablePatches,
        curveAngle * math.pi / 180.0
    )

    # Create a geometry from the classified surfaces
    gmsh.model.mesh.createGeometry()
    gmsh.model.occ.synchronize()

    # Heal shapes to sew faces and try to form solids
    # Adjust tolerance if needed. Smaller tolerances may not close all gaps.
    # Larger tolerances may distort geometry if surfaces are far apart.
    sew_tolerance = 1e-3
    gmsh.model.occ.healShapes(
        gmsh.model.getEntities(dim=2),
        tolerance=sew_tolerance,
        makeSolids=True,
        sewFaces=True
    )
    gmsh.model.occ.synchronize()

    # Now we may have closed volumes. Let's apply transformations if needed.
    # Scale the geometry
    if scale != 1.0:
        gmsh.model.occ.dilate(gmsh.model.getEntities(), 0, 0, 0, scale, scale, scale)
        gmsh.model.occ.synchronize()

    # Mirror if requested
    if mirror:
        # Get existing entities
        entities = gmsh.model.getEntities()
        # Copy all entities
        gmsh.model.occ.copy(entities)
        gmsh.model.occ.synchronize()

        # The copied entities are at the end of the list
        all_entities = gmsh.model.getEntities()
        mirrored_entities = all_entities[-len(entities):]

        # Mirror around Y=0 plane (XZ-plane)
        gmsh.model.occ.mirror(mirrored_entities, 0, 1, 0, 0)
        gmsh.model.occ.synchronize()

        # Fuse original and mirrored
        gmsh.model.occ.fuse(entities, mirrored_entities)
        gmsh.model.occ.synchronize()

        # Heal again after mirroring to ensure watertightness
        gmsh.model.occ.healShapes(
            gmsh.model.getEntities(dim=2),
            tolerance=sew_tolerance,
            makeSolids=True,
            sewFaces=True
        )
        gmsh.model.occ.synchronize()

    # Identify if we now have volumes
    volumes = gmsh.model.getEntities(dim=3)

    # Set meshing algorithm
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    # Generate the mesh
    # If we have volumes, generate a 3D mesh, otherwise a 2D mesh
    if volumes:
        gmsh.model.mesh.generate(3)
    else:
        # If no volumes, fallback to surface mesh
        gmsh.model.mesh.generate(2)

    # Remove duplicate nodes for cleaner mesh
    gmsh.model.mesh.removeDuplicateNodes()

    # Save mesh to STL file
    gmsh.write('hull_surface.stl')

    # Finalize gmsh
    gmsh.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an IGS file to a watertight STL using Gmsh.")
    parser.add_argument("igs_file", type=str, help="Path to the input IGS file")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor (default is 1.0)")
    parser.add_argument("--mirror", action="store_true", help="Mirror the geometry in the XZ-plane")

    args = parser.parse_args()

    main(args.igs_file, args.scale, args.mirror)

# Example usage:
# python3 IGStoSTL.py jbc.igs --scale 0.001
