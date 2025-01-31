OpenFoam Hull Performance (OHP) Tool

1) Geometry Preprocess 1: Hull CAD to Watertight CFD Ready STL file >> now Manual (FreeCAD, Salome, Rhino) >> Auto WIP
2) Geometry Preprocess 2: Transform(Scale,Rotate,Translate), Mass and Inertia(for Motion), Boounding Box(fro Mesh) >> Blender (output: hull.stl, hullBounds.txt, hullMassInertia.txt)
3) Meshing: >> snappyHexMesh OpenFoam ESI 2406
4) Solving: Multiphase(VoF) 2DoF >> interFoam OpenFoam ESI 2406
5) Postprocess: Visualisation >> Paraview(Fields, Free Surface, Cuts), Python(Monitoring)

Future Plan
- Auto Watertight STL
- Add OpenWater, Self-Propulsion
- Direct Connect with Paraview
- WebApp


Change uitl to a package/module/library

Need to remove functionalities from blenderHullProperties.py