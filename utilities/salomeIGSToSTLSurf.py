#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.12.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
kcs_igs_1 = geompy.ImportIGES("kcs.igs")
Sewing_1 = geompy.Sew(kcs_igs_1, 0.1)
Plane_1 = geompy.MakePlaneLCS(None, 2000, 3)
Mirror_1 = geompy.MakeMirrorByPlane(Sewing_1, Plane_1)
Sewing_2 = geompy.Sew([Sewing_1, Mirror_1], 0.1)
geompy.ExportSTL(Sewing_2, "hull_open.stl", True, 1e-06, True)

