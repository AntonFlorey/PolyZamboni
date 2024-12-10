import numpy as np
import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion

def compute_bb_diameter(mesh : bmesh.types.BMesh):
    positions = np.asarray([np.asarray(v.co) for v in mesh.verts])
    mincorner = np.min(positions, axis=0)
    maxcorner = np.max(positions, axis=0)
    
    return np.linalg.norm(maxcorner - mincorner)

def compute_mesh_surface_area(mesh : bmesh.types.BMesh):
    return sum([f.calc_area() for f in mesh.faces])

def compute_voronoi_areas(mesh : bmesh.types.BMesh):
    voronoi_areas = {}
    face_areas = {}
    for f in mesh.faces:
        face_areas[f.index] = f.calc_area()
    for v in mesh.verts:
        voronoi_areas[v.index] = sum([face_areas[f.index] / len(f.verts) for f in v.link_faces])

    # assertion
    # print("area sum:", sum(face_areas.values()))
    # print("voronoi area sum:", sum(voronoi_areas.values()))
    assert np.allclose(sum(face_areas.values()), sum(voronoi_areas.values()))

    return voronoi_areas