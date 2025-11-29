"""
This file contains all functions neccessary to compute isometric uv maps of paper mesh pieces.
All functions here are only allowed to read data from meshes but not to write it back!
"""

import numpy as np
from bpy.types import Mesh
from bmesh.types import BMesh, BMFace
from itertools import product

from . import io
from . import geometry
from . import utils

def compute_all_face_triangulation_indices(bm : BMesh):
    triangulation_indices_per_face = {}
    for face in bm.faces:
        _, tri_ids = geometry.triangulate_3d_polygon([v.co for v in face.verts], face.normal, [v.index for v in face.verts], True)
        triangulation_indices_per_face[face.index] = tri_ids
    return triangulation_indices_per_face

def compute_local_coordinate_system_with_all_transitions_to_it(bmesh : BMesh, face : BMFace):
    inner_affine_transforms = {}
    face_normal = face.normal
    face_vertices_ccw = face.verts
    assert len(face_vertices_ccw) > 2
    # Choose one coordinate system for the face
    local_2d_coord_system = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[0].co, face_vertices_ccw[1].co, face_normal)
    inner_affine_transforms[bmesh.edges.get([face_vertices_ccw[0], face_vertices_ccw[1]]).index] = geometry.AffineTransform2D()
    # Compute affine transformations from all other edges to the first one
    for i in range(1, len(face_vertices_ccw)):
        j = (i + 1) % len(face_vertices_ccw)
        curr_basis = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[i].co, face_vertices_ccw[j].co, face_normal)
        transition = geometry.affine_2d_transformation_between_two_2d_spaces_on_same_plane(local_2d_coord_system, curr_basis)
        inner_affine_transforms[bmesh.edges.get([face_vertices_ccw[i], face_vertices_ccw[j]]).index] = transition

    return local_2d_coord_system, inner_affine_transforms

def compute_2d_unfolded_triangles_of_component(bmesh : BMesh, face_list, pred_dict,
                                               face_triangulation_indices_dict, 
                                               inner_transform_data_per_face,
                                               local_2d_coord_system_per_face,
                                               skip_intersection_test=False):
    unfolded_triangulated_faces = {}
    affine_transform_to_root_coord_system_per_face = {}
    intersection_occured = False

    # propagate affine transformations through the face tree
    bmesh.edges.ensure_lookup_table()
    for processed_face_i, face_index in enumerate(face_list):
        face = bmesh.faces[face_index]
        pred, connecting_edge_id = pred_dict[face_index]

        if pred == face_index:
            # root face
            affine_transform_to_root_coord_system_per_face[face_index] = geometry.AffineTransform2D()
            verts_in_local_space_curr = {v.index : geometry.to_local_coords(v.co, *local_2d_coord_system_per_face[face_index]) for v in face.verts}
            unfolded_triangulated_faces[face_index] = [tuple([verts_in_local_space_curr[i] for i in tri_indices]) for tri_indices in face_triangulation_indices_dict[face_index]]
            continue
        
        # compute affine transformation to the root face coordinates
        connecting_edge = bmesh.edges[connecting_edge_id]
        pred_to_root : geometry.AffineTransform2D = affine_transform_to_root_coord_system_per_face[pred]
        pred_connecting_edge_to_local_2d : geometry.AffineTransform2D = inner_transform_data_per_face[pred][connecting_edge_id]
        curr_local_2d_to_connecting_edge : geometry.AffineTransform2D = inner_transform_data_per_face[face_index][connecting_edge_id].inverse()
        transform_at_edge = geometry.AffineTransform2D(-np.eye(2), np.array([connecting_edge.calc_length(), 0], dtype=np.float64))
        combined_transform = pred_to_root @ (pred_connecting_edge_to_local_2d @ (transform_at_edge @ curr_local_2d_to_connecting_edge))
        affine_transform_to_root_coord_system_per_face[face_index] = combined_transform

        # compute triangles in root face coords
        verts_in_local_space_root = {v.index : combined_transform * geometry.to_local_coords(v.co, *local_2d_coord_system_per_face[face_index]) for v in face.verts}
        unfolded_triangulated_faces[face_index] = [tuple([verts_in_local_space_root[i] for i in tri_indices]) for tri_indices in face_triangulation_indices_dict[face_index]]

        # check for any intersections 
        if skip_intersection_test or intersection_occured:
            continue # skip the test

        for other_face_i in range(processed_face_i):
            if intersection_occured:
                break
            other_face_index = face_list[other_face_i]
            for triangle_a, triangle_b in product(unfolded_triangulated_faces[other_face_index], unfolded_triangulated_faces[face_index]):
                if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                    intersection_occured = True
                    break

    return unfolded_triangulated_faces, affine_transform_to_root_coord_system_per_face, intersection_occured

# this version is more flexible as data can be read from the mesh object on the fly
def get_globally_consistent_2d_coord_in_face(mesh : Mesh, point_on_face_3d, face_index, component_id,
                                             local_coordinate_systems = None, affine_transforms_to_root = None):
    """ Maps a 3D point on a given face to the unfolded face in 2D """
    face_cs = local_coordinate_systems[face_index] if local_coordinate_systems is not None else io.read_local_coordinate_system_of_face(mesh, face_index)
    face_transform_to_root = affine_transforms_to_root[component_id][face_index] if affine_transforms_to_root is not None else io.read_affine_transform_to_roots_of_face_in_component(mesh, component_id, face_index)
    return face_transform_to_root * geometry.to_local_coords(point_on_face_3d, *face_cs)
