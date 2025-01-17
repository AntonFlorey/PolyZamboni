import numpy as np
import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion
import matplotlib.pyplot as plt
from . import geometry

def edge_to_key_e(e : bmesh.types.BMEdge):
    return tuple(sorted([v.index for v in e.verts]))

def edge_to_key(v1 : bmesh.types.BMVert, v2 : bmesh.types.BMVert):
    return tuple(sorted([v1.index, v2.index]))

def basis_at_face_edge(edge : bmesh.types.BMEdge, normal):
    vertices = edge.verts
    return geometry.construct_2d_space_along_face_edge()

class Unfolding():
    """ Part of a 2d mesh cut open and unfolded into the euclidean plane """

    def __init__(self, mesh : bmesh.types.BMesh, face_list, pred_dict):

        # precompute all necessary stuff per face
        interior_affine_transforms = {f_id : {} for f_id in face_list}
        self.local_2d_coord_system_per_face = {}
        mesh.faces.ensure_lookup_table()
        for face_index in face_list:
            face = mesh.faces[face_index]
            face_normal = face.normal

            face_vertices_ccw = face.verts
            assert len(face_vertices_ccw) > 2
            # Choose one coordinate system for the face
            self.local_2d_coord_system_per_face[face_index] = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[0].co, face_vertices_ccw[1].co, face_normal)
            interior_affine_transforms[face_index][edge_to_key(face_vertices_ccw[0], face_vertices_ccw[1])] = geometry.AffineTransform2D()

            # Compute affine transformations from all other edges to the first one
            for i in range(1, len(face_vertices_ccw)):
                j = (i + 1) % len(face_vertices_ccw)
                curr_basis = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[i].co, face_vertices_ccw[j].co, face_normal)
                transition = geometry.affine_2d_transformation_between_two_2d_spaces_on_same_plane(self.local_2d_coord_system_per_face[face_index], curr_basis)
                interior_affine_transforms[face_index][edge_to_key(face_vertices_ccw[i], face_vertices_ccw[j])] = transition

        # propagate affine transformations through the face tree
        self.has_overlaps = False
        self.affine_transform_to_root_coord_system_per_face = {}
        triangulated_faces_2d = {}
        mesh.edges.ensure_lookup_table()
        for processed_face_i, face_index in enumerate(face_list):
            face = mesh.faces[face_index]
            pred, connecting_edge_id = pred_dict[face_index]

            if pred == face_index:
                # root face
                self.affine_transform_to_root_coord_system_per_face[face_index] = geometry.AffineTransform2D()
                verts_in_local_space_curr = [geometry.to_local_coords(v.co, *self.local_2d_coord_system_per_face[face_index]) for v in face.verts]
                triangulated_faces_2d[face_index] = geometry.triangulate_2d_polygon(verts_in_local_space_curr, [v.index for v in face.verts], True)[0]
                print("root face unfolded")
                continue
            
            # compute affine transformation to the root face coordinates
            connecting_edge = mesh.edges[connecting_edge_id]
            pred_to_root : geometry.AffineTransform2D = self.affine_transform_to_root_coord_system_per_face[pred]
            pred_connecting_edge_to_local_2d : geometry.AffineTransform2D = interior_affine_transforms[pred][edge_to_key_e(connecting_edge)]
            curr_local_2d_to_connecting_edge : geometry.AffineTransform2D = interior_affine_transforms[face_index][edge_to_key_e(connecting_edge)].inverse()
            transform_at_edge = geometry.AffineTransform2D(-np.eye(2), np.array([connecting_edge.calc_length(), 0]))
            combined_transform = pred_to_root @ (pred_connecting_edge_to_local_2d @ (transform_at_edge @ curr_local_2d_to_connecting_edge))
            self.affine_transform_to_root_coord_system_per_face[face_index] = combined_transform
            
            # save a figure of the unfolded polygons so far
            if False:
                unfolded_polygons_so_far = []
                for f_id_i in range(processed_face_i + 1):
                    _f_id = face_list[f_id_i]
                    curr_poly = [self.get_globally_consistend_2d_coord_in_face(v.co, _f_id) for v in mesh.faces[_f_id].verts]
                    unfolded_polygons_so_far.append(curr_poly)
                geometry.debug_draw_polygons_2d(unfolded_polygons_so_far)

            # compute triangles in root face coords
            verts_in_local_space_curr = [geometry.to_local_coords(v.co, *self.local_2d_coord_system_per_face[face_index]) for v in face.verts]
            verts_in_local_space_root = [combined_transform * local_coord for local_coord in verts_in_local_space_curr]
            triangulated_faces_2d[face_index] = geometry.triangulate_2d_polygon(verts_in_local_space_root, [v.index for v in face.verts], True)[0]

            # estimate error produced by the affine transformation for debugging
            for connecting_v in connecting_edge.verts:
                coords_in_prev_face = self.get_globally_consistend_2d_coord_in_face(connecting_v.co, pred)
                coords_in_curr_face = self.get_globally_consistend_2d_coord_in_face(connecting_v.co, face_index)
                hopefully_small_dist = np.linalg.norm(coords_in_curr_face - coords_in_prev_face)
                #print("This distance should be small:", hopefully_small_dist)

            print("Face", processed_face_i, "unfolded. Checking for intersections...")

            # check for any intersections 
            if self.has_overlaps:
                continue # already lost :(
            for other_face_i in range(processed_face_i):
                other_face_index = face_list[other_face_i]
                if other_face_index == pred:
                    print("skipping intersection test for this face")
                    continue # dont test intersections with the face connected to the added one
                for tri_in_curr_i in range(0, len(triangulated_faces_2d[face_index]), 3):
                    if self.has_overlaps:
                        break
                    for tri_in_other_i in range(0, len(triangulated_faces_2d[other_face_index]), 3):
                        if geometry.triangle_intersection_test_2d(triangulated_faces_2d[face_index][tri_in_curr_i],
                                                                  triangulated_faces_2d[face_index][tri_in_curr_i +1 ],
                                                                  triangulated_faces_2d[face_index][tri_in_curr_i + 2],
                                                                  triangulated_faces_2d[other_face_index][tri_in_other_i],
                                                                  triangulated_faces_2d[other_face_index][tri_in_other_i + 1],
                                                                  triangulated_faces_2d[other_face_index][tri_in_other_i + 2]): # THIS CODE IS NOT PRETTY
                            self.has_overlaps = True
                            break
            print("Intersection test done.")
                

    def get_globally_consistend_2d_coord_in_face(self, point_on_face_3d, face_index):
        """ Maps a 3D point on a given face to the unfolded face in 2D """

        in_local_coords = geometry.to_local_coords(point_on_face_3d, *self.local_2d_coord_system_per_face[face_index])
        return self.affine_transform_to_root_coord_system_per_face[face_index] * in_local_coords