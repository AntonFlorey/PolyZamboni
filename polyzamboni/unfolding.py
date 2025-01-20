import numpy as np
import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion
from itertools import product
import matplotlib.pyplot as plt
from . import geometry


def edge_to_key_e(e : bmesh.types.BMEdge):
    return tuple(sorted([v.index for v in e.verts]))

def edge_to_key(v1 : bmesh.types.BMVert, v2 : bmesh.types.BMVert):
    return tuple(sorted([v1.index, v2.index]))

def basis_at_face_edge(edge : bmesh.types.BMEdge, normal):
    vertices = edge.verts
    return geometry.construct_2d_space_along_face_edge()

def pos_list_to_triangles(pos_list):
    res = []
    for i in range(0, len(pos_list), 3):
        res.append((pos_list[i], pos_list[i+1], pos_list[i+2]))
    return res

class Unfolding():
    """ Part of a 2d mesh cut open and unfolded into the euclidean plane """

    def __init__(self, mesh : bmesh.types.BMesh, face_list, pred_dict):

        # object attributes
        self.has_overlaps = False
        self.flap_collides_with = {}
        self.affine_transform_to_root_coord_system_per_face = {}
        self.triangulated_faces_2d = {}
        self.interior_affine_transforms = {f_id : {} for f_id in face_list}
        self.local_2d_coord_system_per_face = {}
        self.glue_flaps_per_face = {f_id : {} for f_id in face_list} # store glue flaps here

        # precompute all necessary stuff per face
        mesh.faces.ensure_lookup_table()
        for face_index in face_list:
            face = mesh.faces[face_index]
            face_normal = face.normal

            face_vertices_ccw = face.verts
            assert len(face_vertices_ccw) > 2
            # Choose one coordinate system for the face
            self.local_2d_coord_system_per_face[face_index] = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[0].co, face_vertices_ccw[1].co, face_normal)
            self.interior_affine_transforms[face_index][edge_to_key(face_vertices_ccw[0], face_vertices_ccw[1])] = geometry.AffineTransform2D()

            # Compute affine transformations from all other edges to the first one
            for i in range(1, len(face_vertices_ccw)):
                j = (i + 1) % len(face_vertices_ccw)
                curr_basis = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[i].co, face_vertices_ccw[j].co, face_normal)
                transition = geometry.affine_2d_transformation_between_two_2d_spaces_on_same_plane(self.local_2d_coord_system_per_face[face_index], curr_basis)
                self.interior_affine_transforms[face_index][edge_to_key(face_vertices_ccw[i], face_vertices_ccw[j])] = transition

        # propagate affine transformations through the face tree
        mesh.edges.ensure_lookup_table()
        for processed_face_i, face_index in enumerate(face_list):
            face = mesh.faces[face_index]
            pred, connecting_edge_id = pred_dict[face_index]

            if pred == face_index:
                # root face
                self.affine_transform_to_root_coord_system_per_face[face_index] = geometry.AffineTransform2D()
                verts_in_local_space_curr = [geometry.to_local_coords(v.co, *self.local_2d_coord_system_per_face[face_index]) for v in face.verts]
                self.triangulated_faces_2d[face_index] = geometry.triangulate_2d_polygon(verts_in_local_space_curr, [v.index for v in face.verts], True)[0]
                # print("root face unfolded")
                continue
            
            # compute affine transformation to the root face coordinates
            connecting_edge = mesh.edges[connecting_edge_id]
            pred_to_root : geometry.AffineTransform2D = self.affine_transform_to_root_coord_system_per_face[pred]
            pred_connecting_edge_to_local_2d : geometry.AffineTransform2D = self.interior_affine_transforms[pred][edge_to_key_e(connecting_edge)]
            curr_local_2d_to_connecting_edge : geometry.AffineTransform2D = self.interior_affine_transforms[face_index][edge_to_key_e(connecting_edge)].inverse()
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
                geometry.debug_draw_polygons_2d(unfolded_polygons_so_far, "debug-unfolding.png")

            # compute triangles in root face coords
            verts_in_local_space_root = [self.get_globally_consistend_2d_coord_in_face(v.co, face.index) for v in face.verts]
            self.triangulated_faces_2d[face_index] = geometry.triangulate_2d_polygon(verts_in_local_space_root, [v.index for v in face.verts], True)[0]

            # estimate error produced by the affine transformation for debugging
            for connecting_v in connecting_edge.verts:
                coords_in_prev_face = self.get_globally_consistend_2d_coord_in_face(connecting_v.co, pred)
                coords_in_curr_face = self.get_globally_consistend_2d_coord_in_face(connecting_v.co, face_index)
                hopefully_small_dist = np.linalg.norm(coords_in_curr_face - coords_in_prev_face)
                #print("This distance should be small:", hopefully_small_dist)

            # print("Face", processed_face_i, "unfolded. Checking for intersections...")

            # check for any intersections 
            if self.has_overlaps:
                continue # already lost :(

            for other_face_i in range(processed_face_i):
                if self.has_overlaps:
                    break
                other_face_index = face_list[other_face_i]
                other_face_triangles = pos_list_to_triangles(self.triangulated_faces_2d[other_face_index])
                current_face_triangles = pos_list_to_triangles(self.triangulated_faces_2d[face_index])
                for triangle_a, triangle_b in product(other_face_triangles, current_face_triangles):
                    if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                        self.has_flap_overlaps = True
                        break


    def get_globally_consistend_2d_coord_in_face(self, point_on_face_3d, face_index):
        """ Maps a 3D point on a given face to the unfolded face in 2D """

        in_local_coords = geometry.to_local_coords(point_on_face_3d, *self.local_2d_coord_system_per_face[face_index])
        return self.affine_transform_to_root_coord_system_per_face[face_index] * in_local_coords
    
    def compute_2d_glue_flap_triangles(self, face_index, edge : bmesh.types.BMEdge, flap_angle, flap_height):
        x = flap_height / np.tan(flap_angle)
        h = flap_height
        l = edge.calc_length()
        
        if l <= 2 * abs(x):
            # special emergency case...
            p_1_local_edge = np.array([0, 0])
            p_2_local_edge = np.array([l / 2, -h])
            p_3_local_edge = np.array([l, 0])
            edge_to_root = self.affine_transform_to_root_coord_system_per_face[face_index] @ self.interior_affine_transforms[face_index][edge_to_key_e(edge)]
            p_1_root_coords = edge_to_root * p_1_local_edge
            p_2_root_coords = edge_to_root * p_2_local_edge
            p_3_root_coords = edge_to_root * p_3_local_edge
            return [(p_1_root_coords, p_2_root_coords, p_3_root_coords)]

        # compute all flap points in local edge coordinates
        convex_flap = flap_angle <= np.pi / 2
        p_1_local_edge = np.array([0 if convex_flap else -x, 0])
        p_2_local_edge = np.array([x if convex_flap else 0, -h])
        p_3_local_edge = np.array([l - x if convex_flap else l, -h])
        p_4_local_edge = np.array([l if convex_flap else l + x, 0])

        # transform to root face coords
        edge_to_root = self.affine_transform_to_root_coord_system_per_face[face_index] @ self.interior_affine_transforms[face_index][edge_to_key_e(edge)]
        p_1_root_coords = edge_to_root * p_1_local_edge
        p_2_root_coords = edge_to_root * p_2_local_edge
        p_3_root_coords = edge_to_root * p_3_local_edge
        p_4_root_coords = edge_to_root * p_4_local_edge

        return [(p_1_root_coords, p_2_root_coords, p_3_root_coords), (p_1_root_coords, p_3_root_coords, p_4_root_coords)]

    def check_for_flap_collisions(self, flap_triangles, store_collisions=False, edge : bmesh.types.BMEdge =None):
        """ Returns true if the provided triangles overlap with any face or existing flap triangles """
        # first check against all triangles in the face
        all_triangles = []
        for tri_batch in self.triangulated_faces_2d.values():
            for tri in pos_list_to_triangles(tri_batch):
                all_triangles.append(tri)

        for unfold_triangle in all_triangles:
            for flap_triangle in flap_triangles:
                if geometry.triangle_intersection_test_2d(*unfold_triangle, *flap_triangle):
                    if store_collisions:
                        self.flap_collides_with.setdefault(edge_to_key_e(edge), set())
                        self.flap_collides_with[edge_to_key_e(edge)].add((-1,-1)) # (-1,-1) represents a collision with the unfolded mesh
                    return True
        
        # check agains all existing flaps
        for edge_to_triangles_dict in self.glue_flaps_per_face.values():
            for other_edge_key, other_flap_triangles in edge_to_triangles_dict.items():
                for other_flap_triangle in other_flap_triangles:
                    for flap_triangle in flap_triangles:
                        if geometry.triangle_intersection_test_2d(*other_flap_triangle, *flap_triangle):
                            if store_collisions:
                                self.flap_collides_with.setdefault(edge_to_key_e(edge), set())
                                self.flap_collides_with[edge_to_key_e(edge)].add(other_edge_key) # store the key of the other edge
                            else:
                                return True
                
        return False # no collisions
    
    def add_glue_flap_to_face_edge(self, face_index, input_edge, flap_triangles):
        """ Write flap triangles (in 2d root space) to a dict for later intersection tests and printing"""
        self.check_for_flap_collisions(flap_triangles, store_collisions=True, edge=input_edge)
        self.glue_flaps_per_face[face_index][edge_to_key_e(input_edge)] = flap_triangles

        # draw with glue flaps
        if False:
            unfolded_polygons = []
            for tri_batch in self.triangulated_faces_2d.values():
                for tri in pos_list_to_triangles(tri_batch):
                    unfolded_polygons.append(tri)

            for e_to_flap_dict in self.glue_flaps_per_face.values():
                for flap_triangles in e_to_flap_dict.values():
                    for flap_triangle in flap_triangles:
                        unfolded_polygons.append(flap_triangle)

            geometry.debug_draw_polygons_2d(unfolded_polygons, "debug-unfolding-with-flaps.png")

    def has_overlapping_glue_flaps(self):
        """ Return True if there is any glue flap that overlaps with any other geometry"""
        for registered_collisions in self.flap_collides_with.values():
            if len(registered_collisions) > 0:
                return True
        return False

    def check_if_edge_has_flap(self, face_index, edge):
        edge_key = edge_to_key_e(edge)
        return edge_key in self.glue_flaps_per_face[face_index].keys()

    def remove_flap_from_edge(self, face_index, edge):
        edge_key = edge_to_key_e(edge)
        if edge_key in self.glue_flaps_per_face[face_index]:
            del self.glue_flaps_per_face[face_index][edge_key]
        if edge_key not in self.flap_collides_with.keys():
            return
        for other_edge_key in self.flap_collides_with[edge_key]:
            if other_edge_key in self.flap_collides_with.keys() and edge_key in self.flap_collides_with[other_edge_key]:
                self.flap_collides_with[other_edge_key].remove(edge_key)
        del self.flap_collides_with[edge_key]

    def remove_all_flap_info(self):
        # remove all geometry info
        for face_index in self.glue_flaps_per_face.keys():
            self.glue_flaps_per_face[face_index] = {}
        # reset overlap info
        self.flap_collides_with = {}

    