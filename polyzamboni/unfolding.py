import numpy as np
import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion
from itertools import product
from . import geometry

def edge_to_key_e(e : bmesh.types.BMEdge):
    return tuple(sorted([v.index for v in e.verts]))

def edge_to_key(v1 : bmesh.types.BMVert, v2 : bmesh.types.BMVert):
    return tuple(sorted([v1.index, v2.index]))

class InnerFaceTransforms():
    def __init__(self, face : bmesh.types.BMFace):
        self.interior_affine_transforms = {}
        face_normal = face.normal
        face_vertices_ccw = face.verts
        assert len(face_vertices_ccw) > 2
        # Choose one coordinate system for the face
        self.local_2d_coord_system = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[0].co, face_vertices_ccw[1].co, face_normal)
        self.interior_affine_transforms[edge_to_key(face_vertices_ccw[0], face_vertices_ccw[1])] = geometry.AffineTransform2D()

        # Compute affine transformations from all other edges to the first one
        for i in range(1, len(face_vertices_ccw)):
            j = (i + 1) % len(face_vertices_ccw)
            curr_basis = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[i].co, face_vertices_ccw[j].co, face_normal)
            transition = geometry.affine_2d_transformation_between_two_2d_spaces_on_same_plane(self.local_2d_coord_system, curr_basis)
            self.interior_affine_transforms[edge_to_key(face_vertices_ccw[i], face_vertices_ccw[j])] = transition

class Unfolding():
    """ Part of a 2d mesh cut open and unfolded into the euclidean plane """

    def __init__(self, mesh : bmesh.types.BMesh, face_list, pred_dict, face_triangulation_indices_dict, inner_transform_data_per_face, skip_intersection_test=False):

        # object attributes
        self.has_overlaps = False
        self.flap_collides_with = {}
        self.affine_transform_to_root_coord_system_per_face = {}
        self.triangulated_faces_2d = {}
        self.triangulation_indices = {}
        self.interior_affine_transforms = {f_id : {} for f_id in face_list}
        self.local_2d_coord_system_per_face = {}
        self.glue_flaps_per_face = {f_id : {} for f_id in face_list} # store glue flaps here

        # get inner face transform data
        mesh.faces.ensure_lookup_table()
        for face_index in face_list:
            curr_face_transform_data : InnerFaceTransforms = inner_transform_data_per_face[face_index]
            self.local_2d_coord_system_per_face[face_index] = curr_face_transform_data.local_2d_coord_system
            self.interior_affine_transforms[face_index] = curr_face_transform_data.interior_affine_transforms

        # propagate affine transformations through the face tree
        mesh.edges.ensure_lookup_table()
        for processed_face_i, face_index in enumerate(face_list):
            face = mesh.faces[face_index]
            pred, connecting_edge_id = pred_dict[face_index]

            if pred == face_index:
                # root face
                self.affine_transform_to_root_coord_system_per_face[face_index] = geometry.AffineTransform2D()
                verts_in_local_space_curr = {v.index : geometry.to_local_coords(v.co, *self.local_2d_coord_system_per_face[face_index]) for v in face.verts}
                self.triangulated_faces_2d[face_index] = [tuple([verts_in_local_space_curr[i] for i in tri_indices]) for tri_indices in face_triangulation_indices_dict[face_index]]
                self.triangulation_indices[face_index] = face_triangulation_indices_dict[face_index]
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

            # compute triangles in root face coords
            verts_in_local_space_root = {v.index : self.get_globally_consistent_2d_coord_in_face(v.co, face.index) for v in face.verts}
            self.triangulated_faces_2d[face_index] = [tuple([verts_in_local_space_root[i] for i in tri_indices]) for tri_indices in face_triangulation_indices_dict[face_index]]
            self.triangulation_indices[face_index] = face_triangulation_indices_dict[face_index]

            # check for any intersections 
            if skip_intersection_test or self.has_overlaps:
                continue # skip the test

            for other_face_i in range(processed_face_i):
                if self.has_overlaps:
                    break
                other_face_index = face_list[other_face_i]
                for triangle_a, triangle_b in product(self.triangulated_faces_2d[other_face_index], self.triangulated_faces_2d[face_index]):
                    if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                        self.has_overlaps = True
                        break

    def get_globally_consistent_2d_coord_in_face(self, point_on_face_3d, face_index):
        """ Maps a 3D point on a given face to the unfolded face in 2D """

        in_local_coords = geometry.to_local_coords(point_on_face_3d, *self.local_2d_coord_system_per_face[face_index])
        return self.affine_transform_to_root_coord_system_per_face[face_index] * in_local_coords
    
    def __compute_2d_glue_flap_triangles_edge_local(self, edge : bmesh.types.BMEdge, flap_angle, flap_height):
        x = flap_height / np.tan(flap_angle)
        h = flap_height
        l = edge.calc_length()
        
        if l <= 2 * abs(x):
            # special emergency case...
            p_1_local_edge = np.array([0, 0])
            p_2_local_edge = np.array([l / 2, -h])
            p_3_local_edge = np.array([l, 0])
            return [(p_1_local_edge, p_2_local_edge, p_3_local_edge)]

        # compute all flap points in local edge coordinates
        convex_flap = flap_angle <= np.pi / 2
        p_1_local_edge = np.array([0 if convex_flap else -x, 0])
        p_2_local_edge = np.array([x if convex_flap else 0, -h])
        p_3_local_edge = np.array([l - x if convex_flap else l, -h])
        p_4_local_edge = np.array([l if convex_flap else l + x, 0])
        return [(p_1_local_edge, p_2_local_edge, p_3_local_edge), (p_1_local_edge, p_3_local_edge, p_4_local_edge)]

    def compute_2d_glue_flap_triangles(self, face_index, edge : bmesh.types.BMEdge, flap_angle, flap_height):
        triangles_in_local_edge_coords = self.__compute_2d_glue_flap_triangles_edge_local(edge, flap_angle, flap_height)
        edge_to_root = self.affine_transform_to_root_coord_system_per_face[face_index] @ self.interior_affine_transforms[face_index][edge_to_key_e(edge)]
        return [tuple([edge_to_root * local_coord for local_coord in triangle]) for triangle in triangles_in_local_edge_coords]

    def compute_3d_glue_flap_triangles_inside_face(self, face_index, edge : bmesh.types.BMEdge, flap_angle, flap_height):
        triangles_in_local_edge_coords = self.__compute_2d_glue_flap_triangles_edge_local(edge, flap_angle, flap_height)
        triangles_flipped = [tuple([np.array([local_coord[0], -local_coord[1]]) for local_coord in triangle]) for triangle in triangles_in_local_edge_coords]
        edge_to_local_coords = self.interior_affine_transforms[face_index][edge_to_key_e(edge)]
        triangles_in_3d = [tuple(reversed([geometry.to_world_coords(edge_to_local_coords * local_coord, *self.local_2d_coord_system_per_face[face_index]) for local_coord in triangle])) for triangle in triangles_flipped]
        return triangles_in_3d

    def check_for_flap_collisions(self, flap_triangles, store_collisions=False, edge : bmesh.types.BMEdge =None):
        """ Returns true if the provided triangles overlap with any face or existing flap triangles """
        collision_detected = False
        # first check against all triangles in the face
        all_triangles = []
        for tri_batch in self.triangulated_faces_2d.values():
            for tri in tri_batch:
                all_triangles.append(tri)

        def store_collision(colliding_edge_key, other_edge_key):
            self.flap_collides_with.setdefault(colliding_edge_key, set())
            self.flap_collides_with[colliding_edge_key].add(other_edge_key)            

        for unfold_triangle in all_triangles:
            for flap_triangle in flap_triangles:
                if geometry.triangle_intersection_test_2d(*unfold_triangle, *flap_triangle):
                    collision_detected = True
                    if store_collisions:
                        store_collision(edge_to_key_e(edge), (-1,-1))
                    else:
                        return True
        
        # check against all existing flaps
        for edge_to_triangles_dict in self.glue_flaps_per_face.values():
            for other_edge_key, other_flap_triangles in edge_to_triangles_dict.items():
                for other_flap_triangle in other_flap_triangles:
                    for flap_triangle in flap_triangles:
                        if geometry.triangle_intersection_test_2d(*other_flap_triangle, *flap_triangle):
                            collision_detected = True
                            if store_collisions:
                                store_collision(edge_to_key_e(edge), other_edge_key)
                                store_collision(other_edge_key, edge_to_key_e(edge))
                            else:
                                return True

        return collision_detected 
    
    def add_glue_flap_to_face_edge(self, face_index, input_edge, flap_triangles):
        """ Write flap triangles (in 2d root space) to a dict for later intersection tests and printing"""
        self.check_for_flap_collisions(flap_triangles, store_collisions=True, edge=input_edge)
        self.glue_flaps_per_face[face_index][edge_to_key_e(input_edge)] = flap_triangles

    def flap_is_overlapping(self, edge : bmesh.types.BMEdge):
        return edge_to_key_e(edge) in self.flap_collides_with.keys() and len(self.flap_collides_with[edge_to_key_e(edge)]) > 0

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

def test_if_two_touching_unfolded_components_overlap(unfolding_1 : Unfolding, 
                                                     unfolding_2 : Unfolding, 
                                                     join_face_index_1, 
                                                     join_face_index_2, 
                                                     join_verts_1,
                                                     join_verts_2):
    """ Test if two components undoldings would overlap if they were merged at the given faces. """

    # compute affine transformation from unfolding 1 to unfolding 2

    # translations
    join_point_1 = unfolding_1.get_globally_consistent_2d_coord_in_face(join_verts_1[1].co, join_face_index_1)
    join_point_2 = unfolding_2.get_globally_consistent_2d_coord_in_face(join_verts_2[0].co, join_face_index_2)
    join_point_1_to_orig = geometry.AffineTransform2D(affine_part=-join_point_1)
    orig_to_join_point_2 = geometry.AffineTransform2D(affine_part=join_point_2)
    # rotation
    x_ax_1, y_ax_1 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_1, unfolding_1.get_globally_consistent_2d_coord_in_face(join_verts_1[0].co, join_face_index_1))
    x_ax_2, y_ax_2 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_2, unfolding_2.get_globally_consistent_2d_coord_in_face(join_verts_2[1].co, join_face_index_2))
    basis_mat_1 = np.array([x_ax_1, y_ax_1]).T
    basis_mat_2 = np.array([x_ax_2, y_ax_2]).T
    rotate_edges_together = geometry.AffineTransform2D(linear_part=basis_mat_2 @ np.linalg.inv(basis_mat_1))
    # full transformation
    transform_first_unfolding = orig_to_join_point_2 @ rotate_edges_together @ join_point_1_to_orig

    for triangle_list_1 in unfolding_1.triangulated_faces_2d.values():
        unfolding_1_triangles = [tuple([transform_first_unfolding * tri_coord for tri_coord in triangle]) for triangle in triangle_list_1]
        for triangle_list_2 in unfolding_2.triangulated_faces_2d.values():
            for triangle_a, triangle_b in product(unfolding_1_triangles, triangle_list_2):
                if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                    return True # they do overlap    
    return False
