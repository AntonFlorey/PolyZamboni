"""
This file contains all functions that do glueflap stuff.
All functions here are only allowed to read data from meshes but not to write it back!
"""

from bpy.types import Mesh
import bmesh
import numpy as np

from . import io
from . import geometry

def compute_optimal_flap_angle(edge : bmesh.types.BMEdge, flap_angle, flap_height):
    x = flap_height / np.tan(flap_angle)
    h = flap_height
    l = edge.calc_length()
    if l <= 2 * abs(x):
        return np.arctan(2 * h / l)
    return flap_angle

def compute_2d_glue_flap_endpoints_edge_local(edge : bmesh.types.BMEdge, flap_angle_0, flap_angle_1):
    l = edge.calc_length()
    target_0 = np.array([np.cos(flap_angle_0), np.sin(flap_angle_0)])
    target_1 = np.array([-np.cos(flap_angle_1), np.sin(flap_angle_1)]) + np.array([l, 0])
    return target_0, target_1

def compute_2d_glue_flap_triangles_edge_local(edge : bmesh.types.BMEdge, flap_angle, flap_height):
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

def compute_affine_transform_between_touching_components(component_id_1, component_id_2, join_face_index_1, join_face_index_2, join_verts,
                                                         unfolding_affine_transforms, local_coord_system_per_face):
        """ compute affine transformation from unfolding 1 to unfolding 2 """
        join_verts_1 = join_verts
        join_verts_2 = list(reversed(join_verts))
        # translations
        join_point_1 = unfolding_affine_transforms[component_id_1][join_face_index_1] * geometry.to_local_coords(join_verts_1[1].co, *local_coord_system_per_face[join_face_index_1])
        join_point_2 = unfolding_affine_transforms[component_id_2][join_face_index_2] * geometry.to_local_coords(join_verts_2[0].co, *local_coord_system_per_face[join_face_index_2])
        join_point_1_to_orig = geometry.AffineTransform2D(affine_part=-join_point_1)
        orig_to_join_point_2 = geometry.AffineTransform2D(affine_part=join_point_2)
        # rotation
        other_join_point_1 = unfolding_affine_transforms[component_id_1][join_face_index_1] * geometry.to_local_coords(join_verts_1[0].co, *local_coord_system_per_face[join_face_index_1])
        other_join_point_2 = unfolding_affine_transforms[component_id_2][join_face_index_2] * geometry.to_local_coords(join_verts_2[1].co, *local_coord_system_per_face[join_face_index_2])
        x_ax_1, y_ax_1 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_1, other_join_point_1)
        x_ax_2, y_ax_2 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_2, other_join_point_2)
        basis_mat_1 = np.array([x_ax_1, y_ax_1]).T
        basis_mat_2 = np.array([x_ax_2, y_ax_2]).T
        rotate_edges_together = geometry.AffineTransform2D(linear_part=basis_mat_2 @ np.linalg.inv(basis_mat_1))
        # full transformation
        transform_first_unfolding : geometry.AffineTransform2D = orig_to_join_point_2 @ rotate_edges_together @ join_point_1_to_orig
        return transform_first_unfolding

def compute_3d_glue_flap_coords_in_glued_face(bm : bmesh.types.BMesh, component_index, face_index, edge : bmesh.types.BMEdge, halfedge,
                                              opp_component_index, opp_face_index,
                                              glueflap_geometry, unfolding_affine_transforms, local_coord_system_per_face):
    # read the 2d flap triangles
    glueflap_triangles_2d = glueflap_geometry[component_index][edge.index]

    # transform to 3d world coords
    bm.verts.ensure_lookup_table()
    affine_transform_to_opp_unfolding = compute_affine_transform_between_touching_components(component_index, opp_component_index, face_index, opp_face_index, [bm.verts[v_id] for v_id in halfedge], 
                                                                                             unfolding_affine_transforms, local_coord_system_per_face)
    affine_transform_to_opp_local_face_coords = unfolding_affine_transforms[opp_component_index][opp_face_index].inverse() @ affine_transform_to_opp_unfolding
    local_opp_face_coords = local_coord_system_per_face[opp_face_index]
    glueflap_triangles_3d = [tuple([geometry.to_world_coords(affine_transform_to_opp_local_face_coords * coord, *local_opp_face_coords) for coord in triangle]) for triangle in glueflap_triangles_2d]
    return glueflap_triangles_3d

def check_if_edge_of_face_has_glue_flap(edge_index, face_index, glue_flap_halfedge_dict, halfedge_to_face_dict):
    if edge_index not in glue_flap_halfedge_dict.keys():
        return False
    return face_index == halfedge_to_face_dict[glue_flap_halfedge_dict[edge_index]].index

def component_has_overlapping_glue_flaps(mesh : Mesh, component_id, 
                                         glue_flap_collision_dict = None):
    """ Return True if there is any glue flap that overlaps with any other geometry"""
    if glue_flap_collision_dict is None:
        glue_flap_collision_dict = io.read_glue_flap_collisions_dict(mesh)
    if glue_flap_collision_dict is None:
        return False
    for registered_collisions in glue_flap_collision_dict[component_id].values():
        if len(registered_collisions) > 0:
            return True
    return False

def flap_is_overlapping(mesh : Mesh, component_index, edge_index, 
                        glue_flap_collision_dict = None):
    """ Return True if the given glue flap collides with any other"""
    if glue_flap_collision_dict is None:
        glue_flap_collision_dict = io.read_glue_flap_collisions_dict(mesh)
    return edge_index in glue_flap_collision_dict[component_index].keys() and len(glue_flap_collision_dict[component_index][edge_index]) > 0

