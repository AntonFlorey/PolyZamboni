"""
This file contains all functions that do glueflap stuff.
All functions here are only allowed to read data from meshes but not to write it back!
"""

from bpy.types import Mesh
import bmesh
import numpy as np

from . import io
from . import geometry


def __compute_2d_glue_flap_triangles_edge_local(edge : bmesh.types.BMEdge, flap_angle, flap_height):
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

def compute_2d_glue_flap_triangles(component_index, face_index, edge : bmesh.types.BMEdge, flap_angle, flap_height, affine_transforms_to_roots, inner_face_affine_transforms):
    triangles_in_local_edge_coords = __compute_2d_glue_flap_triangles_edge_local(edge, flap_angle, flap_height)
    edge_to_root = affine_transforms_to_roots[component_index][face_index] @ inner_face_affine_transforms[face_index][edge.index]
    return [tuple([edge_to_root * local_coord for local_coord in triangle]) for triangle in triangles_in_local_edge_coords]

def compute_3d_glue_flap_triangles_inside_face(mesh : Mesh, face_index, edge : bmesh.types.BMEdge, flap_angle, flap_height,
                                               inner_face_affine_transforms = None, local_coordinate_systems = None):
    triangles_in_local_edge_coords = __compute_2d_glue_flap_triangles_edge_local(edge, flap_angle, flap_height)
    triangles_flipped = [tuple([np.array([local_coord[0], -local_coord[1]]) for local_coord in triangle]) for triangle in triangles_in_local_edge_coords]
    local_coords = io.read_local_coordinate_system_of_face(mesh, face_index) if local_coordinate_systems is None else local_coordinate_systems[face_index]
    edge_to_local_coords = io.read_inner_affine_transform_of_edge_in_face(mesh, edge.index, face_index) if inner_face_affine_transforms is None else inner_face_affine_transforms[face_index][edge.index]
    triangles_in_3d = [tuple(reversed([geometry.to_world_coords(edge_to_local_coords * local_coord, *local_coords) for local_coord in triangle])) for triangle in triangles_flipped]
    return triangles_in_3d

def check_if_edge_has_flap_geometry_attached_to_it(mesh : Mesh, component_index, face_index, edge_index, 
                                                   glue_flap_triangles_2d = None):
    glue_flaps_per_face = io.read_glue_flap_2d_triangles_of_component(mesh, component_index) if glue_flap_triangles_2d is None else glue_flap_triangles_2d[component_index]
    return edge_index in glue_flaps_per_face[face_index].keys()

def _component_has_overlapping_glue_flaps(component_id, glue_flap_collision_dict):
    """ Return True if there is any glue flap that overlaps with any other geometry"""
    for registered_collisions in glue_flap_collision_dict[component_id].values():
        if len(registered_collisions) > 0:
            return True
    return False

# more flexible version
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

