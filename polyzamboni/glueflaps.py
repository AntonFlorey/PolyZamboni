"""
This file contains all functions that modify the glue flaps of a paper model
"""

from bpy.types import Mesh

from . import io

def component_has_overlapping_glue_flaps(mesh : Mesh, component_id, glue_flap_collision_dict = None):
    """ Return True if there is any glue flap that overlaps with any other geometry"""
    if glue_flap_collision_dict is None:
        glue_flap_collision_dict = io.read_glue_flap_collisions_dict(mesh)
    if glue_flap_collision_dict is None:
        return False
    for registered_collisions in glue_flap_collision_dict[component_id].values():
        if len(registered_collisions) > 0:
            return True
    return False

def flap_is_overlapping(mesh : Mesh, component_index, edge_index, glue_flap_collision_dict = None):
    """ Return True if the given glue flap collides with any other"""
    if glue_flap_collision_dict is None:
        glue_flap_collision_dict = io.read_glue_flap_collisions_dict(mesh)
    return edge_index in glue_flap_collision_dict[component_index].keys() and len(glue_flap_collision_dict[component_index][edge_index]) > 0
