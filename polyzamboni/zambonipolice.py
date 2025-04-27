"""
Functions that inspect polyzamboni data and decide if it is still valid
"""

from bpy.types import Mesh
from bmesh.types import BMesh

from . import io

# TODO Add more checks to this function to avoid more crashes
def check_if_polyzamobni_data_exists_and_fits_to_bmesh(mesh : Mesh, bmesh : BMesh):
    if not io.check_if_all_polyzamboni_data_exists(mesh):
        return False
    if not io.check_if_all_polyzamboni_data_is_valid(mesh):
        return False
    # Recompute connected components and make sure they match with the stored ones (maybe I can omit this for now ;)
    # connected_components_of_bmesh, _ = compute_all_connected_components(utils.construct_dual_graph_from_bmesh(bmesh), design_constraints)
    # if frozenset([frozenset(face_set) for face_set in connected_components.values()]) != frozenset([frozenset(face_set) for face_set in connected_components_of_bmesh.values()]):
    #     return False
    return True

def all_components_have_unfoldings(mesh : Mesh):
    return len(io.read_components_with_cycles_set(mesh)) == 0

def check_if_build_step_numbers_exist_and_make_sense(mesh : Mesh):
    if not io.build_step_numbers_exist(mesh):
        return False
    if not io.connected_components_exist(mesh):
        return False
    if not io.build_step_numbers_valid(mesh):
        return False
    # check content
    build_step_numbers = io.read_build_step_numbers(mesh)
    connected_components = io.read_connected_component_sets(mesh)
    return set(connected_components.keys()).issubset(build_step_numbers.keys()) and set(build_step_numbers.values()) == set(range(1, len(connected_components.keys()) + 1))
    
def check_if_page_numbers_and_transforms_exist_for_all_components(mesh : Mesh):
    """ Assumes that other polyzamboni data exists. """
    if not io.page_numbers_exist(mesh):
        return False
    if not io.page_transforms_exist(mesh):
        return False
    if not io.connected_components_exist(mesh):
        return False
    if not io.components_with_cycles_set_exist(mesh):
        return False
    page_numbers = io.read_page_numbers(mesh)
    page_transforms = io.read_page_transforms(mesh)
    if set(page_numbers.keys()) != set(page_transforms.keys()):
        return False
    cyclic_components = io.read_components_with_cycles_set(mesh)
    connected_components = io.read_connected_component_sets(mesh)
    unfoldable_component_ids = set(connected_components.keys()).difference(cyclic_components)
    return unfoldable_component_ids == set(page_numbers.keys())
