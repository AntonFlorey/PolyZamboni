"""
Functions that inspect polyzamboni data and decide if it is still valid
"""

from bpy.types import Mesh
from bmesh.types import BMesh

from . import io
from .cutgraph import compute_all_connected_components
from . import utils

def check_if_polyzamobni_data_exists_and_fits_to_bmesh(mesh : Mesh, bmesh : BMesh):
    if not io.check_if_all_polyzamboni_data_exists(mesh):
        return False
    if not io.check_if_all_polyzamboni_data_is_valid(mesh):
        return False

    mesh_face_indices = set([f.index for f in bmesh.faces])
    mesh_edge_indices = set([e.index for e in bmesh.edges])

    # Constraint edges have to exist
    design_constraints = io.read_edge_constraints_dict(mesh)
    for constraint_edge_index in design_constraints.keys():
        if constraint_edge_index not in mesh_edge_indices:
            return False

    # Check if connected components form a partition of all mesh faces
    connected_components = io.read_connected_component_sets(mesh)
    all_faces_covered_by_components = set()
    for face_set in connected_components.values():
        for face_index in face_set:
            if face_index in all_faces_covered_by_components:
                return False
            all_faces_covered_by_components.add(face_index)
    if mesh_face_indices != all_faces_covered_by_components:
        return False
    
    # Recompute connected components and make sure they match with the stored ones (maybe I can omit this for now ;)
    zamoboni_props = mesh.polyzamboni_general_mesh_props
    connected_components_of_bmesh, _ = compute_all_connected_components(utils.construct_dual_graph_from_bmesh(bmesh), design_constraints, zamoboni_props.use_auto_cuts)
    if frozenset([frozenset(face_set) for face_set in connected_components.values()]) != frozenset([frozenset(face_set) for face_set in connected_components_of_bmesh.values()]):
        return False

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
    