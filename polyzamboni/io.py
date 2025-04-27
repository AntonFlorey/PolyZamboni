"""
This file containts all functions that read data from and write data to a mesh objects ID properties. 
All properties come with validity checks (they do not check if the stored data makes semantic sense!).

"""

from bpy.types import Mesh
import bmesh
import numpy as np
import mathutils
from .geometry import AffineTransform2D
from . import utils

# function collections
_all_removal_functions = []
_all_existence_check_functions = []
_all_validity_check_functions = []

# Helper functions

def serialize_affine_transform(affine_transform : AffineTransform2D):
    res = list(affine_transform.A.flatten()) + list(affine_transform.t)
    assert len(res) == 6
    return res

def parse_affine_transform(affine_transform_as_list):
    assert len(affine_transform_as_list) == 6
    return AffineTransform2D(linear_part=np.array(affine_transform_as_list[:4]).reshape((2,2)), affine_part=np.array(affine_transform_as_list[4:]))

# Edge constraints

def edge_constraints_exist(mesh : Mesh):
    return "polyzamboni_manual_cuts" in mesh and "polyzamboni_locked_edges" in mesh and "polyzamboni_auto_cuts" in mesh
_all_existence_check_functions.append(edge_constraints_exist)

def write_edge_constraints_dict(mesh : Mesh, edge_constraints_dict):
    mesh["polyzamboni_manual_cuts"] = [edge_index for edge_index in edge_constraints_dict if edge_constraints_dict[edge_index] == "cut"]
    mesh["polyzamboni_locked_edges"] = [edge_index for edge_index in edge_constraints_dict if edge_constraints_dict[edge_index] == "glued"]
    mesh["polyzamboni_auto_cuts"] = [edge_index for edge_index in edge_constraints_dict if edge_constraints_dict[edge_index] == "auto"]

def read_edge_constraints_dict(mesh : Mesh):
    """ Returns a dict that maps mesh edge indices to constraint types. If the given mesh has no edge constraint props, returns None. """

    if not edge_constraints_exist(mesh):
        return None
    edge_constraints = {}
    for edge_index in mesh["polyzamboni_manual_cuts"]:
        edge_constraints[edge_index] = "cut"
    for edge_index in mesh["polyzamboni_locked_edges"]:
        edge_constraints[edge_index] = "glued"
    for edge_index in mesh["polyzamboni_auto_cuts"]:
        edge_constraints[edge_index] = "auto"
    return edge_constraints

def read_manual_cut_edges(mesh : Mesh):
    if "polyzamboni_manual_cuts" not in mesh:
        return None
    return mesh["polyzamboni_manual_cuts"]

def read_locked_edges(mesh : Mesh):
    if "polyzamboni_locked_edges" not in mesh:
        return None
    return mesh["polyzamboni_locked_edges"]

def read_auto_cut_edges(mesh : Mesh):
    if "polyzamboni_auto_cuts" not in mesh:
        return None
    return mesh["polyzamboni_auto_cuts"]

def remove_edge_constraints(mesh : Mesh):
    if not edge_constraints_exist(mesh):
        return
    del mesh["polyzamboni_manual_cuts"]
    del mesh["polyzamboni_locked_edges"]
    del mesh["polyzamboni_auto_cuts"]
_all_removal_functions.append(remove_edge_constraints)

def edge_constraints_valid(mesh : Mesh):
    """ Returns True iff edge indices make sense. """

    if not edge_constraints_exist(mesh):
        return True

    num_edges = len(mesh.edges)
    for cut_edge_index in mesh["polyzamboni_manual_cuts"]:
        if cut_edge_index >= num_edges:
            return False
    for locked_edge_index in mesh["polyzamboni_locked_edges"]:
        if locked_edge_index >= num_edges:
            return False
    for auto_cut_edge_index in mesh["polyzamboni_auto_cuts"]:
        if auto_cut_edge_index >= num_edges:
            return False
    return True
_all_validity_check_functions.append(edge_constraints_valid)
# Connected components (Islands)

def connected_components_exist(mesh : Mesh):
    return "polyzamboni_connected_components_lists" in mesh
_all_existence_check_functions.append(connected_components_exist)

def write_connected_components(mesh: Mesh, components_as_sets):
    mesh["polyzamboni_connected_components_lists"] = {str(component_key) : list(component_set) for component_key, component_set in components_as_sets.items()}

def read_connected_components(mesh : Mesh):
    """ Returns None, None if no connected component props exist """
    
    if not connected_components_exist(mesh):
        return (None, None)
    
    components_as_sets = {int(component_id) : set(component_set) for component_id, component_set in mesh["polyzamboni_connected_components_lists"].to_dict().items()}
    face_ids_to_component_ids = {}
    for component_id, component_set in components_as_sets.items():
        for face_index in component_set:
            face_ids_to_component_ids[face_index] = component_id

    return components_as_sets, face_ids_to_component_ids

def read_faces_in_connected_component(mesh : Mesh, component_id):
    if not connected_components_exist(mesh):
        return None
    if str(component_id) not in mesh["polyzamboni_connected_components_lists"]:
        return None
    return set(mesh["polyzamboni_connected_components_lists"][str(component_id)])

def read_connected_component_sets(mesh):
    if not connected_components_exist(mesh):
        return None
    return {int(component_id) : set(component_set) for component_id, component_set in mesh["polyzamboni_connected_components_lists"].to_dict().items()}

def remove_connected_components(mesh : Mesh):
    if connected_components_exist(mesh):
        del mesh["polyzamboni_connected_components_lists"]
_all_removal_functions.append(remove_connected_components)

def connected_components_valid(mesh : Mesh):
    """ Does not check for semantic correctness (if components are separated by cuts) """
    if not connected_components_exist(mesh):
        return True
    
    num_faces = len(mesh.polygons) # why tf is this called polygons !?
    components_as_sets = {int(component_id) : set(component_set) for component_id, component_set in mesh["polyzamboni_connected_components_lists"].to_dict().items()}
    faces_in_components = set()

    # make sure the component indices are compact
    if set(components_as_sets.keys()) != set(range(len(components_as_sets.keys()))):
        return False

    for component_set in components_as_sets.values():
        for face_index in component_set:
            if face_index >= num_faces or face_index in faces_in_components:
                return False
            faces_in_components.add(face_index)
    if len(faces_in_components) != num_faces:
        return False
    return True
_all_validity_check_functions.append(connected_components_valid)

# components with cycles

def components_with_cycles_set_exist(mesh : Mesh):
    return "polyzamboni_cyclic_components" in mesh 
_all_existence_check_functions.append(components_with_cycles_set_exist)

def write_components_with_cycles_set(mesh : Mesh, cyclic_components):
    mesh["polyzamboni_cyclic_components"] = list(cyclic_components)

def read_components_with_cycles_set(mesh : Mesh):
    if not components_with_cycles_set_exist(mesh):
        return None
    return set(mesh["polyzamboni_cyclic_components"])

def remove_components_with_cycles_set(mesh : Mesh):
    if components_with_cycles_set_exist(mesh):
        del mesh["polyzamboni_cyclic_components"]
_all_removal_functions.append(remove_components_with_cycles_set)

def components_with_cycles_valid(mesh : Mesh):
    if not components_with_cycles_set_exist(mesh):
        return True
    if not connected_components_exist(mesh):
        return False # invalid without existing connected components
    cyclic_components_set = read_components_with_cycles_set(mesh)
    components_as_sets, _ = read_connected_components(mesh)
    return cyclic_components_set.issubset(components_as_sets.keys())
_all_validity_check_functions.append(components_with_cycles_valid)

# components with outdated render data per component

def outdated_render_data_exists(mesh : Mesh):
    return "polyzamboni_outdated_render_data" in mesh 
_all_existence_check_functions.append(outdated_render_data_exists)

def write_outdated_render_data(mesh : Mesh, outdated_render_data):
    mesh["polyzamboni_outdated_render_data"] = list(outdated_render_data)

def read_outdated_render_data(mesh : Mesh):
    if not outdated_render_data_exists(mesh):
        return None
    return set(mesh["polyzamboni_outdated_render_data"])

def remove_outdated_render_data(mesh : Mesh):
    if outdated_render_data_exists(mesh):
        del mesh["polyzamboni_outdated_render_data"]
_all_removal_functions.append(remove_outdated_render_data)

def outdated_render_data_valid(mesh : Mesh):
    if not outdated_render_data_exists(mesh):
        return True
    if not connected_components_exist(mesh):
        return False # invalid without existing connected components
    outdated_render_data = read_outdated_render_data(mesh)
    components_as_sets, _ = read_connected_components(mesh)
    return outdated_render_data.issubset(components_as_sets.keys())
_all_validity_check_functions.append(outdated_render_data_valid)

# Step number per connected component

def build_step_numbers_exist(mesh : Mesh):
    return "polyzamboni_build_step_numbers" in mesh 
_all_existence_check_functions.append(build_step_numbers_exist)

def write_build_step_numbers(mesh : Mesh, build_step_numbers):
    mesh["polyzamboni_build_step_numbers"] = {str(component_id) : step_number for component_id, step_number in build_step_numbers.items()}

def read_build_step_numbers(mesh : Mesh):
    if not build_step_numbers_exist(mesh):
        return None
    return {int(component_id) : step_number for component_id, step_number in mesh["polyzamboni_build_step_numbers"].to_dict().items()}

def remove_build_step_numbers(mesh : Mesh):
    if build_step_numbers_exist(mesh):
        del mesh["polyzamboni_build_step_numbers"]
_all_removal_functions.append(remove_build_step_numbers)

def build_step_numbers_valid(mesh : Mesh):
    if not build_step_numbers_exist(mesh):
        return True
    if not connected_components_exist(mesh):
        return True
    return True 
    # I dont know how to handle this correctly
    build_step_numbers = read_build_step_numbers(mesh)
    components_as_sets, _ = read_connected_components(mesh)
    return build_step_numbers.keys() == components_as_sets.keys()
_all_validity_check_functions.append(build_step_numbers_valid)

# Render data per connected component

def component_render_data_exists(mesh : Mesh):
    return "polyzamboni_render_vertices_per_component" in mesh and "polyzamboni_render_triangle_indices_per_component" in mesh
_all_existence_check_functions.append(component_render_data_exists)

def write_all_component_render_data(mesh : Mesh, vertex_positions_per_component, triangle_indices_per_component):
    mesh["polyzamboni_render_vertices_per_component"] = {str(component_id) : [[list(vertex_position_entry[0]), [vertex_position_entry[1]]] for vertex_position_entry in vertex_positions] for component_id, vertex_positions in vertex_positions_per_component.items()}
    mesh["polyzamboni_render_triangle_indices_per_component"] = {str(component_id) : [list(triangle) for triangle in triangle_indices] for component_id, triangle_indices in triangle_indices_per_component.items()}

def read_all_component_render_data(mesh : Mesh):
    if not component_render_data_exists(mesh):
        return None, None
    vertex_positions_per_component = {int(component_id) : [(mathutils.Vector(pos[0]), pos[1][0]) for pos in vertex_positions] for component_id, vertex_positions in mesh["polyzamboni_render_vertices_per_component"].to_dict().items()}
    triangle_indices_per_component = {int(component_id) : [tuple(triangle) for triangle in triangle_indices] for component_id, triangle_indices in mesh["polyzamboni_render_triangle_indices_per_component"].to_dict().items()}
    return vertex_positions_per_component, triangle_indices_per_component

def write_render_data_of_one_component(mesh : Mesh, component_id, vertex_positions, triangle_indices):
    """ Returns True iff write was successful """
    if not component_render_data_exists(mesh):
        return False
    mesh["polyzamboni_render_vertices_per_component"][str(component_id)] = [[list(vertex_position_entry[0]), [vertex_position_entry[1]]] for vertex_position_entry in vertex_positions]
    mesh["polyzamboni_render_triangle_indices_per_component"][str(component_id)] = [list(triangle) for triangle in triangle_indices]
    return True

def remove_component_render_data(mesh : Mesh):
    if component_render_data_exists(mesh):
        del mesh["polyzamboni_render_vertices_per_component"]
        del mesh["polyzamboni_render_triangle_indices_per_component"]
_all_removal_functions.append(remove_component_render_data)

def component_render_data_valid(mesh : Mesh):
    if not component_render_data_exists(mesh):
        return True
    if not connected_components_exist(mesh):
        return False
    vertex_positions, triangle_indices = read_all_component_render_data(mesh)
    if not vertex_positions.keys() == triangle_indices.keys():
        return False
    for component_id in vertex_positions.keys():
        num_verts = len(vertex_positions[component_id])
        for tri in triangle_indices[component_id]:
            for index in tri:
                if index >= num_verts:
                    return False
    return True
_all_validity_check_functions.append(component_render_data_valid)

# set of all components with overlaps

def components_with_overlaps_exist(mesh : Mesh):
    return "polyzamboni_components_with_overlaps" in mesh 
_all_existence_check_functions.append(components_with_overlaps_exist)

def write_components_with_overlaps(mesh : Mesh, components_with_overlaps):
    mesh["polyzamboni_components_with_overlaps"] = list(components_with_overlaps)

def read_components_with_overlaps(mesh : Mesh):
    if not components_with_overlaps_exist(mesh):
        return None
    return set(mesh["polyzamboni_components_with_overlaps"])

def remove_components_with_overlaps(mesh : Mesh):
    if components_with_overlaps_exist(mesh):
        del mesh["polyzamboni_components_with_overlaps"]
_all_removal_functions.append(remove_components_with_overlaps)

def components_with_overlaps_valid(mesh : Mesh):
    if not components_with_overlaps_exist(mesh):
        return True
    if not connected_components_exist(mesh):
        return False # invalid without existing connected components
    components_with_overlaps = read_components_with_overlaps(mesh)
    components_as_sets, _ = read_connected_components(mesh)
    return components_with_overlaps.issubset(components_as_sets.keys())
_all_validity_check_functions.append(components_with_overlaps_valid)

# Local coordinate system per face

def local_coordinate_systems_per_face_exist(mesh : Mesh):
    return "polyzamboni_local_coordinate_systems_per_face" in mesh
_all_existence_check_functions.append(local_coordinate_systems_per_face_exist)

def write_local_coordinate_systems_per_face(mesh : Mesh, local_coordinate_systems_per_face):
    mesh["polyzamboni_local_coordinate_systems_per_face"] = {str(face_id) : coord_system for face_id, coord_system in local_coordinate_systems_per_face.items()}

def read_local_coordinate_systems_per_face(mesh : Mesh):
    if not local_coordinate_systems_per_face_exist(mesh):
        return None
    return {int(face_id) : tuple([np.asarray(vec) for vec in coord_system]) for face_id, coord_system in mesh["polyzamboni_local_coordinate_systems_per_face"].to_dict().items()}

def read_local_coordinate_system_of_face(mesh : Mesh, face_index):
    if not local_coordinate_systems_per_face_exist(mesh) or str(face_index) not in mesh["polyzamboni_local_coordinate_systems_per_face"]:
        return None
    return tuple([np.asarray(vec) for vec in mesh["polyzamboni_local_coordinate_systems_per_face"][str(face_index)]])

def remove_local_coordinate_systems_per_face(mesh : Mesh):
    if local_coordinate_systems_per_face_exist(mesh):
        del mesh["polyzamboni_local_coordinate_systems_per_face"]
_all_removal_functions.append(remove_local_coordinate_systems_per_face)

def local_coordinate_systems_per_face_valid(mesh : Mesh):
    if not local_coordinate_systems_per_face_exist(mesh):
        return True
    num_faces = len(mesh.polygons)
    local_coordinate_systems_per_face = read_local_coordinate_systems_per_face(mesh)
    for face_id in local_coordinate_systems_per_face.keys():
        if face_id >= num_faces:
            return False
    return True
_all_validity_check_functions.append(local_coordinate_systems_per_face_valid)

# Affine transformations from edge system to local coordinate system per face

def inner_face_affine_transforms_exist(mesh : Mesh):
    return "polyzamboni_inner_face_affine_transforms" in mesh
_all_existence_check_functions.append(inner_face_affine_transforms_exist)

def write_inner_face_affine_transforms(mesh : Mesh, inner_face_affine_transforms):
    mesh["polyzamboni_inner_face_affine_transforms"] = {str(face_id) : {str(edge_id) : serialize_affine_transform(transform) for edge_id, transform in transforms_per_edges.items()} for face_id, transforms_per_edges in inner_face_affine_transforms.items()}

def read_inner_face_affine_transforms(mesh : Mesh):
    if not inner_face_affine_transforms_exist(mesh):
        return False
    return {int(face_id) : {int(edge_id) : parse_affine_transform(affine_transform_list) for edge_id, affine_transform_list in transforms_per_edges.items()} for face_id, transforms_per_edges in mesh["polyzamboni_inner_face_affine_transforms"].to_dict().items()}

def read_inner_affine_transform_of_edge_in_face(mesh : Mesh, edge_id, face_id):
    if not inner_face_affine_transforms_exist(mesh):
        return False
    return parse_affine_transform(mesh["polyzamboni_inner_face_affine_transforms"][str(face_id)][str(edge_id)])

def remove_inner_face_affine_transforms(mesh : Mesh):
    if inner_face_affine_transforms_exist(mesh):
        del mesh["polyzamboni_inner_face_affine_transforms"]
_all_removal_functions.append(remove_inner_face_affine_transforms)

def inner_face_affine_transforms_valid(mesh : Mesh):
    if not inner_face_affine_transforms_exist(mesh):
        return True
    bm = bmesh.new()
    bm.from_mesh(mesh)

    inner_face_affine_transforms = read_inner_face_affine_transforms(mesh)
    if set(inner_face_affine_transforms.keys()) != set([face.index for face in bm.faces]):
        bm.free()
        return False

    bm.faces.ensure_lookup_table()
    for face_index, transforms_per_edge in read_inner_face_affine_transforms(mesh).items():
        if set(transforms_per_edge.keys()) != set([edge.index for edge in bm.faces[face_index].edges]):
            bm.free()
            return False
        
    bm.free()
    return True
_all_validity_check_functions.append(inner_face_affine_transforms_valid)

# Affine transformation from face to root face coord system per component

def affine_transforms_to_roots_exist(mesh : Mesh):
    return "polyzamboni_affine_transforms_to_roots" in mesh
_all_existence_check_functions.append(affine_transforms_to_roots_exist)

def write_affine_transforms_to_roots(mesh : Mesh, all_affine_transforms):
    mesh["polyzamboni_affine_transforms_to_roots"] = {str(c_id) : {str(f_id) : serialize_affine_transform(transform) for f_id, transform in transform_per_face.items()} for c_id, transform_per_face in all_affine_transforms.items()}

def read_affine_transforms_to_roots(mesh : Mesh):
    if not affine_transforms_to_roots_exist(mesh):
        return None
    return {int(c_id) : {int(f_id) : parse_affine_transform(transform) for f_id, transform in transform_per_face.items()} for c_id, transform_per_face in mesh["polyzamboni_affine_transforms_to_roots"].to_dict().items()}

def read_affine_transform_to_roots_of_face_in_component(mesh : Mesh, component_index, face_index):
    if not affine_transforms_to_roots_exist(mesh) or str(component_index) not in mesh["polyzamboni_affine_transforms_to_roots"] or str(face_index) not in mesh["polyzamboni_affine_transforms_to_roots"][str(component_index)]:
        return None
    return parse_affine_transform(mesh["polyzamboni_affine_transforms_to_roots"][str(component_index)][str(face_index)])

def remove_affine_transforms_to_roots(mesh : Mesh):
    if affine_transforms_to_roots_exist(mesh):
        del mesh["polyzamboni_affine_transforms_to_roots"]
_all_removal_functions.append(remove_affine_transforms_to_roots)

def affine_transforms_to_roots_valid(mesh : Mesh):
    if not affine_transforms_to_roots_exist(mesh):
        return True
    if not connected_components_exist(mesh):
        return False
    components_as_sets, _ = read_connected_components(mesh)
    affine_transforms_to_roots = read_affine_transforms_to_roots(mesh)
    if not  set(affine_transforms_to_roots.keys()).issubset(components_as_sets.keys()):
        return False
    for component_id, facewise_affine_transformations in affine_transforms_to_roots.items():
        if components_as_sets[component_id] != set(facewise_affine_transformations.keys()):
            return False
    return True
_all_validity_check_functions.append(affine_transforms_to_roots_valid)

# 2d triangles per face per connected component

def facewise_triangles_per_component_exist(mesh : Mesh):
    return "polyzamboni_facewise_triangles_per_component" in mesh
_all_existence_check_functions.append(facewise_triangles_per_component_exist)

def write_facewise_triangles_per_component(mesh : Mesh, facewise_triangles_per_component):
    mesh["polyzamboni_facewise_triangles_per_component"] = {str(component_id) : {str(face_id) : triangles for face_id, triangles in facewise_triangles.items()} for component_id, facewise_triangles in facewise_triangles_per_component.items()}

def write_facewise_triangles_of_one_component(mesh : Mesh, component_id, facewise_triangles):
    mesh["polyzamboni_facewise_triangles_per_component"][str(component_id)] = {str(face_id) : triangles for face_id, triangles in facewise_triangles.items()}

def read_facewise_triangles_per_component(mesh : Mesh):
    if not facewise_triangles_per_component_exist(mesh):
        return None
    return {int(component_id) : {int(face_id) : [tuple([np.array(coord) for coord in tri]) for tri in triangles] for face_id, triangles in facewise_triangles.items()} for component_id, facewise_triangles in mesh["polyzamboni_facewise_triangles_per_component"].to_dict().items()}

def read_facewise_triangles_of_one_component(mesh : Mesh, component_id):
    if not ("polyzamboni_facewise_triangles_per_component" in mesh and str(component_id) in mesh["polyzamboni_facewise_triangles_per_component"]):
        return None
    return {int(face_id) : [tuple([np.array(coord) for coord in tri]) for tri in triangles] for face_id, triangles in mesh["polyzamboni_facewise_triangles_per_component"][str(component_id)].to_dict()}

def remove_facewise_triangles_per_component(mesh : Mesh):
    if facewise_triangles_per_component_exist(mesh):
        del mesh["polyzamboni_facewise_triangles_per_component"]
_all_removal_functions.append(remove_facewise_triangles_per_component)

def facewise_triangles_per_component_valid(mesh : Mesh):
    if not facewise_triangles_per_component_exist(mesh):
        return True
    if not connected_components_exist(mesh):
        return False
    components_as_sets, _ = read_connected_components(mesh)
    facewise_triangles_per_component = read_facewise_triangles_per_component(mesh)
    if not set(facewise_triangles_per_component.keys()).issubset(components_as_sets.keys()):
        return False
    for component_id, facewise_triangles_per_component in facewise_triangles_per_component.items():
        if components_as_sets[component_id] != facewise_triangles_per_component.keys():
            return False
    return True
_all_validity_check_functions.append(facewise_triangles_per_component_valid)

# triangulation indices per face

def triangulation_indices_per_face_exist(mesh : Mesh):
    return "polyzamboni_triangulation_indices" in mesh
_all_existence_check_functions.append(triangulation_indices_per_face_exist)

def write_triangulation_indices_per_face(mesh : Mesh, triangulation_indices_per_face):
    mesh["polyzamboni_triangulation_indices"] = {str(face_id) : triangulation_indices for face_id, triangulation_indices in triangulation_indices_per_face.items()}

def read_triangulation_indices_per_face(mesh : Mesh):
    if not triangulation_indices_per_face_exist(mesh):
        return None
    return {int(face_id) : [tuple(tri_indices) for tri_indices in triangulation_indices] for face_id, triangulation_indices in mesh["polyzamboni_triangulation_indices"].to_dict().items()}

def remove_triangulation_indices_per_face(mesh : Mesh):
    if triangulation_indices_per_face_exist(mesh):
        del mesh["polyzamboni_triangulation_indices"]
_all_removal_functions.append(remove_triangulation_indices_per_face)

def triangulation_indices_per_face_valid(mesh : Mesh):
    if not triangulation_indices_per_face_exist(mesh):
        return True
    triangulation_indices_per_face = read_triangulation_indices_per_face(mesh)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if set([face.index for face in bm.faces]) != triangulation_indices_per_face.keys():
        bm.free()
        return False
    bm.free()
    return True
_all_validity_check_functions.append(triangulation_indices_per_face_valid)

# edge to glueflap haldedge dict

def glueflap_halfedge_dict_exists(mesh : Mesh):
    return "polyzamboni_glueflap_halfedge_dict" in mesh
_all_existence_check_functions.append(glueflap_halfedge_dict_exists)

def write_glueflap_halfedge_dict(mesh : Mesh, glueflap_dict):
    mesh["polyzamboni_glueflap_halfedge_dict"] = {str(edge_id) : halfedge for edge_id, halfedge in glueflap_dict.items()}

def read_glueflap_halfedge_dict(mesh : Mesh):
    if not glueflap_halfedge_dict_exists(mesh):
        return None
    return {int(edge_id) : tuple(halfedge) for edge_id, halfedge in mesh["polyzamboni_glueflap_halfedge_dict"].to_dict().items()}

def remove_glueflap_halfedge_dict(mesh : Mesh):
    if glueflap_halfedge_dict_exists(mesh):
        del mesh["polyzamboni_glueflap_halfedge_dict"]
_all_removal_functions.append(remove_glueflap_halfedge_dict)

def glueflap_halfedge_dict_valid(mesh : Mesh):
    if not glueflap_halfedge_dict_exists(mesh):
        return True
    glueflap_halfedge_dict = read_glueflap_halfedge_dict(mesh)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    for edge_id, halfedge in glueflap_halfedge_dict.items():
        edge_of_halfedge = utils.find_bmesh_edge_of_halfedge(bm, halfedge)
        if edge_of_halfedge is None or edge_of_halfedge.index != edge_id:
            bm.free()
            return False
    bm.free()
    return True
_all_validity_check_functions.append(glueflap_halfedge_dict_valid)

# component-wise glue flap triangles 2D per edge

def glue_flap_geometry_per_edge_per_component_exist(mesh : Mesh):
    return "polyzamboni_glue_flap_triangles_2d" in mesh
_all_existence_check_functions.append(glue_flap_geometry_per_edge_per_component_exist)

def write_glue_flap_geometry_per_edge_per_component(mesh : Mesh, glue_flap_triangles_2d):
    mesh["polyzamboni_glue_flap_triangles_2d"] = {str(c_id) : {str(e_id) : tris for e_id, tris in tris_per_edge.items()} for c_id, tris_per_edge in glue_flap_triangles_2d.items()}

def read_glue_flap_geometry_per_edge_per_component(mesh : Mesh):
    if not glue_flap_geometry_per_edge_per_component_exist(mesh):
        return None
    return {int(c_id) : {int(e_id) : [tuple([np.array(pos) for pos in tri]) for tri in tris] for e_id, tris in tris_per_edge.items()} for c_id, tris_per_edge in mesh["polyzamboni_glue_flap_triangles_2d"].to_dict().items()}

def read_glue_flap_2d_triangles_of_component(mesh : Mesh, component_id):
    if not glue_flap_geometry_per_edge_per_component_exist(mesh):
        return None
    if str(component_id) not in mesh["polyzamboni_glue_flap_triangles_2d"]:
        return None
    return {str(e_id) : tris for e_id, tris in mesh["polyzamboni_glue_flap_triangles_2d"][str(component_id)].to_dict().items()} 

def remove_glue_flap_geometry_per_edge_per_component(mesh : Mesh):
    if glue_flap_geometry_per_edge_per_component_exist(mesh):
        del mesh["polyzamboni_glue_flap_triangles_2d"]
_all_removal_functions.append(remove_glue_flap_geometry_per_edge_per_component)

def glue_flap_geometry_per_edge_per_component_valid(mesh : Mesh):
    if not glue_flap_geometry_per_edge_per_component_exist(mesh):
        return True
    if not connected_components_exist(mesh):
        return False
    glue_flap_triangles_2d = read_glue_flap_geometry_per_edge_per_component(mesh)
    connected_component_sets = read_connected_component_sets(mesh)
    for c_id in glue_flap_triangles_2d.keys():
        if c_id not in connected_component_sets.keys():
            return False
    return True
_all_validity_check_functions.append(glue_flap_geometry_per_edge_per_component_valid)

# colliding glue flap edge id per glue flap edge id 

def glue_flap_collision_dict_exists(mesh : Mesh):
    return "polyzamboni_glue_flap_collisions" in mesh
_all_existence_check_functions.append(glue_flap_collision_dict_exists)

def write_glue_flap_collision_dict(mesh : Mesh, glue_flap_collisions):
    mesh["polyzamboni_glue_flap_collisions"] = {str(c_id) : {str(edge_id) : list(collides_with_edge_ids) for edge_id, collides_with_edge_ids in collision_dict.items()} for c_id, collision_dict in glue_flap_collisions.items()}

def read_glue_flap_collisions_dict(mesh : Mesh):
    if not glue_flap_collision_dict_exists(mesh):
        return None
    return {int(c_id) : {int(edge_id) : set(collides_with_edge_ids) for edge_id, collides_with_edge_ids in collision_dict.items()} for c_id, collision_dict in mesh["polyzamboni_glue_flap_collisions"].to_dict().items()}

def remove_glue_flap_collisions_dict(mesh : Mesh):
    if glue_flap_collision_dict_exists(mesh):
        del mesh["polyzamboni_glue_flap_collisions"]
_all_removal_functions.append(remove_glue_flap_collisions_dict)

def glue_flap_collisions_dict_valid(mesh : Mesh):
    if not glue_flap_collision_dict_exists(mesh):
        return True
    if not connected_components_exist(mesh):
        return False
    glue_flap_collision_dict = read_glue_flap_collisions_dict(mesh)
    connected_component_sets, _ = read_connected_components(mesh)
    for c_id in glue_flap_collision_dict.keys():
        if c_id not in connected_component_sets.keys():
            return False
    # maybe todo: check if colliding edge ids belong to faces in the connected component
    return True
_all_validity_check_functions.append(glue_flap_collisions_dict_valid)

# number of the page the component is printed on (chacks omitted)

def page_numbers_exist(mesh : Mesh):
    return "polyzamboni_page_numbers" in mesh

def write_page_numbers(mesh : Mesh, page_numbers_per_component):
    mesh["polyzamboni_page_numbers"] = {str(component_id) : page_number for component_id, page_number in page_numbers_per_component.items()}

def read_page_numbers(mesh : Mesh):
    if not page_numbers_exist(mesh):
        return None
    return {int(component_id) : page_number for component_id, page_number in mesh["polyzamboni_page_numbers"].to_dict().items()}

def remove_page_numbers(mesh : Mesh):
    if page_numbers_exist(mesh):
        del mesh["polyzamboni_page_numbers"]
_all_removal_functions.append(remove_page_numbers)

# page transforms per component (checks omitted)

def page_transforms_exist(mesh : Mesh):
    return "polyzamboni_page_transforms" in mesh

def write_page_transforms(mesh : Mesh, page_transforms_per_component):
    mesh["polyzamboni_page_transforms"] = {str(component_id) : serialize_affine_transform(page_transform) for component_id, page_transform in page_transforms_per_component.items()}

def read_page_transforms(mesh : Mesh):
    if not page_transforms_exist(mesh):
        return None
    return {int(component_id) : parse_affine_transform(page_transform) for component_id, page_transform in mesh["polyzamboni_page_transforms"].to_dict().items()}

def remove_page_transforms(mesh : Mesh):
    if page_transforms_exist(mesh):
        del mesh["polyzamboni_page_transforms"]
_all_removal_functions.append(remove_page_transforms)

def remove_all_polyzamboni_data(mesh : Mesh):
    for removal_function in _all_removal_functions:
        removal_function(mesh)

def check_if_all_polyzamboni_data_exists(mesh : Mesh):
    """ Only checks if the properties exist, not if they contain any data. """
    for existence_check_function in _all_existence_check_functions:
        if not existence_check_function(mesh):
            return False
    return True

def check_if_all_polyzamboni_data_is_valid(mesh : Mesh):
    for validity_check_function in _all_validity_check_functions:
        if not validity_check_function(mesh):
            print("POLYZAMBONI INFO: Failed validity check: ", validity_check_function)
            return False
    return True

