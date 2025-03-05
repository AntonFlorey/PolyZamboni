"""
Functions in this file are responsible for computing connected components or provide cutgraph utility.
"""

from bpy.types import Mesh
import bmesh
import networkx as nx
from collections import deque

from . import io
from . import utils

def compute_all_connected_components(dual_graph : nx.Graph, edge_constraints, use_auto_cuts):
    """ Returns connected components as sets and a dict from face indices to component ids. """
    # do stuff
    dual_graph_with_cuts_applied = nx.subgraph_view(dual_graph, filter_edge = lambda v1, v2 : not utils.mesh_edge_is_cut(dual_graph.edges[(v1, v2)]["mesh_edge_index"], edge_constraints, use_auto_cuts))
    all_components = list(nx.connected_components(dual_graph_with_cuts_applied))

    face_to_component_dict = {}
    connected_components = {}
    for c_id, component_faces in enumerate(all_components):
        for f in component_faces:
            face_to_component_dict[f] = c_id
        connected_components[c_id] = component_faces

    return connected_components, face_to_component_dict

def compute_adjacent_connected_component_ids_of_component(mesh : Mesh, component_id, 
                                                          bmesh : bmesh.types.BMesh = None, 
                                                          dual_graph=None, 
                                                          connected_components=None, 
                                                          face_to_component_dict=None):
        # fetch data if not provided
        if dual_graph is None:
            dual_graph = utils.construct_dual_graph_from_bmesh(bmesh) if bmesh is not None else utils.construct_dual_graph_from_mesh(mesh)
        if connected_components is None:
            connected_components, face_to_component_dict = io.read_connected_components(mesh)
        # do the thing
        neighbour_ids = set()
        for component_face_index in connected_components[component_id]:
            for nb_face_index in dual_graph.neighbors(component_face_index):
                nb_component_index = face_to_component_dict[nb_face_index]
                if nb_component_index == component_id:
                    continue
                neighbour_ids.add(nb_component_index)
        return neighbour_ids

def build_order_bfs(mesh : Mesh, starting_component_index, next_free_build_index, visited_components : set[int], build_step_dict : dict,
                    bmesh : bmesh.types.BMesh = None, 
                    dual_graph=None, 
                    connected_components=None, 
                    face_to_component_dict=None):
    component_queue = deque()
    component_queue.append(starting_component_index)
    next_build_index = next_free_build_index
    while component_queue:
        curr_component_id = component_queue.popleft()
        if curr_component_id in visited_components:
            continue
        visited_components.add(curr_component_id)
        build_step_dict[curr_component_id] = next_build_index
        next_build_index += 1
        for nb_component_index in compute_adjacent_connected_component_ids_of_component(mesh, curr_component_id, bmesh, dual_graph, connected_components, face_to_component_dict):
            if nb_component_index in visited_components:
                continue
            component_queue.append(nb_component_index)
    return next_build_index

def compute_build_step_numbers(mesh : Mesh, starting_face_indices, 
                               bmesh : bmesh.types.BMesh = None, 
                               dual_graph=None, 
                               connected_components=None, 
                               face_to_component_dict=None):
    # fetch data if not provided
    if connected_components is None:
        connected_components, face_to_component_dict = io.read_connected_components(mesh)
    # lets gooo

    build_step_dict = {}
    visited_components = set()
    next_build_index = 1
    for starting_face_index in starting_face_indices:
        next_build_index = build_order_bfs(mesh, face_to_component_dict[starting_face_index], next_build_index, visited_components, build_step_dict, 
                                           bmesh, dual_graph, connected_components, face_to_component_dict)
    # some mesh pieces might not be selected. they have to be collected here
    for component_id in connected_components.keys():
        if component_id in visited_components:
            continue
        next_build_index = build_order_bfs(mesh, component_id, next_build_index, visited_components, build_step_dict, 
                                           bmesh, dual_graph, connected_components, face_to_component_dict)
    # sanity check and return
    assert next_build_index == len(connected_components.keys()) + 1
    return build_step_dict

def connected_component_contains_cycles(component_id, dual_graph : nx.Graph, edge_constraints, connected_components, use_auto_cuts):
    face_set = connected_components[component_id]
    def f_in_faceset(f):
        return f in face_set
    def edge_is_not_cut(v1, v2):
        return not utils.mesh_edge_is_cut(dual_graph.edges[(v1, v2)]["mesh_edge_index"], edge_constraints, use_auto_cuts)
    
    subgraph_with_cuts_applied = nx.subgraph_view(dual_graph, filter_node=f_in_faceset, filter_edge = edge_is_not_cut)
    return not nx.is_tree(subgraph_with_cuts_applied)

def compute_cyclic_components(dual_graph : nx.Graph, connected_components, edge_constraints, use_auto_cuts):
    cyclic_components_set = set()
    for component_id in connected_components.keys():
        if connected_component_contains_cycles(component_id, dual_graph, edge_constraints, connected_components, use_auto_cuts):
            cyclic_components_set.add(component_id)
    return cyclic_components_set
