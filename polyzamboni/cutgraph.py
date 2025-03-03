"""
Functions in this file are responsible for computing connected components or provide cutgraph utility.
"""

import bpy
from bpy.types import Mesh
import bmesh
from bmesh.types import BMesh
import numpy as np
import networkx as nx
from collections import deque

from . import io
from . import utils
from .geometry import triangulate_3d_polygon
from .properties import ZamboniGeneralMeshProps

def compute_all_connected_components(mesh : Mesh, use_auto_cuts : bool, edge_constraints=None,
                                     bm : BMesh = None):
    """ Returns connected components as sets and a dict from face indices to component ids. """
    # fetch data if not provided
    if edge_constraints is None:
        edge_constraints = io.read_edge_constraints_dict(mesh)
        assert edge_constraints is not None
    # do stuff
    dual_graph = utils.construct_dual_graph_from_mesh(mesh) if bm is None else utils.construct_dual_graph_from_bmesh(bm)
    dual_graph_with_cuts_applied = nx.subgraph_view(dual_graph, filter_edge = lambda v1, v2 : not utils.mesh_edge_is_cut(dual_graph.edges[(v1, v2)]["edge_index"], edge_constraints, use_auto_cuts))
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

def connected_component_contains_cycles(mesh : Mesh, component_id,
                                        connected_components = None,
                                        dual_graph : nx.Graph = None,
                                        edge_constraints = None):
    zamboni_props : ZamboniGeneralMeshProps = mesh.polyzamboni_general_mesh_props
    face_set = io.read_faces_in_connected_component(mesh, component_id) if connected_components is None else connected_components[component_id]
    def f_in_faceset(f):
        return f in face_set
    def edge_is_not_cut(v1, v2):
        return not utils.mesh_edge_is_cut(dual_graph.edges[(v1, v2)]["edge_index"], edge_constraints, zamboni_props.use_auto_cuts)
    
    subgraph_with_cuts_applied = nx.subgraph_view(dual_graph, filter_node=f_in_faceset, filter_edge = edge_is_not_cut)
    return not nx.is_tree(subgraph_with_cuts_applied)

def compute_cyclic_components(mesh : Mesh,
                              dual_graph : nx.Graph = None,
                              connected_components = None,
                              edge_constraints = None):
    if dual_graph is None:
        dual_graph = utils.construct_dual_graph_from_mesh(mesh)
    if connected_components is None:
        connected_components = io.read_connected_component_sets(mesh)
        assert connected_components is not None
    if edge_constraints is None:
        edge_constraints = io.read_edge_constraints_dict(mesh)
        assert edge_constraints is not None

    cyclic_components_set = set()
    for component_id in connected_components.keys():
        if connected_component_contains_cycles(mesh, component_id, connected_components, dual_graph, edge_constraints):
            cyclic_components_set.add(component_id)
    return cyclic_components_set




class CutGraph():
    """ This class can be attached to any Mesh object """

    def __init__(self, ao : bpy.types.Object, flap_angle = 1 / 4 * np.pi, flap_height = 1.0, prefer_zigzag_flaps = True, use_auto_cuts=False):
        self.flap_angle = flap_angle
        self.flap_height = flap_height
        self.prefer_zigzag = prefer_zigzag_flaps
        self.use_auto_cuts = use_auto_cuts
        self.construct_dual_graph_from_mesh(ao)
        self.designer_constraints = {}
        if CUT_CONSTRAINTS_PROP_NAME in ao:
            for edge_index in ao[CUT_CONSTRAINTS_PROP_NAME]:
                self.designer_constraints[edge_index] = "cut"
        if LOCKED_EDGES_PROP_NAME in ao:
            for edge_index in ao[LOCKED_EDGES_PROP_NAME]:
                self.designer_constraints[edge_index] = "glued"
        if AUTO_CUT_EDGES_PROP_NAME in ao:
            for edge_index in ao[AUTO_CUT_EDGES_PROP_NAME]:
                self.designer_constraints[edge_index] = "auto"
        
        # print("Dual graph constructed")

        # compute connected components
        self.compute_all_connected_components()
        # print("Connected components computed")
        # try to read build steps
        if BUILD_ORDER_PROPERTY_NAME in ao:
            self.read_sparse_build_steps_dict(ao[BUILD_ORDER_PROPERTY_NAME].to_dict())
            # print("Build steps read")
        # compute undoldings
        self.unfold_all_connected_components()
        # print("Components unfolded")
        # try to read glue flaps or recompute them
        if GLUE_FLAP_PROPERTY_NAME in ao:
            self.read_glue_flaps_dict(ao[GLUE_FLAP_PROPERTY_NAME].to_dict())
        if self.check_if_glue_flaps_exist_and_are_valid():
            self.update_all_flap_geometry() # create flap geometry
        else:
            self.greedy_place_all_flaps()


    #################################
    #     Input Mesh Handling       #
    #################################

    def construct_dual_graph_from_mesh(self, obj : bpy.types.Object):
        self.mesh : bmesh.types.BMesh = bmesh.new()
        self.mesh.from_mesh(obj.data)
        self.world_matrix = obj.matrix_world
        self.dualgraph = nx.Graph()
        for face in self.mesh.faces:
            self.dualgraph.add_node(face.index)
        for face in self.mesh.faces:
            curr_id = face.index
            for nb_id, connecting_edge in [(f.index, e) for e in face.edges for f in e.link_faces if f is not face]:
                if (curr_id, nb_id) in self.dualgraph.edges and self.dualgraph.edges[(curr_id, nb_id)]["edge_index"] != connecting_edge.index:
                    print("POLYZAMBONI ERROR: Please ensure that faces connect at at most one common edge!")
                self.dualgraph.add_edge(curr_id, nb_id, edge_index=connecting_edge.index)
        
        # compute and store all face triangulations
        self.face_triangulation_indices = {}
        for face in self.mesh.faces:
            _, triangulation_indices = triangulate_3d_polygon([v.co for v in face.verts], face.normal, [v.index for v in face.verts], crash_on_fail=True)
            self.face_triangulation_indices[face.index] = triangulation_indices

        # compute and store affine transforms for each face
        self.inner_transform_data_per_face = {}
        for face in self.mesh.faces:
            self.inner_transform_data_per_face[face.index] = InnerFaceTransforms(face)

        # compute offsets for user feedback region drawing
        self.min_edge_len_per_face = {f.index : min([e.calc_length() for e in f.edges]) for f in self.mesh.faces}
        min_edge_len = min([e.calc_length() for e in self.mesh.edges])
        self.small_offset = 0.05 * min_edge_len
        self.large_offset = 0.35 * min_edge_len

        self.halfedge_to_edge = None # delete old halfedge to edge dict
        self.halfedge_to_face = None # delete old halfedge to face dict
        self.valid = True # make sure we check mesh validity again
        if hasattr(self, "connected_components"):
            current_connected_component : ConnectedComponent
            for current_connected_component in self.connected_components.values():
                current_connected_component._render_triangles_outdated = True

    #################################
    #             Misc.             #
    #################################

    #################################
    #      Connected Components     #
    #################################

    #################################
    #           Unfolding           #
    #################################

    #################################
    #           Rendering           #
    #################################

    #################################
    #  User Constraints Interface   #
    #################################

    #################################
    #           Exporting           #
    #################################
