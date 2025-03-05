"""
This file contains several high level functions that meaningful combine multiple polyzamboni operations.
"""

import bpy
from bpy.types import Mesh
import bmesh
from bmesh.types import BMesh
import numpy as np
import networkx as nx
from collections import deque
import time

from . import utils
from . import io
from . import geometry
from . import cutgraph
from . import unfolding
from . import glueflaps
from . import zambonipolice

#################################
#    Init, delete and update    #
#################################

def get_indices_of_multi_touching_faces(bm : BMesh):
    face_pair_set = set()
    multi_touching_face_indices = set()
    for edge in bm.edges:
        if edge.is_boundary:
            continue
        assert len(edge.link_faces) == 2
        face_pair_key = tuple(sorted([f.index for f in edge.link_faces]))
        if face_pair_key in face_pair_set:
            for f in edge.link_faces:
                multi_touching_face_indices.add(f.index)
        face_pair_set.add(face_pair_key)
    return multi_touching_face_indices

def get_indices_of_not_triangulatable_faces(bm : BMesh):
    non_triangulatable_faces = set()
    for face in bm.faces:
        _, tri_ids = geometry.triangulate_3d_polygon([v.co for v in face.verts], face.normal)
        if len(tri_ids) != len(face.verts) - 2:
            non_triangulatable_faces.add(face.index)
    return non_triangulatable_faces

def initialize_precomputable_bmesh_data(mesh: Mesh, bm : BMesh):
    # compute all face triangulations and store them
    io.write_triangulation_indices_per_face(mesh, unfolding.compute_all_face_triangulation_indices(bm))
    # compute and store affine transforms for each face
    facewise_coordinate_systems = {}
    facewise_transitions_per_edge = {}
    for bm_face in bm.faces:
        coord_system, transitions = unfolding.compute_local_coordinate_system_with_all_transitions_to_it(bm, bm_face)
        facewise_coordinate_systems[bm_face.index] = coord_system
        facewise_transitions_per_edge[bm_face.index] = transitions
    io.write_local_coordinate_systems_per_face(mesh, facewise_coordinate_systems)
    io.write_inner_face_affine_transforms(mesh, facewise_transitions_per_edge)

def initialize_paper_model(mesh : Mesh):
    zamboni_props = mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(mesh)
    dual_graph = utils.construct_dual_graph_from_bmesh(bm)

    initialize_precomputable_bmesh_data(mesh, bm)

    # edge_constraints
    io.write_edge_constraints_dict(mesh, {})
    
    # connected_components
    connected_components, face_to_component_dict = cutgraph.compute_all_connected_components(dual_graph,{}, False)
    io.write_connected_components(mesh, connected_components)
    io.write_next_free_component_index(mesh, len(connected_components.keys()))
    io.write_build_step_numbers(mesh, cutgraph.compute_build_step_numbers(mesh, [], bm, dual_graph, connected_components, face_to_component_dict))
    cyclic_components = cutgraph.compute_cyclic_components(dual_graph, connected_components, {}, zamboni_props.use_auto_cuts)
    io.write_components_with_cycles_set(mesh, cyclic_components)

    # unfoldings
    unfolded_triangles, affine_transform_to_root, components_with_overlaps = unfolding.compute_unfolding_data_of_all_components(mesh, bm, dual_graph, 
                                                                                                                                {}, 
                                                                                                                                connected_components, 
                                                                                                                                cyclic_components,
                                                                                                                                io.read_triangulation_indices_per_face(mesh),
                                                                                                                                io.read_inner_face_affine_transforms(mesh),
                                                                                                                                io.read_local_coordinate_systems_per_face(mesh))
    io.write_facewise_triangles_per_component(mesh, unfolded_triangles)
    io.write_affine_transforms_to_roots(mesh, affine_transform_to_root)
    io.write_components_with_overlaps(mesh, components_with_overlaps)

    # glue flaps
    halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bm)
    glueflap_dict, glueflap_geometry, glueflap_collisions, _ = _greedy_place_all_flaps(bm, zamboni_props.glue_flap_angle, zamboni_props.glue_flap_height, zamboni_props.use_auto_cuts,
                                                                                       halfedge_to_face_dict, face_to_component_dict, affine_transform_to_root, 
                                                                                       io.read_inner_face_affine_transforms(mesh), unfolded_triangles,
                                                                                       {}, cyclic_components, zamboni_props.prefer_alternating_flaps)
    io.write_glueflap_halfedge_dict(mesh, glueflap_dict)
    io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, glueflap_geometry)
    io.write_glue_flap_collision_dict(mesh, glueflap_collisions)

    # render data
    io.write_all_component_render_data(mesh, {c_id : [] for c_id in connected_components.keys()}, {c_id : [] for c_id in connected_components.keys()}) # empty
    io.write_outdated_render_data(mesh, set(connected_components.keys()))

    # set zamboni props
    zamboni_props.has_attached_paper_model = True
    zamboni_props.attached_paper_model_data_valid = True

    # sanity check
    assert zambonipolice.check_if_polyzamobni_data_exists_and_fits_to_bmesh(mesh, bm)
    bm.free()

def delete_paper_model(mesh : Mesh):
    zamboni_props = mesh.polyzamboni_general_mesh_props
    zamboni_props.has_attached_paper_model = False
    io.remove_all_polyzamboni_data(mesh)

def sync_paper_model_with_mesh_geometry(mesh : Mesh):
    zamboni_props = mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(mesh)

    if zambonipolice.check_if_polyzamobni_data_exists_and_fits_to_bmesh(mesh, bm):
        zamboni_props.attached_paper_model_data_valid = True

        # get new geometrical data
        initialize_precomputable_bmesh_data(mesh, bm)

        edge_constraints = io.read_edge_constraints_dict(mesh)

        connected_components, face_ids_to_component_ids = io.read_connected_components(mesh)

        cyclic_components = cutgraph.compute_cyclic_components(dual_graph, connected_components, edge_constraints, zamboni_props.use_auto_cuts)
        io.write_components_with_cycles_set(mesh, cyclic_components)

        # recompute unfoldings
        dual_graph = utils.construct_dual_graph_from_bmesh(bm)
        unfolded_triangles, affine_transform_to_root, components_with_overlaps = unfolding.compute_unfolding_data_of_all_components(mesh, bm, dual_graph, 
                                                                                                                                     {}, 
                                                                                                                                     connected_components, 
                                                                                                                                     io.read_components_with_cycles_set(mesh),
                                                                                                                                     io.read_triangulation_indices_per_face(mesh),
                                                                                                                                     io.read_inner_face_affine_transforms(mesh),
                                                                                                                                     io.read_local_coordinate_systems_per_face(mesh))
        io.write_facewise_triangles_per_component(mesh, unfolded_triangles)
        io.write_affine_transforms_to_roots(mesh, affine_transform_to_root)
        io.write_components_with_overlaps(mesh, components_with_overlaps)

        # recompute glue flap geometry
        halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bm)
        glueflap_dict = io.read_glueflap_halfedge_dict(mesh)
        glueflap_geometry = io.read_glue_flap_2d_triangles_per_edge_per_face(mesh)
        glueflap_collisions = io.read_glue_flap_collisions_dict(mesh)
        _update_all_flap_geometry(bm, zamboni_props.glue_flap_angle, zamboni_props.glue_flap_height, halfedge_to_face_dict, face_ids_to_component_ids, affine_transform_to_root,
                                  io.read_inner_face_affine_transforms(mesh), unfolded_triangles, glueflap_dict, glueflap_geometry, glueflap_collisions)
        io.write_glueflap_halfedge_dict(mesh, glueflap_dict)
        io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, glueflap_geometry)
        io.write_glue_flap_collision_dict(mesh, glueflap_collisions)

        # mark all render data as outdated
        io.write_outdated_render_data(mesh, set(connected_components.keys()))

    else:
        # I dont know what I want to do here
        zamboni_props.attached_paper_model_data_valid = False

    bm.free()

#################################
#   Cut, clear and glue edges   #
#################################

class DataForEditing():
    def __init__(self, bm : BMesh,
                 use_auto_cuts,
                 zigzag_flaps,
                 flap_height,
                 flap_angle,
                 edge_constraints,
                 connected_components,
                 face_to_component_dict,
                 next_free_component_index,
                 cyclic_components,
                 outdated_components,
                 build_step_numbers,
                 overlapping_components,
                 local_coords_per_face,
                 inner_affine_transforms,
                 affine_transforms_to_root,
                 unfolded_triangles,
                 triangulation_indices_per_face,
                 glueflap_dict,
                 glueflap_geometry,
                 glueflap_collisions):
        # read all required data from the mesh
        self.bm = bm
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        self.dual_graph = utils.construct_dual_graph_from_bmesh(bm)
        self.halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bm)
        self.use_auto_cuts = use_auto_cuts
        self.zigzag_flaps = zigzag_flaps
        self.flap_height = flap_height
        self.flap_angle = flap_angle 
        self.edge_constraints = edge_constraints
        self.connected_components = connected_components
        self.face_to_component_dict = face_to_component_dict
        self.next_free_component_index = next_free_component_index
        self.cyclic_components = cyclic_components
        self.outdated_components = outdated_components
        self.build_step_numbers = build_step_numbers
        self.overlapping_components = overlapping_components
        self.local_coords_per_face = local_coords_per_face
        self.inner_affine_transforms = inner_affine_transforms
        self.affine_transforms_to_root = affine_transforms_to_root
        self.unfolded_triangles = unfolded_triangles
        self.triangulation_indices_per_face = triangulation_indices_per_face
        self.glueflap_dict = glueflap_dict
        self.glueflap_geometry = glueflap_geometry
        self.glueflap_collisions = glueflap_collisions

    @classmethod 
    def from_mesh(cls, mesh : Mesh):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        zamobni_props = mesh.polyzamboni_general_mesh_props
        connected_components, face_to_component_dict = io.read_connected_components(mesh)
        return cls(bm,
                   zamobni_props.use_auto_cuts,
                   zamobni_props.prefer_alternating_flaps,
                   zamobni_props.glue_flap_height,
                   zamobni_props.glue_flap_angle,
                   io.read_edge_constraints_dict(mesh),
                   connected_components,
                   face_to_component_dict,
                   io.read_next_free_component_index(mesh),
                   io.read_components_with_cycles_set(mesh),
                   io.read_outdated_render_data(mesh),
                   io.read_build_step_numbers(mesh),
                   io.read_components_with_overlaps(mesh),
                   io.read_local_coordinate_systems_per_face(mesh),
                   io.read_inner_face_affine_transforms(mesh),
                   io.read_affine_transforms_to_roots(mesh),
                   io.read_facewise_triangles_per_component(mesh),
                   io.read_triangulation_indices_per_face(mesh),
                   io.read_glueflap_halfedge_dict(mesh),
                   io.read_glue_flap_2d_triangles_per_edge_per_face(mesh),
                   io.read_glue_flap_collisions_dict(mesh))

    def write_back_my_data(self, mesh : Mesh):
        # write back all read data (with some exceptions)
        io.write_edge_constraints_dict(mesh, self.edge_constraints)
        io.write_connected_components(mesh, self.connected_components)
        io.write_next_free_component_index(mesh, self.next_free_component_index)
        io.write_components_with_cycles_set(mesh, self.cyclic_components)
        io.write_outdated_render_data(mesh, self.outdated_components)
        io.write_build_step_numbers(mesh, self.build_step_numbers)
        io.write_components_with_overlaps(mesh, self.overlapping_components)
        io.write_affine_transforms_to_roots(mesh, self.affine_transforms_to_root)
        io.write_facewise_triangles_per_component(mesh, self.unfolded_triangles)
        io.write_glueflap_halfedge_dict(mesh, self.glueflap_dict)
        io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, self.glueflap_geometry)
        io.write_glue_flap_collision_dict(mesh, self.glueflap_collisions)
        self.bm.free()

def _update_paper_model_around_edges(editing_data : DataForEditing, indices_of_touched_edges):
    # connected components
    if len(indices_of_touched_edges) <= 5: #some random number lol
        for edge_index in indices_of_touched_edges:
            next_free_id = _update_connected_components_around_edge(editing_data.bm, editing_data.dual_graph, edge_index, editing_data.connected_components, 
                                                                    editing_data.face_to_component_dict, editing_data.cyclic_components, editing_data.outdated_components, 
                                                                    editing_data.overlapping_components, editing_data.edge_constraints, editing_data.use_auto_cuts, 
                                                                    editing_data.next_free_component_index)
            editing_data.next_free_component_index = next_free_id
    else:
        connected_components, cyclic_components, next_free_id = _update_all_connected_components(editing_data.dual_graph, editing_data.edge_constraints, editing_data.use_auto_cuts)
        _sanitize_all_sets_storing_component_ids(editing_data.connected_components, editing_data.cyclic_components, editing_data.outdated_components, editing_data.overlapping_components)
        editing_data.connected_components = connected_components
        editing_data.cyclic_components = cyclic_components
        editing_data.next_free_component_index = next_free_id

    # unfoldings
    _update_unfoldings_along_edges(editing_data.bm, editing_data.dual_graph, indices_of_touched_edges, editing_data.edge_constraints, editing_data.use_auto_cuts, editing_data.connected_components,
                                   editing_data.face_to_component_dict, editing_data.cyclic_components, editing_data.face_to_component_dict, editing_data.inner_affine_transforms, 
                                   editing_data.local_coords_per_face, editing_data.unfolded_triangles, editing_data.affine_transforms_to_root, editing_data.overlapping_components)
    
    # glue flaps
    _greedy_update_flaps_around_changed_components(editing_data.bm, indices_of_touched_edges, editing_data.flap_angle, editing_data.flap_height, editing_data.use_auto_cuts, editing_data.halfedge_to_face_dict,
                                                   editing_data.face_to_component_dict, editing_data.connected_components, editing_data.affine_transforms_to_root, editing_data.inner_affine_transforms,
                                                   editing_data.unfolded_triangles, editing_data.edge_constraints, editing_data.cyclic_components, editing_data.zigzag_flaps, editing_data.glueflap_dict,
                                                   editing_data.glueflap_geometry, editing_data.glueflap_collisions)

    # render data
    _mark_components_along_edges_as_outdated(editing_data.bm, editing_data.face_to_component_dict, indices_of_touched_edges, editing_data.outdated_components)

def _update_paper_model(editing_data : DataForEditing):
    # connected components
    connected_components, cyclic_components, next_free_id = _update_all_connected_components(editing_data.dual_graph, editing_data.edge_constraints, editing_data.use_auto_cuts)
    _sanitize_all_sets_storing_component_ids(editing_data.connected_components, editing_data.cyclic_components, editing_data.outdated_components, editing_data.overlapping_components)
    editing_data.connected_components = connected_components
    editing_data.cyclic_components = cyclic_components
    editing_data.next_free_component_index = next_free_id

    # unfoldings
    uf_geometry, affine_to_root, overlapping_components = unfolding.compute_unfolding_data_of_all_components(editing_data.bm, editing_data.dual_graph, editing_data.edge_constraints, editing_data.connected_components,
                                                                                                             editing_data.cyclic_components, editing_data.triangulation_indices_per_face, editing_data.inner_affine_transforms,
                                                                                                             editing_data.local_coords_per_face, editing_data.use_auto_cuts)
    editing_data.unfolded_triangles = uf_geometry
    editing_data.affine_transforms_to_root = affine_to_root
    editing_data.overlapping_components = overlapping_components

    # glue flaps
    glueflap_dict, glueflap_geometry, glueflap_collisions, _ = _greedy_place_all_flaps(editing_data.bm, editing_data.flap_angle, editing_data.flap_height, editing_data.use_auto_cuts, editing_data.halfedge_to_face_dict,
                                                                                       editing_data.face_to_component_dict, editing_data.affine_transforms_to_root, editing_data.inner_affine_transforms,
                                                                                       editing_data.unfolded_triangles, editing_data.edge_constraints, editing_data.cyclic_components, editing_data.zigzag_flaps)     
    editing_data.glueflap_dict = glueflap_dict
    editing_data.glueflap_geometry = glueflap_geometry
    editing_data.glueflap_collisions = glueflap_collisions

    # render data
    _mark_all_components_as_outdated(editing_data.bm, editing_data.connected_components, editing_data.outdated_components)

def update_paper_model(mesh : Mesh):
    # read all needed data
    editing_data = DataForEditing.from_mesh(mesh)
    _update_paper_model(editing_data)
    editing_data.write_back_my_data(mesh)

def _write_edge_constraints(editing_data : DataForEditing, indices_of_touched_edges, constraint : str):    
    # add cuts to constraints
    for edge_index in indices_of_touched_edges:
        editing_data.edge_constraints[edge_index] = constraint
    
    # update the paper model
    _update_paper_model_around_edges(editing_data, indices_of_touched_edges)

def _clear_edge_constraints(editing_data : DataForEditing, indices_of_touched_edges):    
    # add cuts to constraints
    for edge_index in indices_of_touched_edges:
        if edge_index in editing_data.edge_constraints:
            del editing_data.edge_constraints[edge_index]
    
    # update the paper model
    _update_paper_model_around_edges(editing_data, indices_of_touched_edges)

def cut_edges(mesh : Mesh, indices_of_edges_to_cut):
    # read all needed data
    read_time_start = time.time()
    editing_data = DataForEditing.from_mesh(mesh)
    print("reading paper model data took", time.time() - read_time_start, "seconds")

    # do the cutting operation
    working_time_start = time.time()
    _write_edge_constraints(editing_data, indices_of_edges_to_cut, "cut")
    print("working on the paper model data took", time.time() - working_time_start, "seconds")

    # write back all data
    write_time_start = time.time()
    editing_data.write_back_my_data(mesh)
    print("writing paper model data took", time.time() - write_time_start, "seconds")

def glue_edges(mesh : Mesh, indices_of_edges_to_lock):
    # read all needed data
    editing_data = DataForEditing.from_mesh(mesh)

    # do the cutting operation
    _write_edge_constraints(editing_data, indices_of_edges_to_lock, "glued")

    # write back all data
    editing_data.write_back_my_data(mesh)

def clear_edges(mesh : Mesh, indices_of_edges_to_clear):
    # read all needed data
    editing_data = DataForEditing.from_mesh(mesh)

    # do the cutting operation
    _clear_edge_constraints(editing_data, indices_of_edges_to_clear)

    # write back all data
    editing_data.write_back_my_data(mesh)

def add_cutout_region(mesh : Mesh, face_indices):
    # read all needed data
    editing_data = DataForEditing.from_mesh(mesh)

    face_set = set(face_indices)
    edges = set()
    for f_id in face_indices:
        edges = edges.union([e.index for e in editing_data.bm.faces[f_id].edges])
    for e_id in edges:
        cut_this_edge = False
        for f in editing_data.bm.edges[e_id].link_faces:
            if f.index not in face_set:
                cut_this_edge = True
                break
        if cut_this_edge:
            editing_data.edge_constraints[e_id] = "cut"
        else:
            editing_data.edge_constraints[e_id] = "glued"

    _update_paper_model_around_edges(editing_data, list(edges))

    editing_data.write_back_my_data(mesh)

def add_cuts_between_different_materials(mesh : Mesh):
    # read all needed data
    editing_data = DataForEditing.from_mesh(mesh)

    edges_cut = []
    for mesh_edge in editing_data.bm.edges:
        if mesh_edge.is_boundary:
            continue
        face_0 = mesh_edge.link_faces[0]
        face_1 = mesh_edge.link_faces[1]
        if face_0.material_index != face_1.material_index:
            editing_data.edge_constraints[mesh_edge.index] = "cut"
            edges_cut.append(mesh_edge.index)
    
    _update_paper_model_around_edges(editing_data, edges_cut)

    editing_data.write_back_my_data(mesh)

def compute_build_step_numbers(mesh : Mesh, selected_start_face_ids):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    dual_graph = utils.construct_dual_graph_from_bmesh(bm)
    connected_components, face_to_component_dict = io.read_connected_components(mesh)
    build_numbers_dict = cutgraph.compute_build_step_numbers(mesh, selected_start_face_ids, bm, dual_graph, connected_components, face_to_component_dict)
    io.write_build_step_numbers(mesh, build_numbers_dict)
    bm.free()

#################################
#     Updates after edits       #
#################################

def _sanitize_all_sets_storing_component_ids(connected_components, cyclic_components : set, outdated_components : set, overlapping_components : set):
    for c_id in cyclic_components:
        if c_id not in connected_components:
            cyclic_components.remove(c_id)
    for c_id in outdated_components:
        if c_id not in connected_components:
            outdated_components.remove(c_id)
    for c_id in overlapping_components:
        if c_id not in connected_components:
            overlapping_components.remove(c_id)

def sanitize_all_sets_storing_component_ids(mesh : Mesh):
    connected_components = io.read_connected_component_sets(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    outdated_components = io.read_outdated_render_data(mesh)
    overlapping_components = io.read_components_with_overlaps(mesh)
    _sanitize_all_sets_storing_component_ids(connected_components, cyclic_components, outdated_components, overlapping_components)
    io.write_components_with_cycles_set(mesh, cyclic_components)
    io.write_outdated_render_data(mesh, outdated_components)
    io.write_components_with_overlaps(mesh, overlapping_components)

class AllComponentData():
    def __init__(self, connected_components = {}, 
                 cyclic_components = set(), 
                 outdated_components = set(), 
                 step_numbers = {}, 
                 vertex_positions_per_component = {}, 
                 triangle_indices_per_component = {}, 
                 overlapping_components = set(), 
                 affine_transforms_to_root = {}, 
                 unfolded_triangles = {}, 
                 glueflap_geometry = {}, 
                 glueflap_collisions = {}):
        self.connected_components = connected_components
        self.cyclic_components = cyclic_components
        self.outdated_components = outdated_components
        self.step_numbers = step_numbers
        self.vertex_positions_per_component = vertex_positions_per_component
        self.triangle_indices_per_component = triangle_indices_per_component
        self.overlapping_components = overlapping_components
        self.affine_transforms_to_root = affine_transforms_to_root
        self.unfolded_triangles = unfolded_triangles
        self.glueflap_geometry = glueflap_geometry
        self.glueflap_collisions = glueflap_collisions

def _compactify(component_data : AllComponentData):
    """ This hurts to write """
    compactified_component_data = AllComponentData()

    component_key_map = {}
    for new_c_id, old_c_id in enumerate(component_data.connected_components.keys()):
        component_key_map[old_c_id] = new_c_id
        compactified_component_data.connected_components[new_c_id] = component_data.connected_components[old_c_id]
        if old_c_id in component_data.step_numbers:
            compactified_component_data.step_numbers[new_c_id] = component_data.step_numbers[old_c_id]
        if old_c_id in component_data.vertex_positions_per_component:
            compactified_component_data.vertex_positions_per_component[new_c_id] = component_data.vertex_positions_per_component[old_c_id]
            compactified_component_data.triangle_indices_per_component[new_c_id] = component_data.triangle_indices_per_component[old_c_id]
        if old_c_id not in component_data.cyclic_components:
            compactified_component_data.affine_transforms_to_root[new_c_id] = component_data.affine_transforms_to_root[old_c_id]
            compactified_component_data.unfolded_triangles[new_c_id] = component_data.unfolded_triangles[old_c_id]
        if old_c_id in component_data.glueflap_geometry:
            compactified_component_data.glueflap_geometry[new_c_id] = component_data.glueflap_geometry[old_c_id]
            compactified_component_data.glueflap_collisions[new_c_id] = component_data.glueflap_collisions[old_c_id]
    
    # copy sets
    compactified_component_data.cyclic_components = set([component_key_map[cyclic_c_id] for cyclic_c_id in component_data.cyclic_components])
    compactified_component_data.outdated_components = set([component_key_map[outdated_c_id] for outdated_c_id in component_data.outdated_components])
    compactified_component_data.overlapping_components = set([component_key_map[overlapping_c_id] for overlapping_c_id in component_data.overlapping_components])

    next_free_component_id = len(compactified_component_data.connected_components.keys())

    return compactified_component_data, next_free_component_id

def compactify_polyzamboni_data(mesh : Mesh):
    """ This may take some time so call this rarely """
    # read all current polyzamboni data
    component_data = AllComponentData(io.read_connected_component_sets(mesh),
                                      io.read_components_with_cycles_set(mesh),
                                      io.read_outdated_render_data(mesh),
                                      io.read_build_step_numbers(mesh),
                                      io.read_all_component_render_data(mesh)[0],
                                      io.read_all_component_render_data(mesh)[1],
                                      io.read_components_with_overlaps(mesh),
                                      io.read_affine_transforms_to_roots(mesh),
                                      io.read_facewise_triangles_per_component(mesh),
                                      io.read_glue_flap_2d_triangles_per_edge_per_face(mesh),
                                      io.read_glue_flap_collisions_dict(mesh))
    compactified_data, next_free_component_id = _compactify(component_data)
    # write compactified data back to the mesh
    io.write_connected_components(mesh, compactified_data.connected_components)
    io.write_components_with_cycles_set(mesh, compactified_data.cyclic_components)
    io.write_outdated_render_data(mesh, compactified_data.outdated_components)
    io.write_build_step_numbers(mesh, compactified_data.step_numbers)
    io.write_all_component_render_data(mesh, compactified_data.vertex_positions_per_component, compactified_data.triangle_indices_per_component)
    io.write_components_with_overlaps(mesh, compactified_data.overlapping_components)
    io.write_affine_transforms_to_roots(mesh, compactified_data.affine_transforms_to_root)
    io.write_facewise_triangles_per_component(mesh, compactified_data.unfolded_triangles)
    io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, compactified_data.glueflap_geometry)
    io.write_glue_flap_collision_dict(mesh, compactified_data.glueflap_collisions)
    io.write_next_free_component_index(mesh, next_free_component_id)

def _update_all_connected_components_and_preserve_old_indices(dual_graph : nx.Graph, edge_constraints, use_auto_cuts, old_face_do_component_dict):
    """ Do NOT forget to sanitize all sets containing component ids afterwards! """
    connected_components, _ = cutgraph.compute_all_connected_components(dual_graph, edge_constraints, use_auto_cuts)
    # do the re-indexing
    re_indexed_components = {}
    for face_set in connected_components.values():
        assert len(face_set) > 0
        re_indexed_components[old_face_do_component_dict[next(iter(face_set))]] = face_set
    cyclic_components = cutgraph.compute_cyclic_components(dual_graph, re_indexed_components, edge_constraints, use_auto_cuts)
    return re_indexed_components, cyclic_components, max(re_indexed_components.keys()) + 1

def _update_all_connected_components(dual_graph : nx.Graph, edge_constraints, use_auto_cuts):
    """ Do NOT forget to sanitize all sets containing component ids afterwards! """
    connected_components, _ = cutgraph.compute_all_connected_components(dual_graph, edge_constraints, use_auto_cuts)
    cyclic_components = cutgraph.compute_cyclic_components(dual_graph, connected_components, edge_constraints, use_auto_cuts)
    return connected_components, cyclic_components, len(connected_components.keys())

def update_all_connected_components(mesh : Mesh, preserve_old_indices=True):
    """ Most data stored for the new components might still be usable. If this is NOT the case, set preserve_old_indices to False to save some time. """
    zamboni_props = mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(mesh)
    edge_constraints = io.read_edge_constraints_dict(mesh)
    dual_graph = utils.construct_dual_graph_from_bmesh(bm)

    if preserve_old_indices:
        _, current_face_to_component_dict = io.read_connected_components(mesh)
        connected_components, cyclic_components, next_free_component_id = _update_all_connected_components_and_preserve_old_indices(dual_graph, edge_constraints, zamboni_props.use_auto_cuts, current_face_to_component_dict)
    else:
        connected_components, cyclic_components, next_free_component_id = _update_all_connected_components(dual_graph, edge_constraints, zamboni_props.use_auto_cuts)
    
    io.write_connected_components(connected_components)
    io.write_components_with_cycles_set(cyclic_components)
    io.write_next_free_component_index(next_free_component_id)
    sanitize_all_sets_storing_component_ids(mesh)

    bm.free()

def _update_connected_components_around_edge(bm : BMesh, dual_graph : nx.Graph, edge_index, connected_components, face_to_component_dict, 
                                             cyclic_components : set, outdated_components : set, overlapping_components : set,
                                             edge_constraints, use_auto_cuts, next_free_component_id):
    """ This function will change the contents of 'connected_components', 'face_to_component_dict', 'cyclic_components', 'outdated_components', 'overlapping_components'. Returns next free component index."""
    bm.edges.ensure_lookup_table()
    changed_edge = bm.edges[edge_index]
    if changed_edge.is_boundary:
        return next_free_component_id
    linked_face_ids = [f.index for f in changed_edge.link_faces]
    assert len(linked_face_ids) == 2
    linked_component_ids = [face_to_component_dict[f_id] for f_id in linked_face_ids]
    faces_in_component_one : set = connected_components[linked_component_ids[0]]
    faces_in_component_two : set = connected_components[linked_component_ids[1]]
    components_are_the_same_pre_update = linked_component_ids[0] == linked_component_ids[1]
    component_union = faces_in_component_one.union(faces_in_component_two)
    def edge_is_not_cut(v1, v2):
        return not utils.mesh_edge_is_cut(dual_graph.edges[(v1, v2)]["mesh_edge_index"], edge_constraints, use_auto_cuts)
    subgraph_with_cuts_applied = nx.subgraph_view(dual_graph, filter_node=lambda v : (v in component_union), filter_edge = edge_is_not_cut)
    updated_faces_in_component_one = nx.node_connected_component(subgraph_with_cuts_applied, linked_face_ids[0])
    updated_faces_in_component_two = nx.node_connected_component(subgraph_with_cuts_applied, linked_face_ids[1])
    components_are_the_same_post_update = next(iter(updated_faces_in_component_one)) in updated_faces_in_component_two

    if components_are_the_same_pre_update and not components_are_the_same_post_update:
        # Split component in two
        connected_components[linked_component_ids[0]] = updated_faces_in_component_one
        connected_components[next_free_component_id] = updated_faces_in_component_two
        for f_id in updated_faces_in_component_one:
            face_to_component_dict[f_id] = linked_component_ids[0]
        for f_id in updated_faces_in_component_two:
            face_to_component_dict[f_id] = next_free_component_id
        # discard old info
        cyclic_components.discard(linked_component_ids[0])
        overlapping_components.discard(linked_component_ids[0])
        # compute new info
        if cutgraph.connected_component_contains_cycles(linked_component_ids[0], dual_graph, edge_constraints, connected_components, use_auto_cuts):
            cyclic_components.add(linked_component_ids[0])
        if cutgraph.connected_component_contains_cycles(next_free_component_id, dual_graph, edge_constraints, connected_components, use_auto_cuts):
            cyclic_components.add(next_free_component_id)
        # overlaps have to be computed later 
        return next_free_component_id + 1
    if not components_are_the_same_pre_update and components_are_the_same_post_update:
        # Merge components
        connected_components[linked_component_ids[0]] = updated_faces_in_component_one
        for f_id in faces_in_component_two:
            face_to_component_dict[f_id] = linked_component_ids[0]
        del connected_components[linked_component_ids[1]] # connected components are not longer compact
        # discard old info
        cyclic_components.discard(linked_component_ids[0])
        overlapping_components.discard(linked_component_ids[0])
        cyclic_components.discard(linked_component_ids[1])
        outdated_components.discard(linked_component_ids[1])
        overlapping_components.discard(linked_component_ids[1])
        # compute new info
        if cutgraph.connected_component_contains_cycles(linked_component_ids[0], dual_graph, edge_constraints, connected_components, use_auto_cuts):
            cyclic_components.add(linked_component_ids[0])
        # overlaps have to be computed later
        return next_free_component_id

def update_connected_components_around_edge(mesh : Mesh, edge_index):
    zamboni_props = mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(mesh)
    connected_components, face_to_component_dict = io.read_connected_components(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    outdated_components = io.read_outdated_render_data(mesh)
    overlapping_components = io.read_components_with_overlaps(mesh)
    edge_constraints = io.read_edge_constraints_dict(mesh)
    next_free_component_id = io.read_next_free_component_index(mesh)
    dual_graph = utils.construct_dual_graph_from_bmesh(bm)
    
    next_free_component_id = _update_connected_components_around_edge(bm, dual_graph, edge_index, connected_components, face_to_component_dict, 
                                                                      cyclic_components, outdated_components, overlapping_components, 
                                                                      edge_constraints, zamboni_props.use_auto_cuts, next_free_component_id)
    
    io.write_connected_components(connected_components)
    io.write_components_with_cycles_set(cyclic_components)
    io.write_outdated_render_data(outdated_components)
    io.write_components_with_overlaps(overlapping_components)
    io.write_next_free_component_index(next_free_component_id)

    bm.free()

def _collect_component_ids_along_edges(bm : BMesh, face_to_component_dict, touched_edges):
    components_along_edges = set()
    bm.edges.ensure_lookup_table()
    for edge_id in touched_edges:
        for f in bm.edges[edge_id].link_faces:
            components_along_edges.add(face_to_component_dict[f.index])
    return components_along_edges

def _mark_components_along_edges_as_outdated(bm : BMesh, face_to_component_dict, touched_edges, outdated_components : set):
    components_to_update = _collect_component_ids_along_edges(bm, face_to_component_dict, touched_edges)
    for c_id in components_to_update:
        outdated_components.add(c_id)

def _mark_all_components_as_outdated(bm : BMesh, connected_components : dict, outdated_components : set):
    outdated_components.clear()
    for component_id in connected_components.keys():
        outdated_components.add(component_id)

def _update_unfoldings_along_edges(bm : BMesh, dual_graph : nx.Graph, touched_edges, edge_constraints, use_auto_cuts, connected_components, face_to_component_dict, cyclic_components,
                                   face_triangulation_indices_dict, inner_transform_data_per_face, local_2d_coord_system_per_face,
                                   unfolded_triangles_per_face_per_component, affine_transform_to_root_cood_system_per_face_per_component, components_with_intersections : set): # data in this row will be written to
    components_to_update = _collect_component_ids_along_edges(bm, face_to_component_dict, touched_edges)
    
    for component_id in components_to_update:
        if component_id in cyclic_components:
            continue
        tree_traversal, pred_dict = unfolding.compute_tree_traversal(connected_components[component_id], dual_graph, edge_constraints, use_auto_cuts)
        unfolded_triangulated_faces, affine_transform_to_root_coord_system_per_face, intersection_occured = unfolding.compute_2d_unfolded_triangles_of_component(bm, tree_traversal, pred_dict, 
                                                                                                                                                                 face_triangulation_indices_dict, 
                                                                                                                                                                 inner_transform_data_per_face,
                                                                                                                                                                 local_2d_coord_system_per_face,
                                                                                                                                                                 skip_intersection_test=False)
        unfolded_triangles_per_face_per_component[component_id] = unfolded_triangulated_faces
        affine_transform_to_root_cood_system_per_face_per_component[component_id] = affine_transform_to_root_coord_system_per_face
        if intersection_occured:
            components_with_intersections.add(component_id)

#################################
#           Glue Flaps          #
#################################

def _check_for_flap_collisions(flap_triangles, component_index, edge_index, glueflap_geometry, unfolded_component_geometry, 
                               glueflap_collisions : dict = None): # output
    """ Returns True if the provided triangles overlap with any face or existing flap triangles """
    collision_detected = False
    # first check against all triangles in the face
    all_triangles = []
    for tri_batch in unfolded_component_geometry[component_index].values():
        for tri in tri_batch:
            all_triangles.append(tri)        

    if glueflap_collisions is not None:
        def store_flap_collision(edge_id, other_id):
            glueflap_collisions.setdefault(component_index, {})
            glueflap_collisions[component_index].setdefault(edge_id, set())
            glueflap_collisions[component_index][edge_id].add(other_id)

    for unfold_triangle in all_triangles:
        for flap_triangle in flap_triangles:
            if geometry.triangle_intersection_test_2d(*unfold_triangle, *flap_triangle):
                collision_detected = True
                if glueflap_collisions is not None:
                    store_flap_collision(edge_index, -1)
                else:
                    return True
    
    # check against all existing flaps
    for edge_to_triangles_dict in glueflap_geometry[component_index].values():
        for other_edge_index, other_flap_triangles in edge_to_triangles_dict.items():
            for other_flap_triangle in other_flap_triangles:
                for flap_triangle in flap_triangles:
                    if geometry.triangle_intersection_test_2d(*other_flap_triangle, *flap_triangle):
                        collision_detected = True
                        if glueflap_collisions is not None:
                            store_flap_collision(edge_index, other_edge_index)
                            store_flap_collision(other_edge_index, edge_index)
                        else:
                            return True

    return collision_detected

def _add_glue_flap_to_face_edge(flap_triangles, component_index, face_index, edge_index, halfedge, unfolded_component_geometry,
                                glueflap_dict, glueflap_collisions, glueflap_geometry): # output
    """ Write flap triangles (in 2d root space) to a dict for later intersection tests and printing"""
    # store glueflap
    glueflap_dict[edge_index] = halfedge
    # store collisions
    collision_occured = _check_for_flap_collisions(flap_triangles, component_index, edge_index, glueflap_geometry, unfolded_component_geometry, glueflap_collisions)
    # store geometry
    glueflap_geometry[component_index][face_index][edge_index] = flap_triangles
    return collision_occured

def _remove_flap_from_edge(component_index, face_index, edge_index,
                           glueflap_dict, glueflap_collisions, glueflap_geometry): # output
        # remove from flap dict
        if edge_index in glueflap_dict:
            del glueflap_dict[edge_index]
        # remove geometry
        if edge_index in glueflap_geometry[component_index][face_index]:
            del glueflap_geometry[component_index][face_index][edge_index]
        # remove collisions 
        for collision_edge_index in glueflap_collisions[component_index][edge_index]:
            if collision_edge_index in glueflap_collisions[component_index] and edge_index in glueflap_collisions[component_index][collision_edge_index]:
                glueflap_collisions[component_index][collision_edge_index].remove(edge_index)
        del glueflap_collisions[component_index][edge_index]

def _remove_all_flap_info(glueflap_dict : dict, glueflap_collisions : dict, glueflap_geometry : dict):
    # clear the dict
    glueflap_dict.clear()
    # clear geometry
    glueflap_geometry.clear()
    # clear collisions
    glueflap_collisions.clear()

def _try_to_add_glue_flap(component_index, face_index, edge : bmesh.types.BMEdge, halfedge, flap_angle, flap_height, 
                          affine_transforms_to_roots, inner_face_affine_transforms, enforce,
                           glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry):
    flap_triangles = glueflaps.compute_2d_glue_flap_triangles(component_index, face_index, edge, flap_angle, flap_height, affine_transforms_to_roots, inner_face_affine_transforms)
    if _check_for_flap_collisions(flap_triangles, component_index, edge.index, glueflap_geometry, unfolded_component_geometry):
        # flap does not work
        if not enforce:
            return False
    _add_glue_flap_to_face_edge(flap_triangles, component_index, face_index, edge.index, halfedge, unfolded_component_geometry, glueflap_dict, glueflap_collisions, glueflap_geometry)
    return True

def _add_one_glue_flap(bm : BMesh, preferred_halfedge, flap_angle, flap_height, 
                       halfedge_to_face_dict, face_to_component_dict, 
                       affine_transforms_to_roots, inner_face_affine_transforms,
                       glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry):
    
    mesh_edge : bmesh.types.BMEdge = utils.find_bmesh_edge_of_halfedge(bm, preferred_halfedge)
    if mesh_edge.index in glueflap_dict.keys():
        print("WARNING: This edge already has a glue flap attached to it")
        return 
    
    if mesh_edge.is_boundary:
        print("WARNING: Boundary edge cannot have a glue flap.")
        return
    
    preferred_mesh_face = halfedge_to_face_dict[preferred_halfedge]
    preferred_component_index = face_to_component_dict[preferred_mesh_face.index]

    # try out the preferred halfedge
    if _try_to_add_glue_flap(preferred_component_index, preferred_mesh_face.index, mesh_edge.index, preferred_halfedge, flap_angle, flap_height, 
                             affine_transforms_to_roots, inner_face_affine_transforms, False, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry):
        return preferred_halfedge, True
    
    other_halfedge = (preferred_halfedge[1], preferred_halfedge[0])
    other_mesh_face = halfedge_to_face_dict[other_halfedge]
    other_component_index = face_to_component_dict[other_mesh_face.index]

    # try out the other flap option
    if _try_to_add_glue_flap(other_component_index, other_mesh_face.index, mesh_edge.index, other_halfedge, flap_angle, flap_height, 
                             affine_transforms_to_roots, inner_face_affine_transforms, False, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry):
        return other_halfedge, True

    # add the faulty but preferred flap
    _try_to_add_glue_flap(preferred_component_index, preferred_mesh_face.index, mesh_edge.index, preferred_halfedge, flap_angle, flap_height, 
                          affine_transforms_to_roots, inner_face_affine_transforms, True, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry)
    return preferred_halfedge, False

def _greedy_place_all_flaps(bm : BMesh, flap_angle, flap_height, use_auto_cuts,
                            halfedge_to_face_dict, face_to_component_dict, 
                            affine_transforms_to_roots, inner_face_affine_transforms,
                            unfolded_component_geometry, edge_constraints,
                            cyclic_components, prefer_zigzag):
    """ Attaches flaps to all cut edges (not on boundary edges) """
    bm.verts.ensure_lookup_table()
    glueflap_dict = {}
    glueflap_collisions = {}
    glueflap_geometry = {}
    no_overlaps_introduced = True

    # build dfs tree of cut edges
    processed_edges = set()
    for edge in bm.edges:
        if edge.index in processed_edges:
            continue
        if edge.is_boundary:
            processed_edges.add(edge.index)
            continue
        if not utils.mesh_edge_is_cut(edge.index, edge_constraints, use_auto_cuts):
            processed_edges.add(edge.index)
            continue

        start_edge = (edge.verts[0], edge.verts[1], True) # arbitrary start vertex
        dfs_stack = deque()
        dfs_stack.append(start_edge)

        while dfs_stack:
            v0 , v1, flap_01 = dfs_stack.pop()
            curr_e = bm.edges.get([v0, v1])  
            if curr_e.index in processed_edges:
                continue
            processed_edges.add(curr_e.index)
            # print("trying to add a flap to edge with index", curr_e.index)

            # add the glue flap
            curr_flap_01 = flap_01
            if not curr_e.is_boundary:
                connected_component_id_01 = face_to_component_dict[halfedge_to_face_dict[(v0.index, v1.index)].index]
                connected_component_id_10 = face_to_component_dict[halfedge_to_face_dict[(v1.index, v0.index)].index]
            if not (curr_e.is_boundary or connected_component_id_01 in cyclic_components or connected_component_id_10 in cyclic_components):
                preferred_halfedge = (v0.index, v1.index) if (prefer_zigzag and not flap_01) or (not prefer_zigzag and flap_01) else (v1.index, v0.index)
                used_halfedge, no_new_overlap = _add_one_glue_flap(bm, preferred_halfedge, flap_angle, flap_height, halfedge_to_face_dict, face_to_component_dict, affine_transforms_to_roots,
                                                                   inner_face_affine_transforms, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry)
                curr_flap_01 = used_halfedge == (v0.index, v1.index)
                if no_overlaps_introduced and not no_new_overlap:
                    no_overlaps_introduced = False

            # collect next edges
            for nb_e in v1.link_edges:
                nb_v = nb_e.other_vert(v1)
                if nb_e.index in processed_edges:
                    # print("already visited edge", nb_e.index)
                    continue
                if not utils.mesh_edge_is_cut(nb_e.index, edge_constraints, use_auto_cuts):
                    # print("not visiting edge", nb_e.index)
                    continue
                dfs_stack.append((v1, nb_v, curr_flap_01))

    return glueflap_dict, glueflap_geometry, glueflap_collisions, no_overlaps_introduced

def _update_all_flap_geometry(bm : BMesh, flap_angle, flap_height, halfedge_to_face_dict, face_to_component_dict, 
                              affine_transforms_to_roots, inner_face_affine_transforms,unfolded_component_geometry,
                              glueflap_dict, glueflap_geometry, glueflap_collisions):
    """ This function only changes flap geometry but not the flap placement """
    # clear geometry
    glueflap_geometry.clear()
    # clear collisions
    glueflap_collisions.clear()
    bm.edges.ensure_lookup_table()
    for edge_id, halfedge in glueflap_dict.items():
        face_index = halfedge_to_face_dict[halfedge].index
        component_index = face_to_component_dict[face_index]
        _try_to_add_glue_flap(component_index, face_index, bm.edges[edge_id], halfedge, flap_angle, flap_height, affine_transforms_to_roots, inner_face_affine_transforms,
                              True, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry)
    
def update_all_flap_geometry(mesh : Mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    zamboni_props = mesh.polyzamboni_general_mesh_props
    halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(mesh)
    _, face_to_component_dict = io.read_connected_components(mesh)
    affine_transforms_to_root = io.read_affine_transforms_to_roots(mesh)
    inner_face_affine_transforms = io.read_inner_face_affine_transforms(mesh)
    component_geometry = io.read_facewise_triangles_per_component(mesh)
    glueflap_geometry = io.read_glue_flap_2d_triangles_per_edge_per_face(mesh)
    glueflap_dict = io.read_glueflap_halfedge_dict(mesh)
    glueflap_collisions = io.read_glue_flap_collisions_dict(mesh)

    _update_all_flap_geometry(bm, zamboni_props.glue_flap_angle, zamboni_props.glue_flap_height, halfedge_to_face_dict, face_to_component_dict, affine_transforms_to_root, inner_face_affine_transforms, 
                              component_geometry, glueflap_dict, glueflap_geometry, glueflap_collisions)
    
    io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, glueflap_geometry)
    io.write_glueflap_halfedge_dict(mesh, glueflap_dict)
    io.write_glue_flap_collision_dict(mesh, glueflap_collisions)

    bm.free()

def _swap_glue_flap(bm : BMesh, edge_index, flap_angle, flap_height, halfedge_to_face_dict, face_to_component_dict, 
                    affine_transforms_to_roots, inner_face_affine_transforms,unfolded_component_geometry,
                    glueflap_dict, glueflap_geometry, glueflap_collisions):
    """ If there is a glue flap attached to this edge, attach it to the opposite halfedge. """
    if edge_index not in glueflap_dict.keys():
        return
    bm.edges.ensure_lookup_table()
    # remove current glue flap
    halfedge = glueflap_dict[edge_index]
    face_index = halfedge_to_face_dict[halfedge].index
    component_index = face_to_component_dict[face_index]

    _remove_flap_from_edge(component_index, face_index, edge_index, glueflap_dict, glueflap_collisions, glueflap_geometry)

    opp_halfedge = (halfedge[1], halfedge[0])
    opp_face_index = halfedge_to_face_dict[halfedge].index
    opp_component_index = face_to_component_dict[face_index]

    # add glue flap on the opposide halfedge
    _try_to_add_glue_flap(opp_component_index, opp_face_index, edge_index, opp_halfedge, flap_angle, flap_height, affine_transforms_to_roots, inner_face_affine_transforms,
                          True, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry)

def flip_glue_flaps(mesh : Mesh, edge_indices):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    zamboni_props = mesh.polyzamboni_general_mesh_props
    _, face_to_component_dict = io.read_connected_components(mesh)
    affine_transform_to_root = io.read_affine_transforms_to_roots(mesh)
    unfolded_triangles = io.read_facewise_triangles_per_component(mesh)
    inner_face_transforms = io.read_inner_face_affine_transforms(mesh)
    halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bm)
    glueflap_dict = io.read_glueflap_halfedge_dict(mesh)
    glueflap_geometry = io.read_glue_flap_2d_triangles_per_edge_per_face(mesh)
    glueflap_collisions = io.read_glue_flap_collisions_dict(mesh)

    for edge_index in edge_indices:
        _swap_glue_flap(bm, edge_index, zamboni_props.glue_flap_angle, zamboni_props.glue_flap_height, halfedge_to_face_dict, face_to_component_dict, affine_transform_to_root, inner_face_transforms,
                        unfolded_triangles, glueflap_dict, glueflap_geometry, glueflap_collisions)

    io.write_glueflap_halfedge_dict(mesh, glueflap_dict)
    io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, glueflap_geometry)
    io.write_glue_flap_collision_dict(mesh, glueflap_collisions)

    bm.free()

def _greedy_update_flaps_around_changed_components(bm : BMesh, touched_edge_ids, flap_angle, flap_height, use_auto_cuts,
                                                   halfedge_to_face_dict, face_to_component_dict, connected_components,
                                                   affine_transforms_to_roots, inner_face_affine_transforms,
                                                   unfolded_component_geometry, edge_constraints,
                                                   cyclic_components, prefer_zigzag,
                                                   glueflap_dict, glueflap_geometry, glueflap_collisions):
    """ This function is cringe lol """
    updated_components = set()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for touched_edge_index in touched_edge_ids:
        linked_face_ids = [f.index for f in bm.edges[touched_edge_index].link_faces]
        if len(linked_face_ids) == 1:
            continue # boundary edge does nothing
        assert len(linked_face_ids) == 2 # manifold mesh please!
        for f_id in linked_face_ids:
            updated_components.add(face_to_component_dict[f_id])
        
    # remove all geometric glueflap info attached to the components
    for component_index in updated_components:
        glueflap_geometry[component_index] = {}
        glueflap_collisions[component_index] = {}

    all_interesting_edge_ids = set()
    for face_set in [connected_components[c_id] for c_id in updated_components]:
        for face_index in face_set:
            for edge in bm.faces[face_index].edges:
                all_interesting_edge_ids.add(edge.index)

    def remove_flap_from_edge_if_existing(edge_index):
        if edge_index in glueflap_dict.keys():
            curr_halfedge = glueflap_dict[edge_index]
            curr_face = halfedge_to_face_dict[curr_halfedge]
            curr_component = face_to_component_dict[curr_face]
            _remove_flap_from_edge(curr_component, face_index, edge_index, glueflap_dict, glueflap_collisions, glueflap_geometry)

    no_flap_overlaps = True

    # build dfs tree of cut edges
    processed_edges = set()
    for edge_id in all_interesting_edge_ids:
        edge = bm.edges[edge_id]

        if edge_id in processed_edges:
            continue
        if edge.is_boundary:
            processed_edges.add(edge_id)
            continue
        if not utils.mesh_edge_is_cut(edge_id, edge_constraints, use_auto_cuts):
            # remove any existing flap here
            remove_flap_from_edge_if_existing(edge_id)
            processed_edges.add(edge_id)
            continue

        # start dfs
        start_edge = (edge.verts[0], edge.verts[1], True) # arbitrary start vertex
        dfs_stack = deque()
        dfs_stack.append(start_edge)

        while dfs_stack:
            v0 , v1, flap_01 = dfs_stack.pop()
            curr_e = bm.edges.get([v0, v1])
            if curr_e.index in processed_edges:
                continue
            processed_edges.add(curr_e.index)
            # print("trying to add a flap to edge with index", curr_e.index)

            # add the glue flap
            curr_flap_01 = flap_01
            if not curr_e.is_boundary:
                connected_component_id_01 = face_to_component_dict[halfedge_to_face_dict[(v0.index, v1.index)].index]
                connected_component_id_10 = face_to_component_dict[halfedge_to_face_dict[(v1.index, v0.index)].index]
            if not curr_e.is_boundary and (connected_component_id_01 in cyclic_components or connected_component_id_10 in cyclic_components):
                # remove any flaps attached to this edge
                remove_flap_from_edge_if_existing(curr_e.index)
            elif not curr_e.is_boundary:
                # add a new glue flap 
                if curr_e.index not in glueflap_dict.keys():
                    preferred_halfedge = (v0.index, v1.index) if (prefer_zigzag and not flap_01) or (not prefer_zigzag and flap_01) else (v1.index, v0.index)
                    used_halfedge, no_new_overlap = _add_one_glue_flap(bm, preferred_halfedge, flap_angle, flap_height, halfedge_to_face_dict, face_to_component_dict, affine_transforms_to_roots,
                                                                       inner_face_affine_transforms, glueflap_dict, glueflap_collisions, glueflap_geometry, unfolded_component_geometry)
                    curr_flap_01 = used_halfedge == (v0.index, v1.index)
                    if not no_new_overlap:
                        no_flap_overlaps = False
                else:
                    # restore old glue flap
                    he_with_flap = glueflap_dict[curr_e.index]
                    face_with_flap = halfedge_to_face_dict[he_with_flap]
                    component_with_flap = face_to_component_dict[face_with_flap]

                    if component_with_flap in updated_components:
                        flap_triangles = glueflaps.compute_2d_glue_flap_triangles(component_with_flap, face_with_flap, curr_e, flap_angle, flap_height, affine_transforms_to_roots, inner_face_affine_transforms)
                        collision_occured = _add_glue_flap_to_face_edge(flap_triangles, component_with_flap, face_with_flap, curr_e.index, he_with_flap, unfolded_component_geometry, glueflap_dict, glueflap_collisions, glueflap_geometry)
                        if collision_occured:
                            no_flap_overlaps = False
                        
                    curr_flap_01 = he_with_flap == (v0.index, v1.index)

            # collect next edges
            for nb_e in v1.link_edges:
                nb_v = nb_e.other_vert(v1)
                if nb_e.index in processed_edges:
                    # print("already visited edge", nb_e.index)
                    continue
                if nb_e.index not in all_interesting_edge_ids:
                    continue
                if not utils.mesh_edge_is_cut(nb_e.index, edge_constraints, use_auto_cuts):
                    # print("not visiting edge", nb_e.index)
                    continue
                dfs_stack.append((v1, nb_v, curr_flap_01))

    return no_flap_overlaps
    
def recompute_all_glue_flaps(mesh : Mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    zamboni_props = mesh.polyzamboni_general_mesh_props
    _, face_to_component_dict = io.read_connected_components(mesh)
    affine_transform_to_root = io.read_affine_transforms_to_roots(mesh)
    unfolded_triangles = io.read_facewise_triangles_per_component(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    inner_face_transforms = io.read_inner_face_affine_transforms(mesh)
    edge_constraints = io.read_edge_constraints_dict(mesh)
    halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bm)

    glueflap_dict, glueflap_geometry, glueflap_collisions, _ = _greedy_place_all_flaps(bm, zamboni_props.glue_flap_angle, zamboni_props.glue_flap_height, zamboni_props.use_auto_cuts,
                                                                                                            halfedge_to_face_dict, face_to_component_dict, affine_transform_to_root, 
                                                                                                            inner_face_transforms, unfolded_triangles, edge_constraints, 
                                                                                                            cyclic_components, zamboni_props.prefer_alternating_flaps)
    io.write_glueflap_halfedge_dict(mesh, glueflap_dict)
    io.write_glue_flap_2d_triangles_per_edge_per_face(mesh, glueflap_geometry)
    io.write_glue_flap_collision_dict(mesh, glueflap_collisions)

    bm.free()