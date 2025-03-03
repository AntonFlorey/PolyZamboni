"""
This file contains several high level functions that meaningful combine multiple polyzamboni operations.
"""

import bpy
from bpy.types import Mesh
import bmesh
from bmesh.types import BMesh
import numpy as np

from .properties import ZamboniGeneralMeshProps
from . import utils
from . import io
from . import geometry
from . import cutgraph
from . import unfolding
from . import glueflaps
from . import zambonipolice

#################################
#    init, delete and update    #
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
    zamboni_props : ZamboniGeneralMeshProps = mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(mesh)
    dual_graph = utils.construct_dual_graph_from_bmesh(bmesh)

    initialize_precomputable_bmesh_data(bm)

    # edge_constraints
    io.write_edge_constraints_dict(mesh, {})
    
    # connected_components
    connected_components, face_to_component_dict = cutgraph.compute_all_connected_components(mesh, False, {}, bm)
    io.write_connected_components(mesh, connected_components)
    io.write_build_step_numbers(mesh, cutgraph.compute_build_step_numbers(mesh, [], bm, dual_graph, connected_components, face_to_component_dict))
    cyclic_components = cutgraph.compute_cyclic_components(mesh, dual_graph, connected_components, {})
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

    # glue flaps TODO

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
    zamboni_props : ZamboniGeneralMeshProps = mesh.polyzamboni_general_mesh_props
    zamboni_props.has_attached_paper_model = False
    io.remove_all_polyzamboni_data(mesh)

def sync_paper_model_with_mesh_geometry(mesh : Mesh):
    zamboni_props : ZamboniGeneralMeshProps = mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(mesh)

    if zambonipolice.check_if_polyzamobni_data_exists_and_fits_to_bmesh(mesh, bm):
        zamboni_props.attached_paper_model_data_valid = True

        # get new geometrical data
        initialize_precomputable_bmesh_data(mesh, bm)

        connected_components, face_ids_to_component_ids = io.read_connected_components(mesh)

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


        # recompute glue flap geometry TODO

        # mark all render data as outdated
        io.write_outdated_render_data(mesh, set(connected_components.keys()))

    else:
        # I dont know what I want to do here
        zamboni_props.attached_paper_model_data_valid = False

    bm.free()




def add_cut_constraint(self, cut_edge_id):
    self.designer_constraints[cut_edge_id] = "cut"

def add_lock_constraint(self, locked_edge_id):
    self.designer_constraints[locked_edge_id] = "glued"

def clear_edge_constraint(self, edge_id):
    if edge_id in self.designer_constraints.keys():
        del self.designer_constraints[edge_id]

def add_cutout_region(self, face_indices):
    self.mesh.faces.ensure_lookup_table()
    self.mesh.edges.ensure_lookup_table()
    face_set = set(face_indices)
    edges = set()
    for f_id in face_indices:
        edges = edges.union([e.index for e in self.mesh.faces[f_id].edges])
    for e_id in edges:
        cut_this_edge = self.mesh.edges[e_id].is_boundary
        for f in self.mesh.edges[e_id].link_faces:
            if f.index not in face_set:
                cut_this_edge = True
                break
        if cut_this_edge:
            self.designer_constraints[e_id] = "cut"
        else:
            self.designer_constraints[e_id] = "glued"

def add_cuts_between_different_materials(self):
    edges_cut = []
    for mesh_edge in self.mesh.edges:
        if mesh_edge.is_boundary:
            continue
        face_0 = mesh_edge.link_faces[0]
        face_1 = mesh_edge.link_faces[1]
        if face_0.material_index != face_1.material_index:
            self.add_cut_constraint(mesh_edge.index)
            edges_cut.append(mesh_edge.index)
    return edges_cut




#################################
#       Cutgraph Editing        #
#################################

def compactify_polyzamboni_data(mesh : Mesh):
    """ This may take some time so call this rarely """
    pass # TODO
    
    if self.components_are_compact:
        return
    next_compact_component_index = 0
    compact_connected_components = {}

    curr_connected_component : ConnectedComponent
    for _, curr_connected_component in self.connected_components.items():
        if curr_connected_component is None:
            continue
        compact_connected_components[next_compact_component_index] = curr_connected_component
        for face_index in curr_connected_component._face_index_set:
            self.vertex_to_component_dict[face_index] = next_compact_component_index
        next_compact_component_index += 1
    
    self.next_free_component_id = next_compact_component_index
    self.connected_components = compact_connected_components.copy()
    self.components_are_compact = True

def update_all_connected_components(mesh : Mesh, preserve_old_indices=True):
    """ Most data stored for the new components might still be usable. If this is NOT the case, set preserve_old_indices to False to save some time. """
    pass # TODO

def update_connected_components_around_edge(mesh : Mesh, edge_index):
    pass # TODO

    self.mesh.edges.ensure_lookup_table()
    changed_edge = self.mesh.edges[mesh_edge_index]
    if changed_edge.is_boundary:
        return
    linked_face_ids = [f.index for f in changed_edge.link_faces]
    assert len(linked_face_ids) == 2
    linked_component_ids = [self.vertex_to_component_dict[mesh_f] for mesh_f in linked_face_ids]
    component_one : ConnectedComponent = self.connected_components[linked_component_ids[0]]
    component_two : ConnectedComponent = self.connected_components[linked_component_ids[1]]
    components_are_the_same_pre_update = linked_component_ids[0] == linked_component_ids[1]
    component_union = component_one._face_index_set.union(component_two._face_index_set)
    subgraph_with_cuts_applied = nx.subgraph_view(self.dualgraph, filter_node=lambda v : (v in component_union), filter_edge = lambda v1, v2 : not self.edge_is_cut(v1, v2))
    first_component_faces = nx.node_connected_component(subgraph_with_cuts_applied, linked_face_ids[0])
    second_component_faces = nx.node_connected_component(subgraph_with_cuts_applied, linked_face_ids[1])
    components_are_the_same_post_update = next(iter(second_component_faces)) in first_component_faces

    # mark as changed for redraw
    component_one._render_triangles_outdated = True
    component_two._render_triangles_outdated = True

    if components_are_the_same_pre_update and not components_are_the_same_post_update:
        # Split component in two
        self.connected_components[linked_component_ids[0]] = ConnectedComponent(first_component_faces)
        self.connected_components[self.next_free_component_id] = ConnectedComponent(second_component_faces)
        for v in second_component_faces:
            self.vertex_to_component_dict[v] = self.next_free_component_id
        self.next_free_component_id += 1
    if not components_are_the_same_pre_update and components_are_the_same_post_update:
        # Merge components
        self.connected_components[linked_component_ids[0]] = ConnectedComponent(first_component_faces)
        for v in second_component_faces:
            self.vertex_to_component_dict[v] = linked_component_ids[0]
        del self.connected_components[linked_component_ids[1]] # remove free component
        self.components_are_compact = False

def update_unfoldings_along_edges(self, edge_ids, skip_intersection_test=False):
    components_to_update = set()
    self.mesh.edges.ensure_lookup_table()
    for touched_edge in edge_ids:
        linked_face_ids = [f.index for f in self.mesh.edges[touched_edge].link_faces]
        if len(linked_face_ids) == 1:
            continue # boundary edge does nothing
        assert len(linked_face_ids) == 2 # manifold mesh please!
        for f_id in linked_face_ids:
            components_to_update.add(self.vertex_to_component_dict[f_id])

    for c_id in components_to_update:
        current_connected_component : ConnectedComponent = self.connected_components[c_id]
        current_connected_component._unfolding = self.unfold_connected_component(c_id, skip_intersection_test=skip_intersection_test)
        # doing this here is a nasty side effect but whatever
        current_connected_component._build_step_number = None
        current_connected_component._render_triangles_outdated = True


#################################
#           Glue Flaps          #
#################################

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

def try_to_add_glue_flap(self, halfedge, enforce=False):
    edge, face, unfold = self.get_edge_face_and_unfolding_of_halfedge(halfedge)
    flap_geometry = unfold.compute_2d_glue_flap_triangles(face.index, edge, self.flap_angle, self.flap_height)
    if unfold.check_for_flap_collisions(flap_geometry):
        # flap does not work
        if not enforce:
            return False
    unfold.add_glue_flap_to_face_edge(face.index, edge, flap_geometry)
    self.glue_flaps[edge.index] = halfedge
    return True

def add_one_glue_flap(self, preferred_halfedge):
    mesh_edge : bmesh.types.BMEdge = self.halfedge_to_edge[preferred_halfedge]
    if mesh_edge.index in self.glue_flaps.keys():
        print("this edge already has a glue flap attached to it")
        return 
    
    if mesh_edge.is_boundary:
        print("boundary edge cannot have a glue flap.")
        return
    
    if self.try_to_add_glue_flap(preferred_halfedge):
        return preferred_halfedge, True
    
    # try out the other flap option
    if self.try_to_add_glue_flap((preferred_halfedge[1], preferred_halfedge[0])):
        return (preferred_halfedge[1], preferred_halfedge[0]), True

    # add the faulty but preferred flap
    self.try_to_add_glue_flap(preferred_halfedge, enforce=True)
    return preferred_halfedge, False

def greedy_place_all_flaps(self):
    """ Attaches flaps to all cut edges (not on boundary edges) """

    self.ensure_halfedge_to_edge_table()
    self.ensure_halfedge_to_face_table()
    self.assert_all_faces_have_components_and_unfoldings()
    no_flap_overlaps = True
    self.glue_flaps = {}
    # clear all flap info stored in the unfoldings
    curr_connected_component : ConnectedComponent
    for curr_connected_component in self.connected_components.values():
        if curr_connected_component._unfolding is not None:
            curr_connected_component._unfolding.remove_all_flap_info()

    # build dfs tree of cut edges
    processed_edges = set()
    for edge in self.mesh.edges:
        if edge.index in processed_edges:
            continue
        if edge.is_boundary:
            processed_edges.add(edge.index)
            continue
        if not self.mesh_edge_is_cut(edge):
            processed_edges.add(edge.index)
            continue

        start_edge = (edge.verts[0], edge.verts[1], True) # arbitrary start vertex
        dfs_stack = deque()
        dfs_stack.append(start_edge)

        while dfs_stack:
            v0 , v1, flap_01 = dfs_stack.pop()
            curr_e = self.halfedge_to_edge[v0.index, v1.index]
            if curr_e.index in processed_edges:
                continue
            processed_edges.add(curr_e.index)
            # print("trying to add a flap to edge with index", curr_e.index)

            # add the glue flap
            curr_flap_01 = flap_01
            if not curr_e.is_boundary:
                connected_component_01 : ConnectedComponent = self.connected_components[self.vertex_to_component_dict[self.halfedge_to_face[(v0.index, v1.index)].index]]
                connected_component_10 : ConnectedComponent = self.connected_components[self.vertex_to_component_dict[self.halfedge_to_face[(v1.index, v0.index)].index]]
            if not (curr_e.is_boundary or connected_component_01._unfolding is None or connected_component_10._unfolding is None):
                preferred_halfedge = (v0.index, v1.index) if (self.prefer_zigzag and not flap_01) or (not self.prefer_zigzag and flap_01) else (v1.index, v0.index)
                used_halfedge, no_overlaps_introduced =  self.add_one_glue_flap(preferred_halfedge)
                curr_flap_01 = used_halfedge == (v0.index, v1.index)
                if not no_overlaps_introduced:
                    no_flap_overlaps = False

            # collect next edges
            for nb_e in v1.link_edges:
                nb_v = nb_e.other_vert(v1)
                if nb_e.index in processed_edges:
                    # print("already visited edge", nb_e.index)
                    continue
                if not self.mesh_edge_is_cut(nb_e):
                    # print("not visiting edge", nb_e.index)
                    continue
                dfs_stack.append((v1, nb_v, curr_flap_01))

    return no_flap_overlaps

def remove_flaps_of_edge(self, edge : bmesh.types.BMEdge):
    if edge.index in self.glue_flaps.keys():
        glue_he = self.glue_flaps[edge.index]
        _, face_with_flap, unfolding_with_flap = self.get_edge_face_and_unfolding_of_halfedge(glue_he)
        if unfolding_with_flap is not None:
            unfolding_with_flap.remove_flap_from_edge(face_with_flap.index, edge) # remove geometric info 
        del self.glue_flaps[edge.index]

def greedy_update_flaps_around_changed_components(self, edge_ids):
    updated_components = set()
    self.mesh.edges.ensure_lookup_table()
    self.mesh.faces.ensure_lookup_table()
    for touched_edge in edge_ids:
        linked_face_ids = [f.index for f in self.mesh.edges[touched_edge].link_faces]
        if len(linked_face_ids) == 1:
            continue # boundary edge does nothing
        assert len(linked_face_ids) == 2 # manifold mesh please!
        for f_id in linked_face_ids:
            updated_components.add(self.vertex_to_component_dict[f_id])
        
    all_interesting_edge_ids = set()
    for face_set in [self.connected_components[c_id]._face_index_set for c_id in updated_components]:
        for face_index in face_set:
            for edge in self.mesh.faces[face_index].edges:
                all_interesting_edge_ids.add(edge.index)


    self.ensure_halfedge_to_edge_table()
    self.ensure_halfedge_to_face_table()
    self.assert_all_faces_have_components_and_unfoldings()
    no_flap_overlaps = True

    # build dfs tree of cut edges
    processed_edges = set()
    for edge_id in all_interesting_edge_ids:
        edge = self.mesh.edges[edge_id]

        if edge_id in processed_edges:
            continue
        if edge.is_boundary:
            processed_edges.add(edge.index)
            continue
        if not self.mesh_edge_is_cut(edge):
            # remove any existing flap here 
            self.remove_flaps_of_edge(edge)
            processed_edges.add(edge.index)
            continue

        # start dfs
        start_edge = (edge.verts[0], edge.verts[1], True) # arbitrary start vertex
        dfs_stack = deque()
        dfs_stack.append(start_edge)

        while dfs_stack:
            v0 , v1, flap_01 = dfs_stack.pop()
            curr_e = self.halfedge_to_edge[v0.index, v1.index]
            if curr_e.index in processed_edges:
                continue
            processed_edges.add(curr_e.index)
            # print("trying to add a flap to edge with index", curr_e.index)

            # add the glue flap
            curr_flap_01 = flap_01
            if not curr_e.is_boundary:
                connected_component_01 : ConnectedComponent = self.connected_components[self.vertex_to_component_dict[self.halfedge_to_face[(v0.index, v1.index)].index]]
                connected_component_10 : ConnectedComponent = self.connected_components[self.vertex_to_component_dict[self.halfedge_to_face[(v1.index, v0.index)].index]]
            if not curr_e.is_boundary and (connected_component_01._unfolding is None or connected_component_10._unfolding is None):
                # remove any flaps attached to this edge
                self.remove_flaps_of_edge(curr_e)
            elif not curr_e.is_boundary:
                # add a new glue flap 
                if curr_e.index not in self.glue_flaps.keys():
                    preferred_halfedge = (v0.index, v1.index) if (self.prefer_zigzag and not flap_01) or (not self.prefer_zigzag and flap_01) else (v1.index, v0.index)
                    used_halfedge, no_overlap_introduced =  self.add_one_glue_flap(preferred_halfedge)
                    curr_flap_01 = used_halfedge == (v0.index, v1.index)
                    if not no_overlap_introduced:
                        no_flap_overlaps = False
                else:
                    # do nothing or restore old glue flap
                    he_with_flap = self.glue_flaps[curr_e.index]
                    _, face_with_flap, unfold_with_flap = self.get_edge_face_and_unfolding_of_halfedge(he_with_flap)
                    has_flap_overlaps_to_begin_with = unfold_with_flap.has_overlapping_glue_flaps()
                    if not unfold_with_flap.check_if_edge_has_flap(face_with_flap.index, curr_e):
                        flap_geometry = unfold_with_flap.compute_2d_glue_flap_triangles(face_with_flap.index, curr_e, self.flap_angle, self.flap_height)
                        unfold_with_flap.add_glue_flap_to_face_edge(face_with_flap.index, curr_e, flap_geometry)
                        if not has_flap_overlaps_to_begin_with and unfold_with_flap.has_overlapping_glue_flaps():
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
                if not self.mesh_edge_is_cut(nb_e):
                    # print("not visiting edge", nb_e.index)
                    continue
                dfs_stack.append((v1, nb_v, curr_flap_01))

    return no_flap_overlaps

def clear_all_flap_geometry(mesh : Mesh):
    """ Clears all flap triangles and collosion dicts """
    pass # TODO


def update_all_flap_geometry(mesh : Mesh, bm : BMesh):
    """ This function only changes flap geometry but not the flap placement """



    self.ensure_halfedge_to_edge_table()
    self.ensure_halfedge_to_face_table()
    self.mesh.edges.ensure_lookup_table()
    self.assert_all_faces_have_components_and_unfoldings()
    # clear all flap info stored in the unfoldings
    curr_connected_component : ConnectedComponent
    for curr_connected_component in self.connected_components.values():
        if curr_connected_component._unfolding is not None:
            curr_connected_component._unfolding.remove_all_flap_info()

    for edge_id, halfedge in self.glue_flaps.items():
        _, face, unfolding = self.get_edge_face_and_unfolding_of_halfedge(halfedge)
        new_geometry = unfolding.compute_2d_glue_flap_triangles(face.index, self.mesh.edges[edge_id], self.flap_angle, self.flap_height)
        unfolding.add_glue_flap_to_face_edge(face.index, self.mesh.edges[edge_id], new_geometry)
    
def swap_glue_flap(self, edge_id):
    """ If there is a glue flap attached to this edge, attach it to the opposite halfedge. """
    if edge_id not in self.glue_flaps.keys():
        return
    self.ensure_halfedge_to_face_table()
    self.mesh.edges.ensure_lookup_table()
    # remove current glue flap
    curr_halfedge = self.glue_flaps[edge_id]
    _, curr_face, curr_unfolding = self.get_edge_face_and_unfolding_of_halfedge(curr_halfedge)
    curr_unfolding.remove_flap_from_edge(curr_face.index, self.mesh.edges[edge_id])
    # add glue flap on the opposide halfedge
    self.try_to_add_glue_flap((curr_halfedge[1], curr_halfedge[0]), enforce=True)

