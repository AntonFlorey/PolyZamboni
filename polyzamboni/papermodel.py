"""
This file contains papermodel classes. It is the core of the polyzamboni addon.
"""

import bpy
from bpy.types import Mesh
import bmesh
from bmesh.types import BMesh
import networkx as nx
import numpy as np
from collections import deque
from itertools import product
import time

from . import utils
from . import io
from . import geometry
from . import unfolding
from . import glueflaps
from . import zambonipolice

class ConnectedComponent():
    def __init__(self, face_indices):
        self.face_index_set = set(face_indices)
        self.cyclic = False
        self.overlapping = False
        self.render_vertices = None
        self.render_triangles = None
        self.unfolding_affine_transforms = None
        self.unfolded_face_geometry = None
        self.glueflap_geometry = None
        self.glueflap_collisions = None
        self.build_step_number = None
        self.render_data_outdated = True
        self.page_number = None
        self.page_transform = None

    @classmethod
    def from_mesh_props(cls, face_indices, cyclic, overlapping, unfolding_affine_transforms, unfolded_face_geometry, glueflap_geometry, glueflap_collisions, 
                        build_step_number, render_data_outdated, render_vertices, render_triangles, page_number, page_transform):
        instance = cls(face_indices)
        instance.cyclic = cyclic
        instance.overlapping = overlapping
        instance.unfolding_affine_transforms = unfolding_affine_transforms
        instance.unfolded_face_geometry = unfolded_face_geometry
        instance.glueflap_geometry = glueflap_geometry
        instance.glueflap_collisions = glueflap_collisions
        instance.build_step_number = build_step_number
        instance.render_data_outdated = render_data_outdated
        instance.render_vertices = render_vertices
        instance.render_triangles = render_triangles
        instance.page_number = page_number
        instance.page_transform = page_transform
        return instance

    @classmethod
    def new_from_cyclic_region(cls, face_indices):
        instance = cls(face_indices)
        instance.cyclic = True
        return instance

    @classmethod
    def new_from_foldable_region(cls, bmesh : BMesh, tree_ordered_face_indices, pred_dict, face_triangulation_indices, local_coords_per_face, inner_affine_transforms_per_face, skip_intersection_test=False):
        instance = cls(tree_ordered_face_indices)
        unfolded_faces, affine_transform_to_root_coord_system_per_face, intersection = unfolding.compute_2d_unfolded_triangles_of_component(bmesh, tree_ordered_face_indices, pred_dict,
                                                                                                                                            face_triangulation_indices, 
                                                                                                                                            inner_affine_transforms_per_face,
                                                                                                                                            local_coords_per_face,
                                                                                                                                            skip_intersection_test)
        instance.unfolded_face_geometry = unfolded_faces
        instance.unfolding_affine_transforms = affine_transform_to_root_coord_system_per_face
        instance.overlapping = intersection
        instance.glueflap_geometry = {}
        instance.glueflap_collisions = {}
        return instance
    
    def compute_2d_glue_flap_triangles(self, face_index, edge : bmesh.types.BMEdge, flap_angle, flap_height, inner_face_affine_transforms):
        triangles_in_local_edge_coords = glueflaps.compute_2d_glue_flap_triangles_edge_local(edge, flap_angle, flap_height)
        edge_to_root = self.unfolding_affine_transforms[face_index] @ inner_face_affine_transforms[face_index][edge.index]
        return [tuple([edge_to_root * local_coord for local_coord in triangle]) for triangle in triangles_in_local_edge_coords]

    def has_overlapping_glue_flaps(self):
        if self.glueflap_collisions is None:
            return False
        for registered_collisions in self.glueflap_collisions.values():
            if len(registered_collisions) > 0:
                return True
        return False

    def __store_glueflap_mesh_collision(self, edge_index):
        self.glueflap_collisions.setdefault(edge_index, set())
        self.glueflap_collisions[edge_index].add(-1)

    def __store_glueflap_glueflap_collision(self, edge_index_1, edge_index_2):
        self.glueflap_collisions.setdefault(edge_index_1, set())
        self.glueflap_collisions.setdefault(edge_index_2, set())
        self.glueflap_collisions[edge_index_1].add(edge_index_2)
        self.glueflap_collisions[edge_index_2].add(edge_index_1)

    def remove_glue_flap(self, edge_index):
        if self.cyclic:
            return
        if edge_index in self.glueflap_geometry:
            del self.glueflap_geometry[edge_index]
        if edge_index in self.glueflap_collisions:
            for other_collision_edge_index in self.glueflap_collisions[edge_index]:
                if other_collision_edge_index != -1:
                    self.glueflap_collisions[other_collision_edge_index].remove(edge_index)
            del self.glueflap_collisions[edge_index]
    
    def remove_all_glue_flaps(self):
        if self.cyclic:
            return
        self.glueflap_collisions.clear()
        self.glueflap_geometry.clear()

    def check_for_flap_collisions(self, flap_edge_index, flap_triangles, store_collisions=False):
        collision_detected = False
        # first check against all triangles in the face
        for unfold_triangle in [triangle for triangle_list in self.unfolded_face_geometry.values() for triangle in triangle_list]:
            for flap_triangle in flap_triangles:
                if geometry.triangle_intersection_test_2d(*unfold_triangle, *flap_triangle):
                    collision_detected = True
                    if store_collisions:
                        self.__store_glueflap_mesh_collision(flap_edge_index)
                    else:
                        return True
        # check against all existing flaps
        for other_edge_index, other_flap_triangles in self.glueflap_geometry.items():
            for other_flap_triangle in other_flap_triangles:
                for flap_triangle in flap_triangles:
                    if geometry.triangle_intersection_test_2d(*other_flap_triangle, *flap_triangle):
                        collision_detected = True
                        if store_collisions:
                            self.__store_glueflap_glueflap_collision(flap_edge_index, other_edge_index)
                        else:
                            return True
        return collision_detected

    def add_glue_flap(self, flap_edge_index, flap_triangles):
        collision_occured = self.check_for_flap_collisions(flap_edge_index, flap_triangles, store_collisions=True)
        self.glueflap_geometry[flap_edge_index] = flap_triangles
        return collision_occured

class PaperModel():
    def __init__(self, mesh):
        self.__init_mesh_data(mesh)
        self.valid = True
        self.edge_constraints = {}
        self.connected_components = {}
        self.face_to_component_index_dict = {}
        self.glueflap_dict = {}

        # geometric data
        self.face_triangulations = {}
        self.local_coord_system_per_face = {}
        self.inner_face_transforms = {}
        self.geometric_data_changed = False
    
    @classmethod
    def from_existing(cls, mesh : Mesh):
        """ Initializes a paper model from data attached to a Mesh object. """
        papermodel_instance = cls(mesh)
        # check if the existing data is valid (or atleast seems valid)
        papermodel_instance.__init_mesh_data(mesh)
        if not zambonipolice.check_if_polyzamobni_data_exists_and_fits_to_bmesh(mesh, papermodel_instance.bm):
            papermodel_instance.valid = False
            return papermodel_instance # forget about reading the garbage data
        # read geometric data
        papermodel_instance.face_triangulations = io.read_triangulation_indices_per_face(mesh)
        papermodel_instance.local_coord_system_per_face = io.read_local_coordinate_systems_per_face(mesh)
        papermodel_instance.inner_face_transforms = io.read_inner_face_affine_transforms(mesh)
        # all other data
        papermodel_instance.edge_constraints = io.read_edge_constraints_dict(mesh)
        connected_component_sets, face_to_component_index_dict = io.read_connected_components(mesh)
        papermodel_instance.face_to_component_index_dict = face_to_component_index_dict
        cyclic_components : set = io.read_components_with_cycles_set(mesh)
        overlapping_components : set = io.read_components_with_overlaps(mesh)
        outdated_components : set = io.read_outdated_render_data(mesh)
        step_numbers : dict = io.read_build_step_numbers(mesh)
        affine_transforms_to_root : dict = io.read_affine_transforms_to_roots(mesh)
        unfolded_mesh_faces : dict = io.read_facewise_triangles_per_component(mesh)
        papermodel_instance.glueflap_dict = io.read_glueflap_halfedge_dict(mesh)
        glueflap_geometry = io.read_glue_flap_geometry_per_edge_per_component(mesh)
        glueflap_collisions = io.read_glue_flap_collisions_dict(mesh)
        component_render_vertices, component_render_triangles = io.read_all_component_render_data(mesh)
        page_numbers_per_component = io.read_page_numbers(mesh)
        page_transforms_per_component = io.read_page_transforms(mesh)
        for component_id, component_face_set in connected_component_sets.items():
            read_component = ConnectedComponent.from_mesh_props(component_face_set, component_id in cyclic_components, component_id in overlapping_components, 
                                                                affine_transforms_to_root[component_id] if component_id in affine_transforms_to_root else None, 
                                                                unfolded_mesh_faces[component_id] if component_id in unfolded_mesh_faces else None, 
                                                                glueflap_geometry[component_id] if component_id in glueflap_geometry else None, 
                                                                glueflap_collisions[component_id] if component_id in glueflap_collisions else None,
                                                                step_numbers[component_id] if component_id in step_numbers else None,
                                                                component_id in outdated_components,
                                                                component_render_vertices[component_id] if component_id in component_render_vertices else None,
                                                                component_render_triangles[component_id] if component_id in component_render_triangles else None,
                                                                page_numbers_per_component[component_id] if page_numbers_per_component is not None and component_id in page_numbers_per_component else None,
                                                                page_transforms_per_component[component_id] if page_transforms_per_component is not None and component_id in page_transforms_per_component else None)
            papermodel_instance.connected_components[component_id] = read_component
        return papermodel_instance

    @classmethod
    def new_from_mesh(cls, mesh : Mesh):
        """ Initializes a new paper model of the given mesh. """
        papermodel_instance = cls(mesh)
        papermodel_instance.__init_mesh_data(mesh)
        papermodel_instance.__recompute_mesh_geometry_data()
        papermodel_instance.__recompute_all_connected_components()
        papermodel_instance.__place_all_glue_flaps_via_greedy_dfs()
        return papermodel_instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            print("POLYZAMBONI ERROR: Exception occured while working on the paper model!")
            self.valid = False
        self.close()
        return True 

    def __init_mesh_data(self, mesh : Mesh):
        self.mesh = mesh
        self.bm = bmesh.new()
        self.bm.from_mesh(mesh)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()
        self.dual_graph = utils.construct_dual_graph_from_bmesh(self.bm)
        self.halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(self.bm)
        zamboni_props = mesh.polyzamboni_general_mesh_props
        self.zamboni_props = mesh.polyzamboni_general_mesh_props
        self.zigzag_flaps = zamboni_props.prefer_alternating_flaps
        self.flap_height = zamboni_props.glue_flap_height
        self.flap_angle = zamboni_props.glue_flap_angle

    def write_back_mesh_data(self):
        # write geometric data
        if self.geometric_data_changed:
            io.write_triangulation_indices_per_face(self.mesh, self.face_triangulations)
            io.write_local_coordinate_systems_per_face(self.mesh, self.local_coord_system_per_face)
            io.write_inner_face_affine_transforms(self.mesh, self.inner_face_transforms)
        # write all other data
        self.__compactify()
        outdated_components = set()
        overlapping_components = set()
        cyclic_components = set()
        step_number_dict = {}
        components_as_sets = {}
        unfolded_geometry = {}
        affine_transforms_to_roots = {}
        glue_flap_geometry = {}
        glue_flap_collisions = {}
        component_render_vertices = {}
        component_render_triangles = {}
        page_numbers_per_component = {}
        page_transforms_per_component = {}
        component : ConnectedComponent
        for c_index, component in self.connected_components.items():
            components_as_sets[c_index] = component.face_index_set
            if component.build_step_number is not None:
                step_number_dict[c_index] = component.build_step_number
            if component.render_data_outdated:
                outdated_components.add(c_index)
            if component.render_vertices is not None:
                assert component.render_triangles is not None
                component_render_vertices[c_index] = component.render_vertices
                component_render_triangles[c_index] = component.render_triangles
            if component.cyclic:
                cyclic_components.add(c_index)
                continue
            # unfolding data
            if component.overlapping:
                overlapping_components.add(c_index)
            unfolded_geometry[c_index] = component.unfolded_face_geometry
            affine_transforms_to_roots[c_index] = component.unfolding_affine_transforms
            glue_flap_geometry[c_index] = component.glueflap_geometry
            glue_flap_collisions[c_index] = component.glueflap_collisions
            # print data
            if component.page_number is not None:
                assert component.page_transform is not None
                page_numbers_per_component[c_index] = component.page_number
                page_transforms_per_component[c_index] = component.page_transform
        io.write_edge_constraints_dict(self.mesh, self.edge_constraints)
        io.write_connected_components(self.mesh, components_as_sets)
        io.write_components_with_cycles_set(self.mesh, cyclic_components)
        io.write_outdated_render_data(self.mesh, outdated_components)
        io.write_build_step_numbers(self.mesh, step_number_dict)
        io.write_components_with_overlaps(self.mesh, overlapping_components)
        io.write_affine_transforms_to_roots(self.mesh, affine_transforms_to_roots)
        io.write_facewise_triangles_per_component(self.mesh, unfolded_geometry)
        io.write_glueflap_halfedge_dict(self.mesh, self.glueflap_dict)
        io.write_glue_flap_geometry_per_edge_per_component(self.mesh, glue_flap_geometry)
        io.write_glue_flap_collision_dict(self.mesh, glue_flap_collisions)
        io.write_all_component_render_data(self.mesh, component_render_vertices, component_render_triangles)
        io.write_page_numbers(self.mesh, page_numbers_per_component)
        io.write_page_transforms(self.mesh, page_transforms_per_component)
        self.zamboni_props.has_attached_paper_model = True

    def close(self):
        """ Write back all altered papermodel data back to mesh properties and frees the bmesh. """
        if self.valid:
            self.write_back_mesh_data()
        else:
            print("Paper Model data was invalid. Removed all data.")
            io.remove_all_polyzamboni_data(self.mesh)
            self.zamboni_props.has_attached_paper_model = False
        self.bm.free()

    #################################
    #      Editing Operations       #
    #################################

    def cut_edges(self, indices_of_edges_to_cut):
        for edge_index in indices_of_edges_to_cut:
            self.edge_constraints[edge_index] = "cut"
        self.__recompute_connected_components_around_edges(indices_of_edges_to_cut)
        self.__greedy_update_flaps_after_touching_edges(indices_of_edges_to_cut)

    def glue_edges(self, indices_of_edges_to_glue):
        for edge_index in indices_of_edges_to_glue:
            self.edge_constraints[edge_index] = "glued"
        self.__recompute_connected_components_around_edges(indices_of_edges_to_glue)
        self.__greedy_update_flaps_after_touching_edges(indices_of_edges_to_glue)

    def clear_edges(self, indices_of_edges_to_clear, skip_intersection_tests=False):
        """ Returns True if clearing the edge constraints did not product any glue flap overlaps. (needed for the automatic cutting algorithm) """
        for edge_index in indices_of_edges_to_clear:
            if edge_index in self.edge_constraints:
                del self.edge_constraints[edge_index]
        self.__recompute_connected_components_around_edges(indices_of_edges_to_clear, skip_intersection_tests)
        return self.__greedy_update_flaps_after_touching_edges(indices_of_edges_to_clear)
    
    def auto_cut_edges(self, indices_of_edges_to_cut, skip_intersection_tests=False):
        for edge_index in indices_of_edges_to_cut:
            self.edge_constraints[edge_index] = "auto"
        self.__recompute_connected_components_around_edges(indices_of_edges_to_cut, skip_intersection_tests)
        self.__greedy_update_flaps_after_touching_edges(indices_of_edges_to_cut)

    def fill_with_auto_cuts(self):
        filled_edges = []
        for edge in self.bm.edges:
            if edge.index in self.edge_constraints.keys() and self.edge_constraints[edge.index] != "auto":
                continue
            self.edge_constraints[edge.index] = "auto"
            filled_edges.append(edge)
        self.__recompute_connected_components_around_edges([e.index for e in filled_edges])
        self.__greedy_update_flaps_after_touching_edges([e.index for e in filled_edges])
        return filled_edges

    def remove_all_auto_cuts(self):
        edge_indices_to_remove = set()
        for edge_index, constraint in self.edge_constraints.items():
            if constraint == "auto":
                edge_indices_to_remove.add(edge_index)
        for edge_index in edge_indices_to_remove:
            del self.edge_constraints[edge_index]
        self.__recompute_connected_components_around_edges(edge_indices_to_remove)
        self.__greedy_update_flaps_after_touching_edges(edge_indices_to_remove)

    def cut_out_face_region(self, indices_of_faces_in_region):
        face_set = set(indices_of_faces_in_region)
        edges = set()
        for f_id in indices_of_faces_in_region:
            edges = edges.union([e.index for e in self.bm.faces[f_id].edges])
        for e_id in edges:
            cut_this_edge = False
            for f in self.bm.edges[e_id].link_faces:
                if f.index not in face_set:
                    cut_this_edge = True
                    break
            if cut_this_edge:
                self.edge_constraints[e_id] = "cut"
            else:
                self.edge_constraints[e_id] = "glued"
        self.__recompute_connected_components_around_edges(edges)
        self.__greedy_update_flaps_after_touching_edges(edges)

    def add_cuts_between_different_materials(self):
        edges_cut = []
        for mesh_edge in self.bm.edges:
            if mesh_edge.is_boundary:
                continue
            face_0 = mesh_edge.link_faces[0]
            face_1 = mesh_edge.link_faces[1]
            if face_0.material_index != face_1.material_index:
                self.edge_constraints[mesh_edge.index] = "cut"
                edges_cut.append(mesh_edge.index)
        self.__recompute_connected_components_around_edges(edges_cut)
        self.__greedy_update_flaps_after_touching_edges(edges_cut)

    def compute_build_step_numbers(self, starting_face_indices):
        visited_components = set()
        next_build_index = 1
        for starting_face_index in starting_face_indices:
            if self.face_to_component_index_dict[starting_face_index] in visited_components:
                continue
            next_build_index = self.__build_order_bfs(self.face_to_component_index_dict[starting_face_index], next_build_index, visited_components)
        # some mesh pieces might not be selected. they have to be collected here
        for component_id in self.connected_components.keys():
            if component_id in visited_components:
                continue
            next_build_index = self.__build_order_bfs(component_id, next_build_index, visited_components)
        # sanity check
        assert next_build_index == len(self.connected_components.keys()) + 1

    def compute_all_glueflaps_greedily(self):
        self.__place_all_glue_flaps_via_greedy_dfs()

    def flip_glue_flaps_around_edges(self, edge_indices):
        for edge_index in edge_indices:
            self.__flip_glue_flap(edge_index)

    def apply_mesh_geometry_changes(self):
        self.__recompute_mesh_geometry_data()
        self.__recompute_all_connected_components()
        self.__recompute_all_flap_geometry()

    def update_all_flap_geometry(self):
        self.__recompute_all_flap_geometry()

    def test_if_two_touching_unfolded_components_overlap(self, component_id_1, component_id_2, join_face_index_1, join_face_index_2, join_verts_1, join_verts_2):
        """ Test if two components undoldings would overlap if they were merged at the given faces. """

        assert join_verts_1[0].index == join_verts_2[1].index
        assert join_verts_1[1].index == join_verts_2[0].index

        # compute affine transformation from unfolding 1 to unfolding 2
        component_1 : ConnectedComponent = self.connected_components[component_id_1]
        component_2 : ConnectedComponent = self.connected_components[component_id_2]

        # translations
        join_point_1 = component_1.unfolding_affine_transforms[join_face_index_1] * geometry.to_local_coords(join_verts_1[1].co, *self.local_coord_system_per_face[join_face_index_1])
        join_point_2 = component_2.unfolding_affine_transforms[join_face_index_2] * geometry.to_local_coords(join_verts_2[0].co, *self.local_coord_system_per_face[join_face_index_2])
        join_point_1_to_orig = geometry.AffineTransform2D(affine_part=-join_point_1)
        orig_to_join_point_2 = geometry.AffineTransform2D(affine_part=join_point_2)
        # rotation
        other_join_point_1 = component_1.unfolding_affine_transforms[join_face_index_1] * geometry.to_local_coords(join_verts_1[0].co, *self.local_coord_system_per_face[join_face_index_1])
        other_join_point_2 = component_2.unfolding_affine_transforms[join_face_index_2] * geometry.to_local_coords(join_verts_2[1].co, *self.local_coord_system_per_face[join_face_index_2])
        x_ax_1, y_ax_1 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_1, other_join_point_1)
        x_ax_2, y_ax_2 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_2, other_join_point_2)
        basis_mat_1 = np.array([x_ax_1, y_ax_1]).T
        basis_mat_2 = np.array([x_ax_2, y_ax_2]).T
        rotate_edges_together = geometry.AffineTransform2D(linear_part=basis_mat_2 @ np.linalg.inv(basis_mat_1))
        # full transformation
        transform_first_unfolding = orig_to_join_point_2 @ rotate_edges_together @ join_point_1_to_orig

        # make a picture for debugging
        # all_triangles_combined = []
        # for triangle_list_1 in component_1.unfolded_face_geometry.values():
        #     unfolding_1_triangles = [tuple([transform_first_unfolding * tri_coord for tri_coord in triangle]) for triangle in triangle_list_1]
        #     all_triangles_combined += unfolding_1_triangles
        # for triangle_list_2 in component_2.unfolded_face_geometry.values():
        #     all_triangles_combined += triangle_list_2
        # geometry.debug_draw_polygons_2d(all_triangles_combined, "adjacent_components_debug_img")

        for triangle_list_1 in component_1.unfolded_face_geometry.values():
            unfolding_1_triangles = [tuple([transform_first_unfolding * tri_coord for tri_coord in triangle]) for triangle in triangle_list_1]
            for triangle_list_2 in component_2.unfolded_face_geometry.values():
                for triangle_a, triangle_b in product(unfolding_1_triangles, triangle_list_2):
                    if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                        return True # they do overlap    
        return False

    #################################
    #       Private Methods         #
    #################################

    def __get_adjacent_connected_component_ids_of_component(self, component_index, connected_component : ConnectedComponent):
        neighbour_ids = set()
        for component_face_index in connected_component.face_index_set:
            for nb_face_index in self.dual_graph.neighbors(component_face_index):
                nb_component_index = self.face_to_component_index_dict[nb_face_index]
                if nb_component_index == component_index:
                    continue
                neighbour_ids.add(nb_component_index)
        return neighbour_ids

    def __build_order_bfs(self, starting_component_index, next_free_build_index, visited_components : set[int]):
        component_queue = deque()
        component_queue.append(starting_component_index)
        next_build_index = next_free_build_index
        while component_queue:
            curr_component_id = component_queue.popleft()
            if curr_component_id in visited_components:
                continue
            curr_component : ConnectedComponent = self.connected_components[curr_component_id]
            visited_components.add(curr_component_id)
            curr_component.build_step_number = next_build_index
            next_build_index += 1
            for nb_component_index in self.__get_adjacent_connected_component_ids_of_component(curr_component_id, curr_component):
                if nb_component_index in visited_components:
                    continue
                component_queue.append(nb_component_index)
        return next_build_index

    def __recompute_mesh_geometry_data(self):
        self.face_triangulations = unfolding.compute_all_face_triangulation_indices(self.bm)
        self.local_coord_system_per_face = {}
        self.inner_face_transforms = {}
        for bm_face in self.bm.faces:
            local_2d_coord_system, inner_affine_transforms = unfolding.compute_local_coordinate_system_with_all_transitions_to_it(self.bm, bm_face)
            self.local_coord_system_per_face[bm_face.index] = local_2d_coord_system
            self.inner_face_transforms[bm_face.index] = inner_affine_transforms
        self.geometric_data_changed = True

    def __compactify(self):
        compactified_components = {}
        connected_component : ConnectedComponent
        for compact_index, connected_component in enumerate(self.connected_components.values()):
            compactified_components[compact_index] = connected_component
        self.face_to_component_index_dict.clear()
        for component_index, connected_component in self.connected_components.items():
            for face_index in connected_component.face_index_set:
                self.face_to_component_index_dict[face_index] = component_index
        self.connected_components = compactified_components.copy()
    
    def __component_tree_search(self, start_face_index, visited_faces : set, subgraph_with_cuts_applied):
        root = start_face_index
        pred_dict = {root : (root, None)}
        tree_traversal = []
        component_has_cycles = False
        q = [root]
        while q:
            curr_f = q.pop()
            if curr_f in visited_faces:
                #component_has_cycles = True # cycle detected!
                continue
            tree_traversal.append(curr_f)
            visited_faces.add(curr_f)

            for nb_f in subgraph_with_cuts_applied.neighbors(curr_f):
                if nb_f == pred_dict[curr_f][0]:
                    continue # ignore the current face
                if nb_f in visited_faces:
                    component_has_cycles = True # cycle detected!
                    continue
                pred_dict[nb_f] = (curr_f, self.dual_graph.edges[(curr_f, nb_f)]["mesh_edge_index"])
                q.append(nb_f)
        
        # validation_graph = nx.subgraph_view(subgraph_with_cuts_applied, filter_node=lambda v : v in tree_traversal)
        # print("Has cycles own function:", component_has_cycles)
        # print("Has cycles truth", not nx.is_tree(validation_graph))
        # assert nx.is_connected(validation_graph) and nx.is_tree(validation_graph) != component_has_cycles

        return tree_traversal, pred_dict, component_has_cycles

    def __recompute_all_connected_components(self):
        self.connected_components.clear()
        self.face_to_component_index_dict.clear()
        def edge_is_not_cut(v1, v2):
            return not utils.mesh_edge_is_cut(self.dual_graph.edges[(v1, v2)]["mesh_edge_index"], self.edge_constraints)
        subgraph_with_cuts_applied = nx.subgraph_view(self.dual_graph, filter_edge = edge_is_not_cut)
        visited_faces = set()
        next_component_index = 0
        for face_index in self.dual_graph.nodes:
            if face_index in visited_faces:
                continue
            tree_traversal, pred_dict, component_has_cycles = self.__component_tree_search(face_index, visited_faces, subgraph_with_cuts_applied)
            if component_has_cycles:
                new_component = ConnectedComponent.new_from_cyclic_region(tree_traversal)
            else:
                new_component = ConnectedComponent.new_from_foldable_region(self.bm, tree_traversal, pred_dict, self.face_triangulations, self.local_coord_system_per_face, self.inner_face_transforms)
            self.connected_components[next_component_index] = new_component
            for face_index in tree_traversal:
                self.face_to_component_index_dict[face_index] = next_component_index
            next_component_index += 1

    def __recompute_connected_components_around_edges(self, edge_index_list, skip_intersection_tests=False):
        def edge_is_not_cut(v1, v2):
            return not utils.mesh_edge_is_cut(self.dual_graph.edges[(v1, v2)]["mesh_edge_index"], self.edge_constraints)
        subgraph_with_cuts_applied = nx.subgraph_view(self.dual_graph, filter_edge = edge_is_not_cut)
        stale_component_indices = set()
        visited_faces = set()
        next_component_index = max(self.connected_components.keys()) + 1
        for face_index in [bm_face.index for bm_edge in [self.bm.edges[edge_index] for edge_index in edge_index_list] for bm_face in bm_edge.link_faces]:
            if face_index in visited_faces:
                continue
            tree_traversal, pred_dict, component_has_cycles = self.__component_tree_search(face_index, visited_faces, subgraph_with_cuts_applied)
            if component_has_cycles:
                new_component = ConnectedComponent.new_from_cyclic_region(tree_traversal)
            else:
                new_component = ConnectedComponent.new_from_foldable_region(self.bm, tree_traversal, pred_dict, self.face_triangulations, self.local_coord_system_per_face, 
                                                                            self.inner_face_transforms, skip_intersection_test=skip_intersection_tests)
            self.connected_components[next_component_index] = new_component
            for face_index in tree_traversal:
                stale_component_indices.add(self.face_to_component_index_dict[face_index])
                self.face_to_component_index_dict[face_index] = next_component_index
            next_component_index += 1
        for component_index in stale_component_indices:
            del self.connected_components[component_index]

        # assertion for testing
        assert nx.number_connected_components(subgraph_with_cuts_applied) == len(self.connected_components.keys())

    def __add_glueflap_to_one_halfedge(self, preferred_halfedge):
        """ Returns: The used halfedge, no new overlaps (bool) """
        bm_edge : bmesh.types.BMEdge = utils.find_bmesh_edge_of_halfedge(self.bm, preferred_halfedge)
        if bm_edge.index in self.glueflap_dict.keys():
            print("POLYZAMBONI ERROR: This edge already has a glue flap attached to it.")
            return
        if bm_edge.is_boundary:
            print("POLYZAMBONI WARNING: Boundary edge cannot have a glue flap.")
            return
        
        preferred_mesh_face = self.halfedge_to_face_dict[preferred_halfedge]
        preferred_component : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[preferred_mesh_face.index]]
        preferred_triangles = preferred_component.compute_2d_glue_flap_triangles(preferred_mesh_face.index, bm_edge, self.flap_angle, self.flap_height, self.inner_face_transforms)
        if not preferred_component.check_for_flap_collisions(bm_edge.index, preferred_triangles, store_collisions=False):
            self.glueflap_dict[bm_edge.index] = preferred_halfedge
            preferred_component.add_glue_flap(bm_edge.index, preferred_triangles)
            return preferred_halfedge, True
        
        other_halfedge = (preferred_halfedge[1], preferred_halfedge[0])
        other_face = self.halfedge_to_face_dict[other_halfedge]
        other_component : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[other_face.index]]
        other_triangles = other_component.compute_2d_glue_flap_triangles(other_face.index, bm_edge, self.flap_angle, self.flap_height, self.inner_face_transforms)
        if not other_component.check_for_flap_collisions(bm_edge.index, other_triangles, store_collisions=False):
            self.glueflap_dict[bm_edge.index] = other_halfedge
            other_component.add_glue_flap(bm_edge.index, other_triangles)
            return other_halfedge, True
        self.glueflap_dict[bm_edge.index] = preferred_halfedge
        preferred_component.add_glue_flap(bm_edge.index, preferred_triangles)
        return preferred_halfedge, False

    def __flip_glue_flap(self, edge_index):
        if edge_index not in self.glueflap_dict:
            return
        curr_halfedge = self.glueflap_dict[edge_index]
        curr_bm_face = self.halfedge_to_face_dict[curr_halfedge]
        curr_component : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[curr_bm_face.index]]
        curr_component.remove_glue_flap(edge_index)
        opp_halfedge = (curr_halfedge[1], curr_halfedge[0])
        opp_bm_face = self.halfedge_to_face_dict[opp_halfedge]
        opp_component : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[opp_bm_face.index]]
        opp_triangles = opp_component.compute_2d_glue_flap_triangles(opp_bm_face.index, self.bm.edges[edge_index], self.flap_angle, self.flap_height, self.inner_face_transforms)
        opp_component.add_glue_flap(edge_index, opp_triangles)
        self.glueflap_dict[edge_index] = opp_halfedge

    def __place_all_glue_flaps_via_greedy_dfs(self):
        """ Attaches flaps to all cut edges (not on boundary edges) """
        self.glueflap_dict.clear()
        connected_component : ConnectedComponent
        for connected_component in self.connected_components.values():
            connected_component.remove_all_glue_flaps()

        # build dfs tree of cut edges
        no_overlaps_introduced = True
        processed_edges = set()
        for edge in self.bm.edges:
            if edge.index in processed_edges:
                continue
            if edge.is_boundary:
                processed_edges.add(edge.index)
                continue
            if not utils.mesh_edge_is_cut(edge.index, self.edge_constraints):
                processed_edges.add(edge.index)
                continue

            start_edge = (edge.verts[0], edge.verts[1], True) # arbitrary start vertex
            dfs_stack = deque()
            dfs_stack.append(start_edge)

            while dfs_stack:
                v0, v1, flap_01 = dfs_stack.pop()
                curr_e = self.bm.edges.get([v0, v1])  
                if curr_e.index in processed_edges:
                    continue
                processed_edges.add(curr_e.index)

                # add the glue flap
                curr_flap_01 = flap_01
                if not curr_e.is_boundary:
                    connected_component_01 : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[self.halfedge_to_face_dict[(v0.index, v1.index)].index]]
                    connected_component_10 : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[self.halfedge_to_face_dict[(v1.index, v0.index)].index]]
                if not (curr_e.is_boundary or connected_component_01.cyclic or connected_component_10.cyclic):
                    preferred_halfedge = (v0.index, v1.index) if (self.zigzag_flaps and not flap_01) or (not self.zigzag_flaps and flap_01) else (v1.index, v0.index)
                    used_halfedge, no_new_overlap = self.__add_glueflap_to_one_halfedge(preferred_halfedge)
                    curr_flap_01 = used_halfedge == (v0.index, v1.index)
                    if no_overlaps_introduced and not no_new_overlap:
                        no_overlaps_introduced = False

                # collect next edges
                for nb_e in v1.link_edges:
                    nb_v = nb_e.other_vert(v1)
                    if nb_e.index in processed_edges:
                        continue
                    if not utils.mesh_edge_is_cut(nb_e.index, self.edge_constraints):
                        continue
                    dfs_stack.append((v1, nb_v, curr_flap_01))

        return no_overlaps_introduced

    def __greedy_update_flaps_after_touching_edges(self, touched_edge_ids):
        """ This function is cringe lol """
        updated_components = set()
        for touched_edge_index in touched_edge_ids:
            linked_face_ids = [f.index for f in self.bm.edges[touched_edge_index].link_faces]
            if len(linked_face_ids) == 1:
                continue # boundary edge does nothing
            assert len(linked_face_ids) == 2 # manifold mesh please!
            for f_id in linked_face_ids:
                updated_components.add(self.face_to_component_index_dict[f_id])
            
        # remove all geometric glueflap info attached to the components
        for component_index in updated_components:
            curr_component : ConnectedComponent = self.connected_components[component_index]
            curr_component.remove_all_glue_flaps()

        all_interesting_edge_ids = set()
        for face_index in [face_index for c_id in updated_components for face_index in self.connected_components[c_id].face_index_set]: 
            for interesting_edge in self.bm.faces[face_index].edges:
                all_interesting_edge_ids.add(interesting_edge.index)

        def remove_flap_from_edge_if_existing(edge_index):
            if edge_index in self.glueflap_dict.keys():
                del_halfedge = self.glueflap_dict[edge_index]
                del_face : bmesh.types.BMFace = self.halfedge_to_face_dict[del_halfedge]
                del_component : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[del_face.index]]
                del_component.remove_glue_flap(edge_index)
                del self.glueflap_dict[edge_index]

        # build dfs tree of cut edges
        processed_edges = set()
        no_flap_overlaps = True
        for curr_edge_id in all_interesting_edge_ids:
            curr_bm_edge = self.bm.edges[curr_edge_id]
            if curr_edge_id in processed_edges:
                continue
            if curr_bm_edge.is_boundary:
                processed_edges.add(curr_edge_id)
                continue
            if not utils.mesh_edge_is_cut(curr_edge_id, self.edge_constraints):
                # remove any existing flap here
                remove_flap_from_edge_if_existing(curr_edge_id)
                processed_edges.add(curr_edge_id)
                continue

            # start dfs
            start_edge = (curr_bm_edge.verts[0], curr_bm_edge.verts[1], True) # arbitrary start vertex
            dfs_stack = deque()
            dfs_stack.append(start_edge)

            while dfs_stack:
                v0 : bmesh.types.BMVert
                v1 : bmesh.types.BMVert
                v0, v1, flap_01 = dfs_stack.pop()
                curr_e = self.bm.edges.get([v0, v1])
                if curr_e.index in processed_edges:
                    continue
                processed_edges.add(curr_e.index)
                # add the glue flap
                curr_flap_01 = flap_01
                if not curr_e.is_boundary:
                    connected_component_01 : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[self.halfedge_to_face_dict[(v0.index, v1.index)].index]]
                    connected_component_10 : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[self.halfedge_to_face_dict[(v1.index, v0.index)].index]]
                if not curr_e.is_boundary and (connected_component_01.cyclic or connected_component_10.cyclic):
                    # remove any flaps attached to this edge
                    remove_flap_from_edge_if_existing(curr_e.index)
                elif not curr_e.is_boundary:
                    # add a new glue flap
                    if curr_e.index not in self.glueflap_dict.keys():
                        preferred_halfedge = (v0.index, v1.index) if (self.zigzag_flaps and not flap_01) or (not self.zigzag_flaps and flap_01) else (v1.index, v0.index)
                        used_halfedge, no_new_overlap = self.__add_glueflap_to_one_halfedge(preferred_halfedge)
                        curr_flap_01 = used_halfedge == (v0.index, v1.index)
                        if not no_new_overlap:
                            no_flap_overlaps = False
                    else:
                        # restore old glue flap
                        he_with_flap = self.glueflap_dict[curr_e.index]
                        face_with_flap : bmesh.types.BMFace = self.halfedge_to_face_dict[he_with_flap]
                        component_index_with_flap = self.face_to_component_index_dict[face_with_flap.index]
                        component_with_flap : ConnectedComponent = self.connected_components[component_index_with_flap]
                        if component_index_with_flap in updated_components:
                            flap_triangles = component_with_flap.compute_2d_glue_flap_triangles(face_with_flap.index, curr_e, self.flap_angle, self.flap_height, self.inner_face_transforms)
                            collision_occured = component_with_flap.add_glue_flap(curr_e.index, flap_triangles)
                            if collision_occured:
                                no_flap_overlaps = False
                        curr_flap_01 = he_with_flap == (v0.index, v1.index)

                # collect next edges
                for nb_e in v1.link_edges:
                    nb_v = nb_e.other_vert(v1)
                    if nb_e.index in processed_edges:
                        continue
                    if nb_e.index not in all_interesting_edge_ids:
                        continue
                    if not utils.mesh_edge_is_cut(nb_e.index, self.edge_constraints):
                        continue
                    dfs_stack.append((v1, nb_v, curr_flap_01))

        return no_flap_overlaps

    def __recompute_all_flap_geometry(self):
        for connected_component in self.connected_components.values():
            connected_component.remove_all_glue_flaps()
        for edge_index, glueflap_halfedge in self.glueflap_dict.items():
            curr_bm_face = self.halfedge_to_face_dict[glueflap_halfedge]
            curr_component : ConnectedComponent = self.connected_components[self.face_to_component_index_dict[curr_bm_face.index]]
            curr_triangles = curr_component.compute_2d_glue_flap_triangles(curr_bm_face.index, self.bm.edges[edge_index], self.flap_angle, self.flap_height, self.inner_face_transforms)
            curr_component.add_glue_flap(edge_index, curr_triangles)
