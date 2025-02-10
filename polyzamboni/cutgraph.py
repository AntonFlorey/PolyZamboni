import bpy
import bmesh
import numpy as np
import os
import networkx as nx
import math
import mathutils
from .geometry import triangulate_3d_polygon, signed_point_dist_to_line, face_corner_convex_3d, solve_for_weird_intersection_point
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME, AUTO_CUT_EDGES_PROP_NAME, PERFECT_REGION, BAD_GLUE_FLAPS_REGION, OVERLAPPING_REGION, NOT_FOLDABLE_REGION, GLUE_FLAP_NO_OVERLAPS, GLUE_FLAP_WITH_OVERLAPS, GLUE_FLAP_TO_LARGE, BUILD_ORDER_PROPERTY_NAME, GLUE_FLAP_PROPERTY_NAME
from .unfolding import Unfolding
from collections import deque
from .printprepper import ColoredTriangleData, ComponentPrintData, CutEdgeData, FoldEdgeData, GlueFlapEdgeData

class CutGraphSaveData():
    """ Stores all data required to save and load a cutgraph """

    def __init__(self, edge_constraints_dict, sparse_build_steps_dict, glue_flaps_dict):
        self.edge_constraints = edge_constraints_dict.copy()
        self.build_steps = sparse_build_steps_dict.copy()
        self.glue_flaps = glue_flaps_dict.copy()

class ConnectedComponent():
    """ One connected component (or island) of a cut and unfolded mesh """
     
    def __init__(self, face_index_set, unfolding=None, render_triangles=None, render_vertex_positions=None):
        self._face_index_set = face_index_set
        self._unfolding : Unfolding = unfolding
        self._triangles_for_rendering = render_triangles
        self._vertices_for_rendering = render_vertex_positions
        self._render_triangles_outdated = False
        self._build_step_number = None

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
        # print("Glue flaps computed")

    #################################
    #          Save & Load          #
    #################################

    def create_save_data(self):
        return CutGraphSaveData(self.designer_constraints, self.create_sparse_build_steps_dict(), self.create_glue_flaps_dict())

    def load_from_save_data(self, save_data : CutGraphSaveData):
        # set edge constraints
        self.designer_constraints = save_data.edge_constraints.copy()
        # compute connected components
        self.compute_all_connected_components()
        # compute undoldings
        self.unfold_all_connected_components()
        # read build steps
        self.read_sparse_build_steps_dict(save_data.build_steps)
        # glue flaps
        self.read_glue_flaps_dict(save_data.glue_flaps)
        if self.check_if_glue_flaps_exist_and_are_valid():
            self.update_all_flap_geometry()
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

        # compute offsets for user feedback region drawing
        min_edge_len = min([e.calc_length() for e in self.mesh.edges])
        self.small_offset = 0.05 * min_edge_len
        self.large_offset = 0.35 * min_edge_len

        self.halfedge_to_edge = None # delete old halfedge to edge dict
        self.halfedge_to_face = None # delete old halfedge to face dict
        self.valid = True # make sure we check mesh validity again
        self.render_corner_cuts_dict = {}
        if hasattr(self, "connected_components"):
            current_connected_component : ConnectedComponent
            for current_connected_component in self.connected_components.values():
                current_connected_component._render_triangles_outdated = True

    #################################
    #             Misc.             #
    #################################

    def check_validity(self):
        if not self.valid:
            return False # invalid once, invalid until something external changes
        mesh_face_indices = set([f.index for f in self.mesh.faces])
        mesh_edge_indices = set([e.index for e in self.mesh.edges])

        # Constraints have to make sense
        for constraint_edge_index in self.designer_constraints.keys():
            if constraint_edge_index not in mesh_edge_indices:
                print("mesh edge does not exist")
                print(constraint_edge_index)
                print(mesh_edge_indices)
                self.valid = False
                break

        # Dual Graph has to match the current mesh
        if self.dualgraph.number_of_nodes() != len(self.mesh.faces):
            print("dual graph n-nodes != mesh face count")
            self.valid = False
        if self.dualgraph.number_of_edges() != (len(self.mesh.edges) - len([e for e in self.mesh.edges if e.is_boundary])):
            print("dual graph n-edges != mesh edge count", self.dualgraph.number_of_edges(), len(self.mesh.edges) - len([e for e in self.mesh.edges if e.is_boundary]))
            self.valid = False
        for dual_graph_node in self.dualgraph.nodes:
            if dual_graph_node not in mesh_face_indices:
                print("dual graph node is no mesh face")
                self.valid = False
                break
        for dual_graph_edge in self.dualgraph.edges:
            if self.dualgraph.edges[dual_graph_edge]["edge_index"] not in mesh_edge_indices:
                print("duaql graph edge is no mesh edge")
                self.valid = False
                break

        return self.valid

    def edge_is_cut(self, v_from, v_to):
        mesh_edge_index = self.dualgraph.edges[(v_from, v_to)]["edge_index"]
        if mesh_edge_index in self.designer_constraints:
            if self.designer_constraints[mesh_edge_index] == "cut":
                return True
            if self.use_auto_cuts and self.designer_constraints[mesh_edge_index] == "auto":
                return True
        return False

    def mesh_edge_is_cut(self, edge):
        if edge.index in self.designer_constraints:
            if self.designer_constraints[edge.index] == "cut":
                return True
            if self.use_auto_cuts and self.designer_constraints[edge.index] == "auto":
                return True
        return False

    def assert_all_faces_have_components_and_unfoldings(self):
        for f in self.dualgraph.nodes:
            assert f in self.vertex_to_component_dict.keys()
            c_id = self.vertex_to_component_dict[f]
            assert c_id in self.connected_components.keys()

    def ensure_halfedge_to_edge_table(self):
        if hasattr(self, "halfedge_to_edge") and self.halfedge_to_edge is not None:
            return
        self.halfedge_to_edge = {}
        for edge in self.mesh.edges:
            self.halfedge_to_edge[tuple([v.index for v in edge.verts])] = edge
            self.halfedge_to_edge[tuple(reversed([v.index for v in edge.verts]))] = edge

    def ensure_halfedge_to_face_table(self):
        if hasattr(self, "halfedge_to_face") and self.halfedge_to_face is not None:
            return
        self.halfedge_to_face = {}
        for face in self.mesh.faces:
            verts_ccw = list(v.index for v in face.verts)
            for i in range(len(verts_ccw)):
                j = (i + 1) % len(verts_ccw)
                self.halfedge_to_face[(verts_ccw[i], verts_ccw[j])] = face

    def get_edge_face_and_unfolding_of_halfedge(self, halfedge):
        self.ensure_halfedge_to_edge_table()
        self.ensure_halfedge_to_face_table()
        edge = self.halfedge_to_edge[halfedge]
        face = self.halfedge_to_face[halfedge]
        unfold : Unfolding = self.connected_components[self.vertex_to_component_dict[face.index]]._unfolding
        return edge, face, unfold

    #################################
    #      Connected Components     #
    #################################

    def compute_all_connected_components(self):
        if self.mesh is None or self.dualgraph is None:
            print("Warning! Connected components can not yet be computed")
            return
        graph_with_cuts_applied = nx.subgraph_view(self.dualgraph, filter_edge = lambda v1, v2 : not self.edge_is_cut(v1, v2))
        all_components = list(nx.connected_components(graph_with_cuts_applied))

        copy_old_component_data = False
        if hasattr(self, "vertex_to_component_dict"):
            old_vertex_to_component_dict = self.vertex_to_component_dict.copy()
        if hasattr(self, "connected_components"):
            old_connected_components = self.connected_components.copy()
            copy_old_component_data = True

        self.num_components = len(all_components)
        self.vertex_to_component_dict = {}
        self.connected_components = {}
        for c_id, component_faces in enumerate(all_components):
            for v in component_faces:
                self.vertex_to_component_dict[v] = c_id
            if copy_old_component_data:
                old_component : ConnectedComponent = old_connected_components[old_vertex_to_component_dict[list(component_faces)[0]]]
                self.connected_components[c_id] = ConnectedComponent(component_faces, old_component._unfolding, old_component._triangles_for_rendering, old_component._vertices_for_rendering) # copied data might be outdated!
            else:
                self.connected_components[c_id] = ConnectedComponent(component_faces)
        self.next_free_component_id = self.num_components
        self.components_are_compact = True
        # do a sanity check
        self.assert_all_faces_have_components_and_unfoldings()

    def update_connected_components_around_edge(self, mesh_edge_index):
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

    def compactify_components(self):
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

    def adjacent_connected_components_of_component(self, component_id):
        self.mesh.faces.ensure_lookup_table()
        self.mesh.edges.ensure_lookup_table()
        neighbour_ids = set()
        connected_component : ConnectedComponent = self.connected_components[component_id]
        for component_face_index in connected_component._face_index_set:
            for nb_face_index in self.dualgraph.neighbors(component_face_index):
                nb_component_index = self.vertex_to_component_dict[nb_face_index]
                if nb_component_index == component_id:
                    continue
                neighbour_ids.add(nb_component_index)
        return neighbour_ids

    def __build_order_bfs(self, starting_component_index, next_free_build_index, visited_components):
        component_queue = deque()
        component_queue.append(starting_component_index)
        next_build_index = next_free_build_index
        while component_queue:
            curr_component_id = component_queue.popleft()
            if curr_component_id in visited_components:
                continue
            visited_components.add(curr_component_id)
            curr_conn_component : ConnectedComponent = self.connected_components[curr_component_id]
            curr_conn_component._build_step_number = next_build_index
            next_build_index += 1
            for nb_component_index in self.adjacent_connected_components_of_component(curr_component_id):
                if nb_component_index in visited_components:
                    continue
                component_queue.append(nb_component_index)
        return next_build_index

    def generate_bfs_build_oder(self, starting_face_indices):
        self.compactify_components()
        visited_components = set()
        next_build_index = 1
        for starting_face_index in starting_face_indices:
            next_build_index = self.__build_order_bfs(self.vertex_to_component_dict[starting_face_index], next_build_index, visited_components)
        for component_id in self.connected_components.keys():
            if component_id in visited_components:
                continue
            next_build_index = self.__build_order_bfs(component_id, next_build_index, visited_components)

        # sanity check
        assert next_build_index == len(self.connected_components.keys()) + 1
        self.build_steps_valid = True

    def every_component_has_build_step(self):
        curr_connected_component : ConnectedComponent
        for curr_connected_component in self.connected_components.values():
            if curr_connected_component._build_step_number is None:
                return False
        return True

    def check_if_build_steps_are_present_and_valid(self):
        if hasattr(self, "build_steps_valid") and not self.build_steps_valid:
            return self.build_steps_valid
        self.build_steps_valid = True
        if not self.every_component_has_build_step():
            self.build_steps_valid = False
            return False
        all_build_step_numbers = set()
        curr_connected_component : ConnectedComponent
        for curr_connected_component in self.connected_components.values():
            curr_build_number = curr_connected_component._build_step_number
            if curr_build_number > len(self.connected_components) or curr_build_number in all_build_step_numbers:
                self.build_steps_valid = False
                break
            all_build_step_numbers.add(curr_build_number)
        return self.build_steps_valid

    def create_sparse_build_steps_dict(self):
        if not self.check_if_build_steps_are_present_and_valid():
            return {}
        sparse_face_dict = {}
        curr_connected_component : ConnectedComponent
        for c_id, curr_connected_component in self.connected_components.items():
            if len(curr_connected_component._face_index_set) == 0:
                continue # wtf case
            sparse_face_dict[str(list(curr_connected_component._face_index_set)[0])] = curr_connected_component._build_step_number
        return sparse_face_dict

    def read_sparse_build_steps_dict(self, sparse_build_steps):
        curr_connected_component : ConnectedComponent
        for _, curr_connected_component in self.connected_components.items():
            for face_index in curr_connected_component._face_index_set:
                if str(face_index) in sparse_build_steps.keys():
                    curr_connected_component._build_step_number = sparse_build_steps[str(face_index)]
                    break

    #################################
    #           Unfolding           #
    #################################

    def unfold_connected_face_set(self, face_set : set, skip_intersection_test=False):
        """Unfolds one connected component of the cut mesh into the 2D plane. Returns None if the component containts cycles."""

        def f_in_faceset(f):
            return f in face_set
        subgraph_with_cuts_applied = nx.subgraph_view(self.dualgraph, filter_node=f_in_faceset, filter_edge = lambda v1, v2 : not self.edge_is_cut(v1, v2))
        assert nx.connected.is_connected(subgraph_with_cuts_applied)
        if not nx.is_tree(subgraph_with_cuts_applied):
            return None
        
        # custom tree data
        root = list(face_set)[0]
        pred_dict = {root : (root, None)}
        tree_traversal = []

        # custom tree search
        q = [root]
        visited = set()
        while q:
            curr_f = q.pop()
            if curr_f in visited:
                continue
            tree_traversal.append(curr_f)
            visited.add(curr_f)

            for nb_f in subgraph_with_cuts_applied.neighbors(curr_f):
                if nb_f in visited:
                    continue
                if nb_f not in face_set:
                    continue
                pred_dict[nb_f] = (curr_f, self.dualgraph.edges[(curr_f, nb_f)]["edge_index"])
                q.append(nb_f)

        # unfold the component
        assert len(tree_traversal) == len(face_set)
        return Unfolding(self.mesh, tree_traversal, pred_dict, self.face_triangulation_indices, skip_intersection_test=skip_intersection_test)

    def unfold_connected_component(self, component_id, skip_intersection_test=False):
        """Unfolds one connected component of the cut mesh into the 2D plane. Returns None if the component containts cycles."""
        connected_component : ConnectedComponent = self.connected_components[component_id]
        return self.unfold_connected_face_set(connected_component._face_index_set, skip_intersection_test=skip_intersection_test)

    def unfold_all_connected_components(self):
        current_connected_component : ConnectedComponent
        for c_id, current_connected_component in self.connected_components.items():
            current_connected_component._unfolding = self.unfold_connected_component(c_id, False)

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

    def all_components_have_unfoldings(self):
        current_connected_component : ConnectedComponent
        for current_connected_component in self.connected_components.values():
            if current_connected_component._unfolding is None:
                return False
        return True

    #################################
    #           Rendering           #
    #################################

    def mesh_edge_id_list_to_coordinate_list(self, edge_ids, normal_offset):
        if not self.check_validity():
            print("Cutgraph and mesh are not compatible!")
            return []
        cut_coordinates = []
        self.mesh.edges.ensure_lookup_table()
        for edge_id in edge_ids:
            corresponding_mesh_edge = self.mesh.edges[edge_id]
            vert1, vert2 = corresponding_mesh_edge.verts
            cut_coordinates.append(self.world_matrix @ (vert1.co + normal_offset * vert1.normal))
            cut_coordinates.append(self.world_matrix @ (vert2.co + normal_offset * vert1.normal))
        return cut_coordinates

    def __get_corner_points_all_interior_or_cut(self, corner_pos, prev_pos, next_pos, normal, dist_to_edges):
        ctp : mathutils.Vector = (prev_pos - corner_pos)
        ctn : mathutils.Vector = (next_pos - corner_pos)
        corner_angle = math.acos(np.clip(ctp.normalized().dot(ctn.normalized()), -1, 1))
        shell_factor = 1 / math.sin(corner_angle / 2)
        corner_is_convex = face_corner_convex_3d(prev_pos, corner_pos, next_pos, normal)
        rotation_to_angle_bisect_vec : mathutils.Matrix = mathutils.Matrix.Rotation((corner_angle if corner_is_convex else 2 * np.pi - corner_angle) / 2, 3, normal)
        angle_bisect_vec = rotation_to_angle_bisect_vec @ ctn.normalized()
        return [mathutils.Vector(corner_pos + dist_to_edges * shell_factor * angle_bisect_vec)]

    def __get_corner_points_only_vertex_on_cut(self, corner_pos, prev_pos, next_pos, normal : mathutils.Vector, cut_dist, offset_dist):
        ctp : mathutils.Vector = (prev_pos - corner_pos)
        ctn : mathutils.Vector = (next_pos - corner_pos)
        corner_is_convex = face_corner_convex_3d(prev_pos, corner_pos, next_pos, normal)
        corner_angle = math.acos(np.clip(ctp.normalized().dot(ctn.normalized()), -1, 1))
        if not corner_is_convex:
            corner_angle = 2 * np.pi - corner_angle
        rotation_to_angle_bisect_vec : mathutils.Matrix = mathutils.Matrix.Rotation(corner_angle / 2, 3, normal)
        angle_bisect_vec = rotation_to_angle_bisect_vec @ ctn.normalized()
        shell_factor = 1 / math.sin(corner_angle / 2)
        one_point_th_angle = math.asin(offset_dist / cut_dist)
        if corner_angle / 2 <= one_point_th_angle:
            return [mathutils.Vector(corner_pos + offset_dist * shell_factor * angle_bisect_vec)]
        else:
            dist_parallel_to_edges = math.sqrt(cut_dist**2 - offset_dist**2)
            point_a = mathutils.Vector(corner_pos + dist_parallel_to_edges * ctp.normalized() + offset_dist * ctp.cross(normal).normalized())
            point_b = mathutils.Vector(corner_pos + cut_dist * angle_bisect_vec)
            point_c = mathutils.Vector(corner_pos + dist_parallel_to_edges * ctn.normalized() + offset_dist * -ctn.cross(normal).normalized())
            return [point_a, point_b, point_c]

    def __get_corner_points_one_edge_cut(self, corner_pos, prev_pos, next_pos, normal : mathutils.Vector, cut_dist, offset_dist, prev_is_cut):
        ctp : mathutils.Vector = (prev_pos - corner_pos)
        ctn : mathutils.Vector = (next_pos - corner_pos)
        corner_is_convex = face_corner_convex_3d(prev_pos, corner_pos, next_pos, normal)
        corner_angle = math.acos(np.clip(ctp.normalized().dot(ctn.normalized()), -1, 1))
        dist_parallel_to_edges = math.sqrt(cut_dist**2 - offset_dist**2)
        n_a = ctp.cross(normal).normalized()
        n_b = -ctn.cross(normal).normalized()
        if not corner_is_convex:
            corner_angle = 2 * np.pi - corner_angle
            rotation_to_angle_bisect_vec : mathutils.Matrix = mathutils.Matrix.Rotation(corner_angle / 2, 3, normal)
            angle_bisect_vec = rotation_to_angle_bisect_vec @ ctn.normalized()
            if prev_is_cut:
                point_a = mathutils.Vector(corner_pos + cut_dist * n_a)
                point_b = mathutils.Vector(corner_pos + cut_dist * angle_bisect_vec)
                point_c = mathutils.Vector(corner_pos + dist_parallel_to_edges * ctn.normalized() + offset_dist * n_b)
            else:
                point_a = mathutils.Vector(corner_pos + dist_parallel_to_edges * ctp.normalized() + offset_dist * n_a)
                point_b = mathutils.Vector(corner_pos + cut_dist * angle_bisect_vec)
                point_c = mathutils.Vector(corner_pos + cut_dist * n_b)
            return [point_a, point_b, point_c]
        # convex cases
        one_point_th_angle = math.asin(offset_dist / cut_dist) + math.pi / 2
        if corner_angle > one_point_th_angle:
            # two points
            if prev_is_cut:
                point_a = mathutils.Vector(corner_pos + cut_dist * n_a)
                point_b = mathutils.Vector(corner_pos + dist_parallel_to_edges * ctn.normalized() + offset_dist * n_b)
            else:
                point_a = mathutils.Vector(corner_pos + dist_parallel_to_edges * ctp.normalized() + offset_dist * n_a)
                point_b = mathutils.Vector(corner_pos + cut_dist * n_b)
            return [point_a, point_b]
        else:
            # one point (intersection computation)
            if prev_is_cut:
                return [mathutils.Vector(solve_for_weird_intersection_point(corner_pos, prev_pos, next_pos, normal, cut_dist, offset_dist))]
            else:
                return [mathutils.Vector(solve_for_weird_intersection_point(corner_pos, prev_pos, next_pos, normal, offset_dist, cut_dist))]

    def get_polygon_outline_for_face_drawing(self, face_index):
        self.ensure_halfedge_to_edge_table()
        face = self.mesh.faces[face_index]
        face_normal = face.normal
        verts = list(face.verts)
        if face_index not in self.render_corner_cuts_dict.keys():
            self.render_corner_cuts_dict[face_index] = {v.index : {} for v in verts}
        cool_vertices = []
        for v_id in range(len(verts)):
            curr_v = verts[v_id]
            prev_v = verts[(v_id + len(verts) - 1) % len(verts)]
            next_v = verts[(v_id + 1) % len(verts)]

            # v_on_cutting_edge_or_boundary = curr_v.is_boundary or np.any([self.mesh_edge_is_cut(e) for e in curr_v.link_edges])
            v_on_cutting_edge = np.any([self.mesh_edge_is_cut(e) for e in curr_v.link_edges])

            e_to_curr : bmesh.types.BMEdge = self.halfedge_to_edge[(prev_v.index, curr_v.index)]
            e_from_curr : bmesh.types.BMEdge = self.halfedge_to_edge[(curr_v.index, next_v.index)]

            e_prev_curr_is_cutting = self.mesh_edge_is_cut(e_to_curr)
            e_curr_to_next_is_cutting = self.mesh_edge_is_cut(e_from_curr)

            if not v_on_cutting_edge and not e_prev_curr_is_cutting and not e_curr_to_next_is_cutting:
                if "000" not in self.render_corner_cuts_dict[face_index][curr_v.index].keys():
                    self.render_corner_cuts_dict[face_index][curr_v.index]["000"] = self.__get_corner_points_all_interior_or_cut(curr_v.co, prev_v.co, next_v.co, face_normal, self.small_offset)
                cool_vertices += self.render_corner_cuts_dict[face_index][curr_v.index]["000"]
            elif v_on_cutting_edge and e_prev_curr_is_cutting and e_curr_to_next_is_cutting:
                if "111" not in self.render_corner_cuts_dict[face_index][curr_v.index].keys():
                    self.render_corner_cuts_dict[face_index][curr_v.index]["111"] = self.__get_corner_points_all_interior_or_cut(curr_v.co, prev_v.co, next_v.co, face_normal, self.large_offset)
                cool_vertices += self.render_corner_cuts_dict[face_index][curr_v.index]["111"]
            elif v_on_cutting_edge and not e_prev_curr_is_cutting and not e_curr_to_next_is_cutting:
                if "100" not in self.render_corner_cuts_dict[face_index][curr_v.index].keys():
                    self.render_corner_cuts_dict[face_index][curr_v.index]["100"] = self.__get_corner_points_only_vertex_on_cut(curr_v.co, prev_v.co, next_v.co, face_normal, self.large_offset, self.small_offset)
                cool_vertices += self.render_corner_cuts_dict[face_index][curr_v.index]["100"]
            elif e_prev_curr_is_cutting:
                if "101" not in self.render_corner_cuts_dict[face_index][curr_v.index].keys():
                    self.render_corner_cuts_dict[face_index][curr_v.index]["101"] = self.__get_corner_points_one_edge_cut(curr_v.co, prev_v.co, next_v.co, face_normal, self.large_offset, self.small_offset, True)
                cool_vertices += self.render_corner_cuts_dict[face_index][curr_v.index]["101"]
            else:
                if "110" not in self.render_corner_cuts_dict[face_index][curr_v.index].keys():
                    self.render_corner_cuts_dict[face_index][curr_v.index]["110"] = self.__get_corner_points_one_edge_cut(curr_v.co, prev_v.co, next_v.co, face_normal, self.large_offset, self.small_offset, False)
                cool_vertices += self.render_corner_cuts_dict[face_index][curr_v.index]["110"]

        return cool_vertices

    def get_connected_component_triangle_lists_for_drawing(self, face_offset):
        self.mesh.faces.ensure_lookup_table()

        if not hasattr(self, "component_triangle_list_cache"):
            self.component_triangle_list_cache = {}

        triangles_per_component = {}
        all_vertex_positions = []

        curr_connected_component : ConnectedComponent
        for component_id, curr_connected_component in self.connected_components.items():
            # only compute render data if it is missing or marked to be recomputed
            if curr_connected_component._render_triangles_outdated or curr_connected_component._triangles_for_rendering is None:
                component_triangles = []
                component_vertex_positions = []
                for face_index in curr_connected_component._face_index_set:
                    polygon_outline = self.get_polygon_outline_for_face_drawing(face_index)
                    curr_offset = len(component_vertex_positions)
                    _, curr_triangle_ids = triangulate_3d_polygon(polygon_outline, self.mesh.faces[face_index].normal, list(range(curr_offset, curr_offset + len(polygon_outline))))
                    component_vertex_positions += [(v, face_index) for v in polygon_outline]
                    component_triangles += curr_triangle_ids
                curr_connected_component._vertices_for_rendering = component_vertex_positions.copy()
                curr_connected_component._triangles_for_rendering = component_triangles.copy()

            vertex_id_offset = len(all_vertex_positions)
            all_vertex_positions += [self.world_matrix @ (v + face_offset * self.mesh.faces[f_index].normal) for (v, f_index) in curr_connected_component._vertices_for_rendering]
            triangles_per_component[component_id] = [(vertex_id_offset + tri[0], vertex_id_offset + tri[1], vertex_id_offset + tri[2]) for tri in curr_connected_component._triangles_for_rendering]

        return all_vertex_positions, triangles_per_component
    
    def get_triangle_list_per_cluster_quality(self, face_offset):
        if not self.check_validity():
            return [], {} # draw nothing but at least no crash

        all_v_positions, tris_per_cluster = self.get_connected_component_triangle_lists_for_drawing(face_offset)
        quality_dict = {
            PERFECT_REGION : [],
            BAD_GLUE_FLAPS_REGION : [],
            OVERLAPPING_REGION : [],
            NOT_FOLDABLE_REGION : []
            }
        for cluster_id, triangles in tris_per_cluster.items():
            current_connected_component : ConnectedComponent = self.connected_components[cluster_id]
            if current_connected_component._unfolding is None:
                quality_dict[NOT_FOLDABLE_REGION] += triangles
                continue
            unfolding : Unfolding = current_connected_component._unfolding
            if unfolding.has_overlaps:
                quality_dict[OVERLAPPING_REGION] += triangles
                continue
            if unfolding.has_overlapping_glue_flaps():
                quality_dict[BAD_GLUE_FLAPS_REGION] += triangles
                continue
            quality_dict[PERFECT_REGION] += triangles
        return all_v_positions, quality_dict

    def get_lines_array_per_glue_flap_quality(self, face_offset):
        if not self.check_validity():
            return {} # draw nothing but at least no crash
        if not hasattr(self, "glue_flaps"):
            return {} 
        self.assert_all_faces_have_components_and_unfoldings()
        self.mesh.edges.ensure_lookup_table()
        quality_dict = {
            GLUE_FLAP_NO_OVERLAPS : [],
            GLUE_FLAP_WITH_OVERLAPS : [],
            GLUE_FLAP_TO_LARGE : []
            }
        for edge_index, halfedge in self.glue_flaps.items():
            opp_halfedge = (halfedge[1], halfedge[0])
            opp_edge, opp_face, opp_unfolding = self.get_edge_face_and_unfolding_of_halfedge(opp_halfedge)
            if opp_unfolding is None:
                continue
            flap_triangles_3d = opp_unfolding.compute_3d_glue_flap_triangles_inside_face(opp_face.index, self.halfedge_to_edge[halfedge], self.flap_angle, self.flap_height)
            flap_line_array = []
            if len(flap_triangles_3d) == 1:
                tri = flap_triangles_3d[0]
                flap_line_array = [tri[0], tri[1], tri[1], tri[2]]
            else:
                tri_0 = flap_triangles_3d[0]
                tri_1 = flap_triangles_3d[1]
                flap_line_array = [tri_0[0], tri_0[1], tri_0[1], tri_0[2], tri_1[0], tri_1[1]]
            # apply offset
            flap_line_array = [self.world_matrix @ (mathutils.Vector(v) + 1.5 * face_offset * opp_face.normal) for v in flap_line_array]
            if opp_unfolding.flap_is_overlapping(self.mesh.edges[edge_index]):
                quality_dict[GLUE_FLAP_WITH_OVERLAPS] += flap_line_array
                continue
            # TODO Flaps that are to large... maybe not needed
            quality_dict[GLUE_FLAP_NO_OVERLAPS] += flap_line_array
        return quality_dict

    #################################
    #           Glue Flaps          #
    #################################

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

    def update_all_flap_geometry(self):
        """ This mehthod only changes flap geometry but not the flap placement """
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
    
    def create_glue_flaps_dict(self):
        if not hasattr(self, "glue_flaps"):
            return {}
        return {str(edge_id) : halfedge for edge_id, halfedge in self.glue_flaps.items()}

    def read_glue_flaps_dict(self, glue_flap_dict):
        self.glue_flaps = {}
        for edge_id_str, halfedge in glue_flap_dict.items():
            self.glue_flaps[int(edge_id_str)] = tuple(halfedge)

    def check_if_glue_flaps_exist_and_are_valid(self):
        if not hasattr(self, "glue_flaps"):
            return False
        # every cut edge needs a glue flap
        for dual_graph_edge in self.dualgraph.edges:
            c_1 = self.vertex_to_component_dict[dual_graph_edge[0]]
            c_2 = self.vertex_to_component_dict[dual_graph_edge[1]]
            if c_1 is not None and c_2 is not None and self.edge_is_cut(*dual_graph_edge):
                if self.dualgraph.edges[dual_graph_edge]["edge_index"] not in self.glue_flaps.keys():
                    return False
        # every glue flap needs a cut edge between 
        self.ensure_halfedge_to_face_table()
        self.mesh.edges.ensure_lookup_table()
        for edge_index in self.glue_flaps.keys():
            if not self.mesh_edge_is_cut(self.mesh.edges[edge_index]):
                return False
            for adj_face in self.mesh.edges[edge_index].link_faces:
                if self.connected_components[self.vertex_to_component_dict[adj_face.index]]._unfolding is None:
                    return False
        return True
    
    #################################
    #  User Constraints Interface   #
    #################################

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

    def get_manual_cuts_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "cut"]

    def get_locked_edges_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "glued"]

    def get_auto_cuts_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "auto"]

    #################################
    #           Exporting           #
    #################################

    def compute_max_print_component_dimensions(self):
        # compute all geometric print information and get max dimensions
        self.mesh.edges.ensure_lookup_table()
        self.mesh.faces.ensure_lookup_table()
        max_height = -np.inf
        max_width = -np.inf
        curr_connected_component : ConnectedComponent
        for c_id, curr_connected_component in self.connected_components.items():
            faces_ids_in_component = curr_connected_component._face_index_set
            if faces_ids_in_component is None:
                continue
            unfolding_of_curr_component : Unfolding = curr_connected_component._unfolding
            if unfolding_of_curr_component is None:
                continue
            curr_component_print_data = ComponentPrintData()

            # collect cut edges and fold edges
            for face_index in faces_ids_in_component:
                curr_face : bmesh.types.BMFace = self.mesh.faces[face_index]

                # collect all edges
                for curr_edge in curr_face.edges:
                    # compute edge coords in unfolding space
                    vertex_coords_3d = [v.co for v in curr_edge.verts]    
                    vertex_coords_unfolded = [unfolding_of_curr_component.get_globally_consistend_2d_coord_in_face(co_3d, face_index) for co_3d in vertex_coords_3d]
                    if curr_edge.is_boundary or self.mesh_edge_is_cut(curr_edge):
                        # this is a cut edge
                        curr_component_print_data.add_cut_edge(CutEdgeData(tuple(vertex_coords_unfolded), curr_edge.index))

                # collect edges for glue flaps
                for flap_triangles in unfolding_of_curr_component.glue_flaps_per_face[face_index].values():
                    assert len(flap_triangles) > 0
                    curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((flap_triangles[0][0],flap_triangles[0][1])))
                    curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((flap_triangles[0][1],flap_triangles[0][2])))
                    if len(flap_triangles) == 2:
                        curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((flap_triangles[1][1],flap_triangles[1][2])))
            
            # align vertically 
            curr_component_print_data.align_vertically_via_pca()
            # get width and height
            max_width = max(curr_component_print_data.upper_right[0] - curr_component_print_data.lower_left[0], max_width)
            max_height = max(curr_component_print_data.upper_right[1] - curr_component_print_data.lower_left[1], max_height)
        return max_width, max_height

    def compute_mesh_height(self):
        mesh_top = max([v.co[2] for v in self.mesh.verts])
        mesh_bot = min([v.co[2] for v in self.mesh.verts])
        mesh_height = mesh_top - mesh_bot
        return mesh_height

    def compute_scaling_factor_for_target_model_height(self, target_height):
        mesh_height = self.compute_mesh_height()
        if mesh_height <= 0:
            print("POLYZAMBONI WARNING: Mesh has zero height!")
            return 1
        return target_height / mesh_height

    def create_print_data_for_all_components(self, obj : bpy.types.Object, scaling_factor):
        self.mesh.edges.ensure_lookup_table()
        self.mesh.faces.ensure_lookup_table()

        all_print_data = []
        curr_connected_component : ConnectedComponent
        for c_id, curr_connected_component in self.connected_components.items():
            faces_ids_in_component = curr_connected_component._face_index_set
            if faces_ids_in_component is None:
                continue
            unfolding_of_curr_component : Unfolding = curr_connected_component._unfolding
            if unfolding_of_curr_component is None:
                continue
            curr_component_print_data = ComponentPrintData()

            # collect material index of first face
            first_face_in_component : bmesh.types.BMFace = self.mesh.faces[list(faces_ids_in_component)[0]]
            curr_component_print_data.dominating_mat_index = first_face_in_component.material_index

            # collect cut edges, fold edges and triangles of all faces
            fold_edge_index_set = set()
            face_cog_scores = {}
            for face_index in faces_ids_in_component:
                curr_face : bmesh.types.BMFace = self.mesh.faces[face_index]

                # map edges to correct halfedges
                self.ensure_halfedge_to_edge_table()
                edge_to_correct_halfedge_map = {}
                face_vertex_loop = list(curr_face.verts)
                for v_i in range(len(face_vertex_loop)):
                    v_j = (v_i + 1) % len(face_vertex_loop)
                    curr_e = self.halfedge_to_edge[(face_vertex_loop[v_i].index, face_vertex_loop[v_j].index)]
                    edge_to_correct_halfedge_map[curr_e.index] = (face_vertex_loop[v_i], face_vertex_loop[v_j])

                face_cog = np.mean([scaling_factor * unfolding_of_curr_component.get_globally_consistend_2d_coord_in_face(v.co, face_index) for v in face_vertex_loop], axis=0)

                # collect all edges
                dist_cog_edge_sum = 0
                for curr_edge in curr_face.edges:
                    # compute edge coords in unfolding space
                    vertex_coords_3d = [v.co for v in edge_to_correct_halfedge_map[curr_edge.index]]    
                    vertex_coords_unfolded = [scaling_factor * unfolding_of_curr_component.get_globally_consistend_2d_coord_in_face(co_3d, face_index) for co_3d in vertex_coords_3d]
                    dist_cog_edge_sum += signed_point_dist_to_line(face_cog, vertex_coords_unfolded[0], vertex_coords_unfolded[1])
                    if curr_edge.is_boundary or self.mesh_edge_is_cut(curr_edge):
                        # this is a cut edge
                        curr_component_print_data.add_cut_edge(CutEdgeData(tuple(vertex_coords_unfolded), curr_edge.index))
                    elif curr_edge.index not in fold_edge_index_set:
                        # this is a fold edge
                        fold_edge_index_set.add(curr_edge.index)
                        curr_component_print_data.add_fold_edge(FoldEdgeData(tuple(vertex_coords_unfolded), curr_edge.is_convex, curr_edge.calc_face_angle()))
                # set cog score
                face_cog_scores[face_index] = dist_cog_edge_sum / len(curr_face.edges)

                # get face material info
                mat_slots = obj.material_slots
                color = None
                text_path = None
                if curr_face.material_index < len(mat_slots):
                    curr_material_slot : bpy.types.MaterialSlot = mat_slots[curr_face.material_index]
                    curr_material : bpy.types.Material = curr_material_slot.material
                    color = curr_material.diffuse_color

                    # try to get color and texture image path from node tree
                    if curr_material.use_nodes:
                        for node in curr_material.node_tree.nodes:
                            if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
                                color = np.array(node.inputs['Base Color'].default_value)
                            if not isinstance(node, bpy.types.ShaderNodeTexImage):
                                continue
                            if not node.image:
                                continue                            
                            full_path = bpy.path.abspath(node.image.filepath, library=node.image.library)
                            norm_path = os.path.normpath(full_path)
                            text_path = norm_path

                # collect all faces
                triangles_in_unfolding_space = [tuple(scaling_factor * np.asarray(tri_coords)) for tri_coords in unfolding_of_curr_component.triangulated_faces_2d[face_index]]
                
                # uvs
                uvs_available = len(obj.data.uv_layers) > 0
                uv_layer = self.mesh.loops.layers.uv.verify()
                v_id_to_uv_in_face = {}

                for loop in curr_face.loops:
                    v_id_to_uv_in_face[loop.vert.index] = loop[uv_layer].uv

                triangle_uvs = [tuple([v_id_to_uv_in_face[v_id] for v_id in triangle_indices]) if uvs_available else None for triangle_indices in unfolding_of_curr_component.triangulation_indices[face_index]]

                for tri_coords, tri_uv in zip(triangles_in_unfolding_space, triangle_uvs):
                    curr_component_print_data.add_texured_triangle(ColoredTriangleData(tri_coords, tri_uv, text_path, color))

                # collect edges for glue flaps
                for flap_triangles in unfolding_of_curr_component.glue_flaps_per_face[face_index].values():
                    assert len(flap_triangles) > 0
                    curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((scaling_factor * flap_triangles[0][0],scaling_factor * flap_triangles[0][1])))
                    curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((scaling_factor * flap_triangles[0][1],scaling_factor * flap_triangles[0][2])))
                    if len(flap_triangles) == 2:
                        curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((scaling_factor * flap_triangles[1][1],scaling_factor * flap_triangles[1][2])))

            face_with_step_number = list(sorted(faces_ids_in_component, key=lambda face_index : face_cog_scores[face_index], reverse=True))[0] # sorting in the end is a bit meh but whatever
            step_number_pos = np.mean([scaling_factor * unfolding_of_curr_component.get_globally_consistend_2d_coord_in_face(v.co, face_with_step_number) for v in self.mesh.faces[face_with_step_number].verts], axis=0)

            curr_component_print_data.build_step_number_position = step_number_pos
            if curr_connected_component._build_step_number is not None:
                curr_component_print_data.build_step_number = curr_connected_component._build_step_number

            all_print_data.append(curr_component_print_data)
        return all_print_data
