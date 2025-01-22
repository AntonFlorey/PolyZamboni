import bpy
import bmesh
import numpy as np
import networkx as nx
import random
import mathutils
from .geometry import triangulate_3d_polygon, solve_for_weird_point
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME, PERFECT_REGION, BAD_GLUE_FLAPS_REGION, OVERLAPPING_REGION, NOT_FOLDABLE_REGION, GLUE_FLAP_NO_OVERLAPS, GLUE_FLAP_WITH_OVERLAPS, GLUE_FLAP_TO_LARGE
from .unfolding import Unfolding
from collections import deque

class CutGraph():
    """ This class can be attached to any Mesh object """

    def __init__(self, ao : bpy.types.Object, flap_angle = 1 / 4 * np.pi, flap_height = 1.0, prefer_zigzag_flaps = True):
        self.flap_angle = flap_angle
        self.flap_height = flap_height
        self.prefer_zigzag = prefer_zigzag_flaps
        self.construct_dual_graph_from_mesh(ao)
        self.designer_constraints = {}
        if CUT_CONSTRAINTS_PROP_NAME in ao:
            for edge_index in ao[CUT_CONSTRAINTS_PROP_NAME]:
                self.designer_constraints[edge_index] = "cut"
        if LOCKED_EDGES_PROP_NAME in ao:
            for edge_index in ao[LOCKED_EDGES_PROP_NAME]:
                self.designer_constraints[edge_index] = "glued"
        
        self.compute_all_connected_components()
        self.unfold_all_connected_components()
        self.greedy_place_all_flaps()

        print("Generated a dual graph with", self.dualgraph.number_of_nodes(), "nodes and", self.dualgraph.number_of_edges(), "edges.")

    #################################
    #     Input Mesh Handling       #
    #################################

    def construct_dual_graph_from_mesh(self, obj : bpy.types.Object):
        self.mesh : bmesh.types.BMesh = bmesh.new()
        bpy.ops.object.mode_set(mode="EDIT")
        self.mesh = bmesh.from_edit_mesh(obj.data).copy()
        self.world_matrix = obj.matrix_world
        bpy.ops.object.mode_set(mode="OBJECT")
        self.dualgraph = nx.Graph()
        for face in self.mesh.faces:
            self.dualgraph.add_node(face.index)
        for face in self.mesh.faces:
            curr_id = face.index
            for nb_id, connecting_edge in [(f.index, e) for e in face.edges for f in e.link_faces if f is not face]:
                if (curr_id, nb_id) in self.dualgraph.edges and self.dualgraph.edges[(curr_id, nb_id)]["edge_index"] != connecting_edge.index:
                    print("POLYZAMBONI ERROR: Please ensure that faces connect at at most one common edge!")
                self.dualgraph.add_edge(curr_id, nb_id, edge_index=connecting_edge.index)
        
        self.halfedge_to_edge = None # delete old halfedge to edge dict
        self.halfedge_to_face = None # delete old halfedge to face dict
        self.valid = True # make sure we check mesh validity again

    #################################
    #             Misc.             #
    #################################

    def check_validity(self):
        if not self.valid:
            return False
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

    def edge_is_not_cut(self, v_from, v_to):
        mesh_edge_index = self.dualgraph.edges[(v_from, v_to)]["edge_index"]
        if mesh_edge_index in self.designer_constraints and self.designer_constraints[mesh_edge_index] == "cut":
            return False
        return True
    
    def mesh_edge_is_not_cut(self, edge):
        if edge.index in self.designer_constraints and self.designer_constraints[edge.index] == "cut":
            return False
        return True

    def assert_all_faces_have_components_and_unfoldings(self):
        for f in self.dualgraph.nodes:
            assert f in self.vertex_to_component_dict.keys()
            c_id = self.vertex_to_component_dict[f]
            assert c_id in self.components_as_sets.keys()
            assert c_id in self.unfolded_components.keys()

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

    #################################
    #      Connected Components     #
    #################################

    def compute_all_connected_components(self):
        if self.mesh is None or self.dualgraph is None:
            print("Warning! Connected components can not yet be computed")
            return
        graph_with_cuts_applied = nx.subgraph_view(self.dualgraph, filter_edge = self.edge_is_not_cut)
        all_components = list(nx.connected_components(graph_with_cuts_applied))

        if hasattr(self, "vertex_to_component_dict"):
            old_vertex_to_component_dict = self.vertex_to_component_dict.copy()

        self.num_components = len(all_components)
        self.vertex_to_component_dict = {}
        for c_id, component in enumerate(all_components):
            for v in component:
                self.vertex_to_component_dict[v] = c_id
        self.components_as_sets = {c_id : component for c_id, component in enumerate(all_components)}
        self.next_free_component_id = self.num_components # TODO remove probably
        if not hasattr(self, "unfolded_components"):
            return
        # copy old unfoldings WARNING: these might not be correct anymore and need to be updated
        old_unfolded_components = self.unfolded_components.copy()
        for c_id, component in self.components_as_sets.items():
            self.unfolded_components[c_id] = old_unfolded_components[old_vertex_to_component_dict[list(component)[0]]]
        # do a sanity check
        self.assert_all_faces_have_components_and_unfoldings()

    #################################
    #           Unfolding           #
    #################################

    def unfold_connected_component(self, component_id):
        """Unfolds one connected component of the cut mesh into the 2D plane. Returns None if the component containts cycles."""
        def vertex_in_component(v):
            if v in self.components_as_sets[component_id]:
                return True
            return False
        subgraph_with_cuts_applied = nx.subgraph_view(self.dualgraph, filter_node=vertex_in_component, filter_edge = self.edge_is_not_cut)
        if not nx.is_tree(subgraph_with_cuts_applied):
            return None
        
        # custom tree data
        root = list(self.components_as_sets[component_id])[0]
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
                if self.vertex_to_component_dict[nb_f] != component_id:
                    continue
                pred_dict[nb_f] = (curr_f, self.dualgraph.edges[(curr_f, nb_f)]["edge_index"])
                q.append(nb_f)

        # unfold the component
        return Unfolding(self.mesh, tree_traversal, pred_dict)

    def unfold_all_connected_components(self):
        self.unfolded_components = {}
        for c_id in self.components_as_sets.keys():
            self.unfolded_components[c_id] = self.unfold_connected_component(c_id)

    def update_unfoldings_along_edges(self, edge_ids):
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
            self.unfolded_components[c_id] = self.unfold_connected_component(c_id)

        # assert that we got all components that changed this 
        for c_id in self.components_as_sets.keys():
            assert c_id in self.unfolded_components.keys()

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

    def get_polygon_outline_for_face_drawing(self, face_index, edge_margin = 0.02):
        self.ensure_halfedge_to_edge_table()
        face = self.mesh.faces[face_index]
        face_normal = face.normal
        face_component_id = self.vertex_to_component_dict[face_index] # vertex of dual graph is a face of the mesh.. confusing...
        verts = list(face.verts)
        cool_vertices = []
        for v_id in range(len(verts)):
            curr_v = verts[v_id]
            prev_v = verts[(v_id + len(verts) - 1) % len(verts)]
            next_v = verts[(v_id + 1) % len(verts)]

            prev_point = curr_v.co + 0.25 * (prev_v.co - curr_v.co)
            next_point = curr_v.co + 0.25 * (next_v.co - curr_v.co)

            v_on_component_boundary = curr_v.is_boundary or (np.array([self.vertex_to_component_dict[f.index] != face_component_id for f in curr_v.link_faces]).any())

            e_to_curr : bmesh.types.BMEdge = self.halfedge_to_edge[(prev_v.index, curr_v.index)]
            e_from_curr : bmesh.types.BMEdge = self.halfedge_to_edge[(curr_v.index, next_v.index)]

            e_prev_curr_is_cutting = e_to_curr.is_boundary or (e_to_curr.index in self.designer_constraints.keys() and self.designer_constraints[e_to_curr.index] == "cut")
            e_curr_to_next_is_cutting = e_from_curr.is_boundary or (e_from_curr.index in self.designer_constraints.keys() and self.designer_constraints[e_from_curr.index] == "cut")

            if v_on_component_boundary and not e_prev_curr_is_cutting and not e_curr_to_next_is_cutting:
                cool_vertices.append(prev_point + edge_margin * (next_v.co - curr_v.co))
                #cool_vertices.append(mathutils.Vector(solve_for_weird_point(prev_point, curr_v.co, next_point, face_normal) + edge_margin / 2 * (next_v.co + prev_v.co - 2 * curr_v.co)))
                #cool_vertices.append(curr_v.co + (0.25 + edge_margin) / 2 * (next_v.co + prev_v.co - 2 * curr_v.co))
                cool_vertices.append(next_point + edge_margin * (prev_v.co - curr_v.co))
            else:
                to_prev_offset = 0.25 if e_curr_to_next_is_cutting else edge_margin
                to_next_offset = 0.25 if e_prev_curr_is_cutting else edge_margin
                cool_vertices.append(curr_v.co + to_prev_offset * (prev_v.co - curr_v.co) + to_next_offset * (next_v.co - curr_v.co))

        return cool_vertices

    def get_connected_component_triangle_lists_for_drawing(self, face_offset):
        self.mesh.faces.ensure_lookup_table()
        triangles_per_component = {}

        all_vertex_positions = []

        for component_id, connected_component in self.components_as_sets.items():
            component_triangles = []
            for face_index in connected_component:
                polygon_outline = self.get_polygon_outline_for_face_drawing(face_index)
                curr_start_index = len(all_vertex_positions)
                all_vertex_positions += [self.world_matrix @ (v + face_offset * self.mesh.faces[face_index].normal) for v in polygon_outline]
                _, curr_triangle_ids =  triangulate_3d_polygon(polygon_outline, self.mesh.faces[face_index].normal, list(range(curr_start_index, len(all_vertex_positions))))
                component_triangles += curr_triangle_ids
            triangles_per_component[component_id] = component_triangles
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
            if self.unfolded_components[cluster_id] is None:
                quality_dict[NOT_FOLDABLE_REGION] += triangles
                continue
            unfolding : Unfolding = self.unfolded_components[cluster_id]
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
        self.assert_all_faces_have_components_and_unfoldings()
        self.ensure_halfedge_to_edge_table()
        self.ensure_halfedge_to_face_table()
        self.mesh.edges.ensure_lookup_table()
        quality_dict = {
            GLUE_FLAP_NO_OVERLAPS : [],
            GLUE_FLAP_WITH_OVERLAPS : [],
            GLUE_FLAP_TO_LARGE : []
            }
        for edge_index, halfedge in self.glue_flaps.items():
            opp_halfedge = (halfedge[1], halfedge[0])
            opp_face : bmesh.types.BMFace = self.halfedge_to_face[opp_halfedge]
            opp_unfolding : Unfolding = self.unfolded_components[self.vertex_to_component_dict[opp_face.index]]
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
            flap_line_array = [self.world_matrix @ (mathutils.Vector(v) + face_offset * opp_face.normal) for v in flap_line_array]
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
        edge = self.halfedge_to_edge[halfedge]
        face = self.halfedge_to_face[halfedge]
        unfold : Unfolding = self.unfolded_components[self.vertex_to_component_dict[face.index]]
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
            return preferred_halfedge
        
        # try out the other flap option
        if self.try_to_add_glue_flap((preferred_halfedge[1], preferred_halfedge[0])):
            return (preferred_halfedge[1], preferred_halfedge[0])

        # add the faulty but preferred flap
        self.try_to_add_glue_flap(preferred_halfedge, enforce=True)
        return preferred_halfedge

    def greedy_place_all_flaps(self):
        """ Attaches flaps to all cut edges (not on boundary edges) """

        self.ensure_halfedge_to_edge_table()
        self.ensure_halfedge_to_face_table()
        self.assert_all_faces_have_components_and_unfoldings()
        self.glue_flaps = {}
        # clear all flap info stored in the unfoldings
        for unfolding in self.unfolded_components.values():
            if unfolding is not None:
                unfolding.remove_all_flap_info()

        # build dfs tree of cut edges
        processed_edges = set()
        for edge in self.mesh.edges:
            if edge.index in processed_edges:
                continue
            if edge.is_boundary:
                processed_edges.add(edge.index)
                continue
            if self.mesh_edge_is_not_cut(edge):
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
                if not (curr_e.is_boundary or self.unfolded_components[self.vertex_to_component_dict[self.halfedge_to_face[(v0.index, v1.index)].index]] is None or self.unfolded_components[self.vertex_to_component_dict[self.halfedge_to_face[(v1.index, v0.index)].index]] is None):
                    preferred_halfedge = (v0.index, v1.index) if (self.prefer_zigzag and not flap_01) or (not self.prefer_zigzag and flap_01) else (v1.index, v0.index)
                    used_halfedge =  self.add_one_glue_flap(preferred_halfedge)
                    curr_flap_01 = used_halfedge == (v0.index, v1.index)
                    # print("flap added!")

                # collect next edges
                for nb_e in v1.link_edges:
                    nb_v = nb_e.other_vert(v1)
                    if nb_e.index in processed_edges:
                        # print("already visited edge", nb_e.index)
                        continue
                    if self.mesh_edge_is_not_cut(nb_e):
                        # print("not visiting edge", nb_e.index)
                        continue
                    dfs_stack.append((v1, nb_v, curr_flap_01))

    def remove_flaps_of_edge(self, edge : bmesh.types.BMEdge):
        self.ensure_halfedge_to_face_table()
        if edge.index in self.glue_flaps.keys():
            glue_he = self.glue_flaps[edge.index]
            face_with_flap = self.halfedge_to_face[glue_he]
            unfolding_with_flap : Unfolding = self.unfolded_components[self.vertex_to_component_dict[face_with_flap.index]]
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
        for face_set in [self.components_as_sets[c_id] for c_id in updated_components]:
            for face_index in face_set:
                for edge in self.mesh.faces[face_index].edges:
                    all_interesting_edge_ids.add(edge.index)


        self.ensure_halfedge_to_edge_table()
        self.ensure_halfedge_to_face_table()
        self.assert_all_faces_have_components_and_unfoldings()

        # build dfs tree of cut edges
        processed_edges = set()
        for edge_id in all_interesting_edge_ids:
            edge = self.mesh.edges[edge_id]

            if edge_id in processed_edges:
                continue
            if edge.is_boundary:
                processed_edges.add(edge.index)
                continue
            if self.mesh_edge_is_not_cut(edge):
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
                if not curr_e.is_boundary and (self.unfolded_components[self.vertex_to_component_dict[self.halfedge_to_face[(v0.index, v1.index)].index]] is None or self.unfolded_components[self.vertex_to_component_dict[self.halfedge_to_face[(v1.index, v0.index)].index]] is None):
                    # remove any flaps attached to this edge
                    self.remove_flaps_of_edge(curr_e)
                elif not curr_e.is_boundary:
                    # add a new glue flap 
                    if curr_e.index not in self.glue_flaps.keys():
                        preferred_halfedge = (v0.index, v1.index) if (self.prefer_zigzag and not flap_01) or (not self.prefer_zigzag and flap_01) else (v1.index, v0.index)
                        used_halfedge =  self.add_one_glue_flap(preferred_halfedge)
                        curr_flap_01 = used_halfedge == (v0.index, v1.index)
                    else:
                        # do nothing or restore old glue flap
                        he_with_flap = self.glue_flaps[curr_e.index]
                        face_with_flap = self.halfedge_to_face[he_with_flap]
                        unfold_with_flap : Unfolding = self.unfolded_components[self.vertex_to_component_dict[face_with_flap.index]]
                        if not unfold_with_flap.check_if_edge_has_flap(face_with_flap.index, curr_e):
                            flap_geometry = unfold_with_flap.compute_2d_glue_flap_triangles(face_with_flap.index, curr_e, self.flap_angle, self.flap_height)
                            unfold_with_flap.add_glue_flap_to_face_edge(face_with_flap.index, curr_e, flap_geometry)
                        curr_flap_01 = he_with_flap == (v0.index, v1.index)

                # collect next edges
                for nb_e in v1.link_edges:
                    nb_v = nb_e.other_vert(v1)
                    if nb_e.index in processed_edges:
                        # print("already visited edge", nb_e.index)
                        continue
                    if nb_e.index not in all_interesting_edge_ids:
                        continue
                    if self.mesh_edge_is_not_cut(nb_e):
                        # print("not visiting edge", nb_e.index)
                        continue
                    dfs_stack.append((v1, nb_v, curr_flap_01))

    def update_all_flap_geometry(self):
        """ This mehthod only changes flap geometry but not the flap placement """
        self.ensure_halfedge_to_edge_table()
        self.ensure_halfedge_to_face_table()
        self.mesh.edges.ensure_lookup_table()
        self.assert_all_faces_have_components_and_unfoldings()
        # clear all flap info stored in the unfoldings
        for unfolding in self.unfolded_components.values():
            if unfolding is not None:
                unfolding.remove_all_flap_info()

        for edge_id, halfedge in self.glue_flaps.items():
            face : bmesh.types.BMFace = self.halfedge_to_face[halfedge]
            unfolding : Unfolding = self.unfolded_components[self.vertex_to_component_dict[face.index]]
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
        curr_face = self.halfedge_to_face[curr_halfedge]
        curr_unfolding : Unfolding = self.unfolded_components[self.vertex_to_component_dict[curr_face.index]]
        curr_unfolding.remove_flap_from_edge(curr_face.index, self.mesh.edges[edge_id])
        # add glue flap on the opposide halfedge
        self.try_to_add_glue_flap((curr_halfedge[1], curr_halfedge[0]), enforce=True)
    
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
