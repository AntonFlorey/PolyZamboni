import bpy
import bmesh
import numpy as np
import networkx as nx
import random
import mathutils
import queue
from .geometry import triangulate_3d_polygon, solve_for_weird_point
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME, PERFECT_REGION, BAD_GLUE_FLAPS_REGION, OVERLAPPING_REGION, NOT_FOLDABLE_REGION
from .unfolding import Unfolding

class CutGraph():
    """ This class can be attached to any Mesh object """

    def __init__(self, ao : bpy.types.Object):
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
        
        print("Generated a dual graph with", self.dualgraph.number_of_nodes(), "nodes and", self.dualgraph.number_of_edges(), "edges.")

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
        
        self.edge_via_verts = None # delete old edge via vert dict
        self.valid = True # make sure we check mesh validity again

    def edge_is_not_cut(self, v_from, v_to):
        mesh_edge_index = self.dualgraph.edges[(v_from, v_to)]["edge_index"]
        if mesh_edge_index in self.designer_constraints and self.designer_constraints[mesh_edge_index] == "cut":
            return False
        return True

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

    def update_connected_components_around_edge(self, mesh_edge_index):
        self.mesh.edges.ensure_lookup_table()

        changed_edge = self.mesh.edges[mesh_edge_index]
        linked_face_ids = [f.index for f in changed_edge.link_faces]
        if len(linked_face_ids) == 1:
            return
        assert len(linked_face_ids) == 2
        linked_component_ids = [self.vertex_to_component_dict[mesh_f] for mesh_f in linked_face_ids]

        def vertex_in_component(v):
            for c_id in linked_component_ids:
                if v in self.components_as_sets[c_id]:
                    return True
            return False

        subgraph_with_cuts_applied = nx.subgraph_view(self.dualgraph, filter_node=vertex_in_component, filter_edge = self.edge_is_not_cut)
        first_component = nx.node_connected_component(subgraph_with_cuts_applied, linked_face_ids[0])
        second_component = nx.node_connected_component(subgraph_with_cuts_applied, linked_face_ids[1])
        
        if linked_component_ids[0] == linked_component_ids[1] and linked_face_ids[1] not in first_component:
            # Split component in two
            self.components_as_sets[linked_component_ids[0]] = first_component
            self.components_as_sets[self.next_free_component_id] = second_component
            for v in second_component:
                self.vertex_to_component_dict[v] = self.next_free_component_id
            self.next_free_component_id += 1
        if linked_component_ids[0] != linked_component_ids[1] and linked_component_ids[1] in first_component:
            # Merge components
            self.components_as_sets[linked_component_ids[0]] = first_component
            for v in second_component:
                self.vertex_to_component_dict[v] = linked_component_ids[0]

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

            for nb_f in self.dualgraph.neighbors(curr_f):
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

        print("updating unfolding of components:", components_to_update)

        for c_id in components_to_update:
            self.unfolded_components[c_id] = self.unfold_connected_component(c_id)

        # assert that we got all components that changed this 
        for c_id in self.components_as_sets.keys():
            assert c_id in self.unfolded_components.keys()

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

    def ensure_edge_lookup_via_verts_table(self):
        if hasattr(self, "edge_via_verts") and self.edge_via_verts is not None:
            return
        self.edge_via_verts = {}
        for edge in self.mesh.edges:
            self.edge_via_verts[tuple([v.index for v in edge.verts])] = edge
            self.edge_via_verts[tuple(reversed([v.index for v in edge.verts]))] = edge

    def get_polygon_outline_for_face_drawing(self, face_index, edge_margin = 0.02):
        self.ensure_edge_lookup_via_verts_table()
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

            e_to_curr : bmesh.types.BMEdge = self.edge_via_verts[(prev_v.index, curr_v.index)]
            e_from_curr : bmesh.types.BMEdge = self.edge_via_verts[(curr_v.index, next_v.index)]

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
            # TODO add glue flaps intersection test
            quality_dict[PERFECT_REGION] += triangles
        return all_v_positions, quality_dict

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
            cut_this_edge = False
            for f in self.mesh.edges[e_id].link_faces:
                if f.index not in face_set:
                    cut_this_edge = True
                    break
            if cut_this_edge:
                self.designer_constraints[e_id] = "cut"
            else:
                self.designer_constraints[e_id] = "glued"

    def get_manual_cuts_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "cut"]

    def get_locked_edges_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "glued"]



