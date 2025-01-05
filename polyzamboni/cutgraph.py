import bpy
import bmesh
import numpy as np
import networkx as nx
import random
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME

class CutGraph():
    """ This class can be attached to any Mesh object """

    def __init__(self, ao : bpy.types.Object):

        self.mesh : bmesh.types.BMesh = bmesh.new()
        bpy.ops.object.mode_set(mode="EDIT")
        self.mesh = bmesh.from_edit_mesh(ao.data).copy()
        self.world_matrix = ao.matrix_world
        bpy.ops.object.mode_set(mode="OBJECT")

        self.dualgraph = nx.Graph()
        self.designer_constraints = {}
        if CUT_CONSTRAINTS_PROP_NAME in ao:
            for edge_index in ao[CUT_CONSTRAINTS_PROP_NAME]:
                self.designer_constraints[edge_index] = "cut"
        if LOCKED_EDGES_PROP_NAME in ao:
            for edge_index in ao[LOCKED_EDGES_PROP_NAME]:
                self.designer_constraints[edge_index] = "locked"

        for face in self.mesh.faces:
            self.dualgraph.add_node(face.index)
        for face in self.mesh.faces:
            curr_id = face.index
            for nb_id, connecting_edge in [(f.index, e) for e in face.edges for f in e.link_faces if f is not face]:
                self.dualgraph.add_edge(curr_id, nb_id, edge_index=connecting_edge.index)
        
        # for edge in self.dualgraph.edges:
        #     if random.choice([True, False]):
        #         self.manual_cuts.append(self.dualgraph.edges[edge]["edge_index"])

        print("Generated a dual graph with", self.dualgraph.number_of_nodes(), "nodes and", self.dualgraph.number_of_edges(), "edges.")
    
    def mesh_edge_id_list_to_coordinate_list(self, edge_ids):
        cut_coordinates = []
        self.mesh.edges.ensure_lookup_table()
        for edge_id in edge_ids:
            corresponding_mesh_edge = self.mesh.edges[edge_id]
            vert1, vert2 = corresponding_mesh_edge.verts
            cut_coordinates.append(self.world_matrix @ vert1.co)
            cut_coordinates.append(self.world_matrix @ vert2.co)
        return cut_coordinates

    def get_manual_cuts_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "cut"]
    
    def get_locked_edges_list(self):
        return [edge_index for edge_index in self.designer_constraints if self.designer_constraints[edge_index] == "locked"]