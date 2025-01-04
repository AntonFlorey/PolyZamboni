import bpy
import bmesh
import numpy as np
import networkx as nx
import random

class CutGraph():
    """ This class can be attached to any Mesh object """

    def __init__(self, ao : bpy.types.Object):

        self.mesh : bmesh.types.BMesh = bmesh.new()
        bpy.ops.object.mode_set(mode="EDIT")
        self.mesh = bmesh.from_edit_mesh(ao.data).copy()
        self.world_matrix = ao.matrix_world
        bpy.ops.object.mode_set(mode="OBJECT")

        self.manual_cuts = ao["manual_cuts"].to_list() if "manual_cuts" in ao else []
        self.dualgraph = nx.Graph()

        for face in self.mesh.faces:
            self.dualgraph.add_node(face.index)
        for face in self.mesh.faces:
            curr_id = face.index
            for nb_id, connecting_edge in [(f.index, e) for e in face.edges for f in e.link_faces if f is not face]:
                self.dualgraph.add_edge(curr_id, nb_id, edge_index=connecting_edge.index)
        
        for edge in self.dualgraph.edges:
            if random.choice([True, False]):
                self.manual_cuts.append(self.dualgraph.edges[edge]["edge_index"])

        print("Generated a dual graph with", self.dualgraph.number_of_nodes(), "nodes and", self.dualgraph.number_of_edges(), "edges.")

    def get_user_provided_cuts_as_coordinate_list(self):
        cut_coordinates = []
        self.mesh.edges.ensure_lookup_table()
        for cut in self.manual_cuts:
            corresponding_mesh_edge = self.mesh.edges[cut]
            vert1, vert2 = corresponding_mesh_edge.verts
            cut_coordinates.append(self.world_matrix @ vert1.co)
            cut_coordinates.append(self.world_matrix @ vert2.co)
        return cut_coordinates

    def get_manual_cuts_list(self):
        return self.manual_cuts