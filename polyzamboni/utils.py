"""
A collection of utility functions used by polyzamboni
"""

import bpy
import bmesh
import numpy as np
import networkx as nx

def compute_mesh_height(mesh : bpy.types.Mesh):
    mesh_top = max([v.co[2] for v in mesh.vertices])
    mesh_bot = min([v.co[2] for v in mesh.vertices])
    mesh_height = mesh_top - mesh_bot
    return mesh_height

def compute_bmesh_height(bm : bmesh.types.BMesh):
    mesh_top = max([v.co[2] for v in bm.verts])
    mesh_bot = min([v.co[2] for v in bm.verts])
    mesh_height = mesh_top - mesh_bot
    return mesh_height

def mesh_edge_is_cut(mesh_edge_index, edge_constraints, use_auto_cuts):
    if mesh_edge_index in edge_constraints:
        if edge_constraints[mesh_edge_index] == "cut" or (use_auto_cuts and edge_constraints[mesh_edge_index] == "auto"):
            return True
    return False

def get_edge_indices_of_mesh_face(mesh : bpy.types.Mesh, face_index):
    return [loop.edge_index for loop in [mesh.loops[loop_index] for loop_index in mesh.polygons[face_index].loop_indices]]

def find_bmesh_edge_of_halfedge(bm : bmesh.types.BMesh, halfedge):
    """ Returns None in case the edge does not exist """
    bm.verts.ensure_lookup_table()
    return bm.edges.get([bm.verts[i] for i in halfedge])

def construct_halfedge_to_face_dict(mesh : bmesh.types.BMesh):
    """ Assumes that the given mesh is manifold and has no flipped normals! """
    halfedge_to_face = {}
    for face in mesh.faces:
        verts_ccw = list(v.index for v in face.verts)
        for i in range(len(verts_ccw)):
            j = (i + 1) % len(verts_ccw)
            halfedge_to_face[(verts_ccw[i], verts_ccw[j])] = face
    return halfedge_to_face

def construct_dual_graph_from_bmesh(bm : bmesh.types.BMesh):
    """ Tries to create a dual graph of the given mesh. Multi-edges are not allowed! Returns None is that case. """
    dual_graph = nx.Graph()
    for face in bm.faces:
        dual_graph.add_node(face.index)
    for face in bm.faces:
        curr_id = face.index
        for nb_id, connecting_edge in [(f.index, e) for e in face.edges for f in e.link_faces if f is not face]:
            if (curr_id, nb_id) in dual_graph.edges and dual_graph.edges[(curr_id, nb_id)]["mesh_edge_index"] != connecting_edge.index:
                print("POLYZAMBONI ERROR: Multi edges in dual graph detected.")
                return None # Multi edge detected!
            dual_graph.add_edge(curr_id, nb_id, mesh_edge_index=connecting_edge.index)
    return dual_graph

def construct_dual_graph_from_mesh(mesh : bpy.types.Mesh):
    """ Tries to create a dual graph of the given mesh. Multi-edges are not allowed! Returns None is that case. """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    dual_graph = construct_dual_graph_from_bmesh(bm)
    bm.free()
    return dual_graph
