"""
This file contains all functions neccessary to compute isometric uv maps of paper mesh pieces.
All functions here are only allowed to read data from meshes but not to write it back!
"""

import numpy as np
import networkx as nx
from bpy.types import Mesh
from bmesh.types import BMesh, BMFace
from itertools import product

from .properties import ZamboniGeneralMeshProps
from . import io
from . import geometry
from . import utils


def compute_all_face_triangulation_indices(bm : BMesh):
    triangulation_indices_per_face = {}
    for face in bm.faces:
        _, tri_ids = geometry.triangulate_3d_polygon([v.co for v in face.verts], face.normal, [v.index for v in face.verts], True)
        triangulation_indices_per_face[face.index] = tri_ids
    return triangulation_indices_per_face

def compute_tree_traversal(mesh : Mesh, face_set : set, dual_graph : nx.Graph,
                           edge_constraints = None):
    """ The given faces must be a tree in the dual graph with cuts applied """

    zamboni_props : ZamboniGeneralMeshProps = mesh.polyzamboni_general_mesh_props
    if edge_constraints is None:
        edge_constraints = io.read_edge_constraints_dict(mesh)
        assert edge_constraints is not None
    
    def f_in_faceset(f):
        return f in face_set
    def edge_is_not_cut(v1, v2):
        return not utils.mesh_edge_is_cut(dual_graph.edges[(v1, v2)]["edge_index"], edge_constraints, zamboni_props.use_auto_cuts)
    
    subgraph_with_cuts_applied = nx.subgraph_view(dual_graph, filter_node=f_in_faceset, filter_edge = edge_is_not_cut)
    assert nx.connected.is_connected(subgraph_with_cuts_applied)
    assert nx.is_tree(subgraph_with_cuts_applied)
    
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
            pred_dict[nb_f] = (curr_f, dual_graph.edges[(curr_f, nb_f)]["edge_index"])
            q.append(nb_f)

    assert len(tree_traversal) == len(face_set)
    return tree_traversal, pred_dict
    
def compute_local_coordinate_system_with_all_transitions_to_it(bmesh : BMesh, face : BMFace):
    inner_affine_transforms = {}
    face_normal = face.normal
    face_vertices_ccw = face.verts
    assert len(face_vertices_ccw) > 2
    # Choose one coordinate system for the face
    local_2d_coord_system = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[0].co, face_vertices_ccw[1].co, face_normal)
    inner_affine_transforms[bmesh.edges.get([face_vertices_ccw[0], face_vertices_ccw[1]]).index] = geometry.AffineTransform2D()
    # Compute affine transformations from all other edges to the first one
    for i in range(1, len(face_vertices_ccw)):
        j = (i + 1) % len(face_vertices_ccw)
        curr_basis = geometry.construct_2d_space_along_face_edge(face_vertices_ccw[i].co, face_vertices_ccw[j].co, face_normal)
        transition = geometry.affine_2d_transformation_between_two_2d_spaces_on_same_plane(local_2d_coord_system, curr_basis)
        inner_affine_transforms[bmesh.edges.get([face_vertices_ccw[i], face_vertices_ccw[j]]).index] = transition
    return local_2d_coord_system, inner_affine_transforms

def compute_2d_unfolded_triangles_of_component(mesh : Mesh, bmesh : BMesh, face_list, pred_dict, skip_intersection_test=False,
                                               face_triangulation_indices_dict = None, 
                                               inner_transform_data_per_face = None,
                                               local_2d_coord_system_per_face = None):
    # try to read data that is missing
    if face_triangulation_indices_dict is None:
        face_triangulation_indices_dict = io.read_triangulation_indices_per_face(mesh)
        assert face_triangulation_indices_dict is not None
    if inner_transform_data_per_face is None:
        inner_transform_data_per_face = io.read_inner_face_affine_transforms(mesh)
        assert inner_transform_data_per_face is not None
    if local_2d_coord_system_per_face is None:
        local_2d_coord_system_per_face = io.read_local_coordinate_systems_per_face(mesh)
        assert local_2d_coord_system_per_face is not None

    #yeet
    unfolded_triangulated_faces = {}
    affine_transform_to_root_coord_system_per_face = {}
    intersection_occured = False

    # propagate affine transformations through the face tree
    bmesh.edges.ensure_lookup_table()
    for processed_face_i, face_index in enumerate(face_list):
        face = bmesh.faces[face_index]
        pred, connecting_edge_id = pred_dict[face_index]

        if pred == face_index:
            # root face
            affine_transform_to_root_coord_system_per_face[face_index] = geometry.AffineTransform2D()
            verts_in_local_space_curr = {v.index : geometry.to_local_coords(v.co, *local_2d_coord_system_per_face[face_index]) for v in face.verts}
            unfolded_triangulated_faces[face_index] = [tuple([verts_in_local_space_curr[i] for i in tri_indices]) for tri_indices in face_triangulation_indices_dict[face_index]]
            continue
        
        # compute affine transformation to the root face coordinates
        connecting_edge = bmesh.edges[connecting_edge_id]
        pred_to_root : geometry.AffineTransform2D = affine_transform_to_root_coord_system_per_face[pred]
        pred_connecting_edge_to_local_2d : geometry.AffineTransform2D = inner_transform_data_per_face[pred][connecting_edge_id]
        curr_local_2d_to_connecting_edge : geometry.AffineTransform2D = inner_transform_data_per_face[face_index][connecting_edge_id].inverse()
        transform_at_edge = geometry.AffineTransform2D(-np.eye(2), np.array([connecting_edge.calc_length(), 0]))
        combined_transform = pred_to_root @ (pred_connecting_edge_to_local_2d @ (transform_at_edge @ curr_local_2d_to_connecting_edge))
        affine_transform_to_root_coord_system_per_face[face_index] = combined_transform

        # compute triangles in root face coords
        verts_in_local_space_root = {v.index : combined_transform * geometry.to_local_coords(v.co, *local_2d_coord_system_per_face[face_index]) for v in face.verts}
        unfolded_triangulated_faces[face_index] = [tuple([verts_in_local_space_root[i] for i in tri_indices]) for tri_indices in face_triangulation_indices_dict[face_index]]

        # check for any intersections 
        if skip_intersection_test or intersection_occured:
            continue # skip the test

        for other_face_i in range(processed_face_i):
            if intersection_occured:
                break
            other_face_index = face_list[other_face_i]
            for triangle_a, triangle_b in product(unfolded_triangulated_faces[other_face_index], unfolded_triangulated_faces[face_index]):
                if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                    intersection_occured = True
                    break

    return unfolded_triangulated_faces, affine_transform_to_root_coord_system_per_face, intersection_occured

def get_globally_consistent_2d_coord_in_face(mesh : Mesh, point_on_face_3d, face_index, component_id,
                                             local_coordinate_systems = None, affine_transforms_to_root = None):
        """ Maps a 3D point on a given face to the unfolded face in 2D """
        face_cs = local_coordinate_systems[face_index] if local_coordinate_systems is not None else io.read_local_coordinate_system_of_face(mesh, face_index)
        face_transform_to_root = affine_transforms_to_root[component_id][face_index] if affine_transforms_to_root is not None else io.read_affine_transform_to_roots_of_face_in_component(mesh, component_id, face_index)
        return face_transform_to_root * geometry.to_local_coords(point_on_face_3d, *face_cs[face_index])

def compute_unfolding_data_of_all_components(mesh : Mesh, bm : BMesh,
                                             dual_graph = None,
                                             edge_constraints = None,
                                             connected_components = None,
                                             cyclic_components = None,
                                             face_triangulation_indices_dict = None, 
                                             inner_transform_data_per_face = None,
                                             local_2d_coord_system_per_face = None):
    if dual_graph is None:
        dual_graph = utils.construct_dual_graph_from_bmesh(bm)
    if edge_constraints is None:
        edge_constraints = io.read_edge_constraints_dict(mesh)
        assert edge_constraints is not None
    if connected_components is None:
        connected_components = io.read_connected_component_sets(mesh)
        assert connected_components is not None
    if cyclic_components is None:
        cyclic_components = io.read_components_with_cycles_set(mesh)
        assert cyclic_components is not None

    unfolded_triangles_per_face_per_component = {}
    affine_transform_to_root_cood_system_per_face_per_component = {}
    components_with_intersections = set()

    for component_id, component_faces in connected_components.items():
        if component_id in cyclic_components:
            continue
        tree_traversal, pred_dict = compute_tree_traversal(mesh, component_faces, dual_graph, edge_constraints)
        unfolded_triangulated_faces, affine_transform_to_root_coord_system_per_face, intersection_occured = compute_2d_unfolded_triangles_of_component(mesh, bm, tree_traversal, pred_dict, False, 
                                                                                                                                                       face_triangulation_indices_dict, 
                                                                                                                                                       inner_transform_data_per_face, 
                                                                                                                                                       local_2d_coord_system_per_face)
        unfolded_triangles_per_face_per_component[component_id] = unfolded_triangulated_faces
        affine_transform_to_root_cood_system_per_face_per_component[component_id] = affine_transform_to_root_coord_system_per_face
        if intersection_occured:
            components_with_intersections.add(component_id)
    return unfolded_triangles_per_face_per_component, affine_transform_to_root_cood_system_per_face_per_component, components_with_intersections

def test_if_two_touching_unfolded_components_overlap(mesh : Mesh, component_id_1, component_id_2, join_face_index_1, join_face_index_2, join_verts_1, join_verts_2,
                                                     local_2d_coord_system_per_face = None,
                                                     affine_transforms_to_root = None,
                                                     unfolded_triangles_dict = None):
    """ Test if two components undoldings would overlap if they were merged at the given faces. """

    # try to read data that is missing
    if local_2d_coord_system_per_face is None:
        local_2d_coord_system_per_face = io.read_local_coordinate_systems_per_face(mesh)
        assert local_2d_coord_system_per_face is not None
    if affine_transforms_to_root is None:
        affine_transforms_to_root = io.read_affine_transforms_to_roots(mesh)
        assert affine_transforms_to_root is not None
    if unfolded_triangles_dict is None:
        unfolded_triangles_dict = io.read_facewise_triangles_per_component(mesh)
        assert unfolded_triangles_dict is not None

    # compute affine transformation from unfolding 1 to unfolding 2

    # translations
    join_point_1 = get_globally_consistent_2d_coord_in_face(mesh, join_verts_1[1].co, join_face_index_1, component_id_1, local_2d_coord_system_per_face, affine_transforms_to_root)
    join_point_2 = get_globally_consistent_2d_coord_in_face(mesh, join_verts_2[1].co, join_face_index_2, component_id_2, local_2d_coord_system_per_face, affine_transforms_to_root)
    join_point_1_to_orig = geometry.AffineTransform2D(affine_part=-join_point_1)
    orig_to_join_point_2 = geometry.AffineTransform2D(affine_part=join_point_2)
    # rotation
    x_ax_1, y_ax_1 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_1, get_globally_consistent_2d_coord_in_face(mesh, join_verts_1[0].co, join_face_index_1, component_id_1, local_2d_coord_system_per_face, affine_transforms_to_root))
    x_ax_2, y_ax_2 = geometry.construct_orthogonal_basis_at_2d_edge(join_point_2, get_globally_consistent_2d_coord_in_face(mesh, join_verts_2[1].co, join_face_index_2, component_id_2, local_2d_coord_system_per_face, affine_transforms_to_root))
    basis_mat_1 = np.array([x_ax_1, y_ax_1]).T
    basis_mat_2 = np.array([x_ax_2, y_ax_2]).T
    rotate_edges_together = geometry.AffineTransform2D(linear_part=basis_mat_2 @ np.linalg.inv(basis_mat_1))
    # full transformation
    transform_first_unfolding = orig_to_join_point_2 @ rotate_edges_together @ join_point_1_to_orig

    for triangle_list_1 in unfolded_triangles_dict[component_id_1].values():
        unfolding_1_triangles = [tuple([transform_first_unfolding * tri_coord for tri_coord in triangle]) for triangle in triangle_list_1]
        for triangle_list_2 in unfolded_triangles_dict[component_id_2].values():
            for triangle_a, triangle_b in product(unfolding_1_triangles, triangle_list_2):
                if geometry.triangle_intersection_test_2d(*triangle_a, *triangle_b):
                    return True # they do overlap    
    return False

