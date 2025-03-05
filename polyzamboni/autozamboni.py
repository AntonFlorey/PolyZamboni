from bpy.types import Mesh
import bmesh
import numpy as np
import time
import copy

from .unfolding import test_if_two_touching_unfolded_components_overlap
from . import operators_backend
from .properties import ZamboniGeneralMeshProps

axis_dict = {
    "X" : np.array([1.0, 0.0, 0.0]),
    "Y" : np.array([0.0, 1.0, 0.0]),
    "Z" : np.array([0.0, 0.0, 1.0])
}

def greedy_auto_cuts(mesh : Mesh, quality_level="NO_OVERLAPS_ALLOWED", target_loop_axis="Z", max_faces_per_component=10):
    """ Generates auto-cuts by first cutting all edges and then removing as many cuts as possible. 
    
    Parameters
    ----------

    cutgraph: The CutGraph to add auto cuts to

    quality_level: String in ["NO_OVERLAPS_ALLOWED", "GLUE_FLAP_OVERLAPS_ALLOWED", "ALL_OVERLAPS_ALLOWED"]

    target_loop_axis: Sting in ["X", "Y", "Z"]

    """

    # timers 
    algo_start_time = time.time()
    init_time = 0
    init_start_time = time.time()

    # read all paper model data
    edit_data = operators_backend.DataForEditing.from_mesh(mesh)

    # First, add auto cuts to all free edges
    free_edges = []
    for edge in edit_data.bm.edges:
        if edge.index in edit_data.edge_constraints.keys() and edit_data.edge_constraints[edge.index] != "auto":
            continue
        edit_data.edge_constraints[edge.index] = "auto"
        free_edges.append(edge)

    # sort all edges to process by their alignment to the target-loop-axis
    loops_axis_vec = axis_dict[target_loop_axis]
    def edge_alignment_to_axis_vec(e : bmesh.types.BMEdge):
        e_dir = np.array(e.verts[0].co - e.verts[1].co)
        return abs(np.dot(loops_axis_vec, e_dir / np.linalg.norm(e_dir)))

    sorted_autocut_edges = sorted(free_edges, key=edge_alignment_to_axis_vec, reverse=True)

    # recompute components, unfoldings, flaps
    edit_data.use_auto_cuts = True
    zamboni_props : ZamboniGeneralMeshProps = mesh.polyzamboni_general_mesh_props
    zamboni_props.use_auto_cuts = True
    operators_backend._update_paper_model(edit_data)
    init_time = time.time() - init_start_time

    # Try to remove auto cuts
    edge : bmesh.types.BMEdge
    for iter, edge in enumerate(sorted_autocut_edges):
        auto_cut_progress = iter / len(sorted_autocut_edges)
        yield auto_cut_progress
        # boundary edges do not have to be cut
        if edge.is_boundary:
            operators_backend._clear_edge_constraints(edit_data, [edge.index])
            continue
        # look ahead to see if we can remove the cut
        
        # get connected component
        linked_face_ids = [f.index for f in edge.link_faces]
        assert len(linked_face_ids) == 2
        linked_component_ids = [edit_data.face_to_component_dict[f_index] for f_index in linked_face_ids]
        # check for cyclic components
        if linked_component_ids[0] == linked_component_ids[1]:
            continue
        faces_in_component_1 : set = edit_data.connected_components[linked_component_ids[0]]
        faces_in_component_2 : set = edit_data.connected_components[linked_component_ids[1]]
        component_union = faces_in_component_1.union(faces_in_component_2)
        if len(component_union) > max_faces_per_component:
            continue # component would be to large!
        # check for overlapping mesh pieces
        halfedge_1 = tuple(edge.verts)
        halfedge_2 = tuple(reversed(halfedge_1))
        face_index_1 = edit_data.halfedge_to_face_dict[tuple([v.index for v in halfedge_1])].index
        face_index_2 = edit_data.halfedge_to_face_dict[tuple([v.index for v in halfedge_2])].index
        component_index_1 = edit_data.face_to_component_dict[face_index_1]
        component_index_2 = edit_data.face_to_component_dict[face_index_2]
        if len(edit_data.unfolded_triangles[component_index_1].keys()) < len(edit_data.unfolded_triangles[component_index_2].keys()):
            merging_produces_overlaps = test_if_two_touching_unfolded_components_overlap(mesh, component_index_1, component_index_2, face_index_1, face_index_2, halfedge_1, halfedge_2,
                                                                                         edit_data.local_coords_per_face, edit_data.affine_transforms_to_root, edit_data.unfolded_triangles)
        else:
            merging_produces_overlaps = test_if_two_touching_unfolded_components_overlap(mesh,component_index_2, component_index_1, face_index_2, face_index_1, halfedge_2, halfedge_1,
                                                                                         edit_data.local_coords_per_face, edit_data.affine_transforms_to_root, edit_data.unfolded_triangles)
        if merging_produces_overlaps and not quality_level == "ALL_OVERLAPS_ALLOWED":
            continue # no mesh overlaps allowed


        # remove the auto cut
        assert edge.index in edit_data.edge_constraints.keys()
        del edit_data.edge_constraints[edge.index]

        # update the paper model
        next_free_id = operators_backend._update_connected_components_around_edge(edit_data.bm, edit_data.dual_graph, edge.index, edit_data.connected_components, 
                                                                                  edit_data.face_to_component_dict, edit_data.cyclic_components, edit_data.outdated_components, 
                                                                                  edit_data.overlapping_components, edit_data.edge_constraints, edit_data.use_auto_cuts, 
                                                                                  edit_data.next_free_component_index)
        edit_data.next_free_component_index = next_free_id
        # unfoldings
        operators_backend._update_unfoldings_along_edges(edit_data.bm, edit_data.dual_graph, [edge.index], edit_data.edge_constraints, edit_data.use_auto_cuts, 
                                                         edit_data.connected_components, edit_data.face_to_component_dict, edit_data.cyclic_components, edit_data.face_to_component_dict, 
                                                         edit_data.inner_affine_transforms, edit_data.local_coords_per_face, edit_data.unfolded_triangles, edit_data.affine_transforms_to_root, 
                                                         edit_data.overlapping_components)
        # glue flaps
        no_flap_overlaps = operators_backend._greedy_update_flaps_around_changed_components(edit_data.bm, [edge.index], edit_data.flap_angle, edit_data.flap_height, edit_data.use_auto_cuts, 
                                                                                            edit_data.halfedge_to_face_dict, edit_data.face_to_component_dict, edit_data.connected_components, 
                                                                                            edit_data.affine_transforms_to_root, edit_data.inner_affine_transforms,edit_data.unfolded_triangles, 
                                                                                            edit_data.edge_constraints, edit_data.cyclic_components, edit_data.zigzag_flaps, edit_data.glueflap_dict,
                                                                                            edit_data.glueflap_geometry, edit_data.glueflap_collisions)
        # check for overlapping glue flaps
        if not no_flap_overlaps and quality_level == "NO_OVERLAPS_ALLOWED":
            # oh no, revert the decision
            operators_backend._write_edge_constraints(edit_data, [edge.index], "auto")

    # write the result back to the mesh
    edit_data.write_back_my_data(mesh)

    # timings
    total_time = time.time() - algo_start_time
    print("Greedy Auto Cuts total time:", total_time)
    if total_time > 0:
        print("Init time: {:.2f}s ({:.1f}%)".format(init_time, 100 * init_time / total_time))

    yield 1.0
    