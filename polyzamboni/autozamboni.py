import bmesh
import numpy as np
import time

from .cutgraph import CutGraph
from .unfolding import Unfolding, test_if_two_touching_unfolded_components_overlap

axis_dict = {
    "X" : np.array([1.0, 0.0, 0.0]),
    "Y" : np.array([0.0, 1.0, 0.0]),
    "Z" : np.array([0.0, 0.0, 1.0])
}

def greedy_auto_cuts(cutgraph : CutGraph, quality_level="NO_OVERLAPS_ALLOWED", target_loop_axis="Z", max_faces_per_component=10):
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
    component_computation_time = 0
    unfoldings_time = 0
    flaps_time = 0
    undo_time = 0

    init_start_time = time.time()
    # First, add auto cuts to all free edges
    free_edges = []
    for edge in cutgraph.mesh.edges:
        if edge.index in cutgraph.designer_constraints.keys() and cutgraph.designer_constraints[edge.index] != "auto":
            continue
        cutgraph.designer_constraints[edge.index] = "auto"
        free_edges.append(edge)

    # sort all edges to process by their alignment to the target-loop-axis
    loops_axis_vec = axis_dict[target_loop_axis]
    def edge_alignment_to_axis_vec(e : bmesh.types.BMEdge):
        e_dir = np.array(e.verts[0].co - e.verts[1].co)
        return abs(np.dot(loops_axis_vec, e_dir / np.linalg.norm(e_dir)))

    sorted_autocut_edges = sorted(free_edges, key=edge_alignment_to_axis_vec, reverse=True)

    # recompute components, unfoldings, flaps
    use_auto_cuts_given_settings = cutgraph.use_auto_cuts
    cutgraph.use_auto_cuts = True
    cutgraph.compute_all_connected_components()
    cutgraph.update_unfoldings_along_edges([e.index for e in sorted_autocut_edges])
    cutgraph.greedy_update_flaps_around_changed_components([e.index for e in sorted_autocut_edges])
    init_time = time.time() - init_start_time

    # Try to remove auto cuts
    edge : bmesh.types.BMEdge
    for iter, edge in enumerate(sorted_autocut_edges):
        auto_cut_progress = iter / len(sorted_autocut_edges)
        yield auto_cut_progress
        # boundary edges do not have to be cut
        if edge.is_boundary:
            cutgraph.clear_edge_constraint(edge.index)
            continue
        # look ahead to see if we can remove the cut
        
        # get connected component
        linked_face_ids = [f.index for f in edge.link_faces]
        assert len(linked_face_ids) == 2
        linked_component_ids = [cutgraph.vertex_to_component_dict[mesh_f] for mesh_f in linked_face_ids]
        # check for cyclic components
        if linked_component_ids[0] == linked_component_ids[1]:
            continue 
        component_union = cutgraph.components_as_sets[linked_component_ids[0]].union(cutgraph.components_as_sets[linked_component_ids[1]])
        if len(component_union) > max_faces_per_component:
            continue # component would be to large!
        # check for overlapping mesh pieces
        halfedge_1 = tuple(edge.verts)
        halfedge_2 = tuple(reversed(halfedge_1))
        cutgraph.ensure_halfedge_to_face_table()
        face_index_1 = cutgraph.halfedge_to_face[tuple([v.index for v in halfedge_1])].index
        face_index_2 = cutgraph.halfedge_to_face[tuple([v.index for v in halfedge_2])].index
        unfolding_1 : Unfolding = cutgraph.unfolded_components[cutgraph.vertex_to_component_dict[face_index_1]]
        unfolding_2 : Unfolding = cutgraph.unfolded_components[cutgraph.vertex_to_component_dict[face_index_2]]
        if len(unfolding_1.triangulated_faces_2d.keys()) < len(unfolding_2.triangulated_faces_2d.keys()):
            merging_produces_overlaps = test_if_two_touching_unfolded_components_overlap(unfolding_1, unfolding_2, face_index_1, face_index_2, halfedge_1, halfedge_2)
        else:
            merging_produces_overlaps = test_if_two_touching_unfolded_components_overlap(unfolding_2, unfolding_1, face_index_2, face_index_1, halfedge_2, halfedge_1)
        if merging_produces_overlaps and not quality_level == "ALL_OVERLAPS_ALLOWED":
            continue # no mesh overlaps allowed

        # remove the auto cut
        cutgraph.clear_edge_constraint(edge.index)
        component_start_time = time.time()
        cutgraph.update_connected_components_around_edge(edge.index)
        component_computation_time += time.time() - component_start_time
        unfolding_start_time = time.time()
        cutgraph.update_unfoldings_along_edges([edge.index], skip_intersection_test=(not merging_produces_overlaps))
        unfoldings_time += time.time() - unfolding_start_time
        flaps_start_time = time.time()
        flaps_success = cutgraph.greedy_update_flaps_around_changed_components([edge.index])
        flaps_time += time.time() - flaps_start_time
        # check for overlapping glue flaps
        if not flaps_success and quality_level == "NO_OVERLAPS_ALLOWED":
            # oh no, revert the decision
            undo_start_time = time.time()
            cutgraph.designer_constraints[edge.index] = "auto"
            component_start_time = time.time()
            cutgraph.update_connected_components_around_edge(edge.index)
            component_computation_time += time.time() - component_start_time
            unfolding_start_time = time.time()
            cutgraph.update_unfoldings_along_edges([edge.index], skip_intersection_test=(not merging_produces_overlaps))
            unfoldings_time += time.time() - unfolding_start_time
            flaps_start_time = time.time()
            cutgraph.greedy_update_flaps_around_changed_components([edge.index])
            flaps_time += time.time() - flaps_start_time
            undo_time += time.time() - undo_start_time

    # compactify
    cutgraph.compactify_components()

    # restore input settings
    if not use_auto_cuts_given_settings:
        cutgraph.use_auto_cuts = use_auto_cuts_given_settings
        cutgraph.compute_all_connected_components()
        cutgraph.unfold_all_connected_components()
        cutgraph.greedy_place_all_flaps()

    # timings
    total_time = time.time() - algo_start_time
    print("Greedy Auto Cuts total time:", total_time)
    print("Init time: {:.2f}s ({:.1f}%)".format(init_time, 100 * init_time / total_time))
    print("Components time: {:.2f}s ({:.1f}%)".format(component_computation_time, 100 * component_computation_time / total_time))
    print("Unfoldings time: {:.2f}s ({:.1f}%)".format(unfoldings_time, 100 * unfoldings_time / total_time))
    print("Flaps time: {:.2f}s ({:.1f}%)".format(flaps_time, 100 * flaps_time / total_time))
    print("Undo time: {:.2f}s ({:.1f}%)".format(undo_time, 100 * undo_time / total_time))

    yield 1.0
    