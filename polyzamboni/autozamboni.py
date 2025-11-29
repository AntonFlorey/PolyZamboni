from bpy.types import Mesh
import bmesh
import numpy as np
import time

from .papermodel import PaperModel, ConnectedComponent

axis_dict = {
    "X" : np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "Y" : np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Z" : np.array([0.0, 0.0, 1.0], dtype=np.float64)
}

def greedy_auto_cuts(paper_model : PaperModel, quality_level="NO_OVERLAPS_ALLOWED", target_loop_axis="Z", max_faces_per_component=10):
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

    # First, add auto cuts to all free edges
    filled_edges = paper_model.fill_with_auto_cuts()

    # sort all edges to process by their alignment to the target-loop-axis
    loops_axis_vec = axis_dict[target_loop_axis]
    def edge_alignment_to_axis_vec(e : bmesh.types.BMEdge):
        e_dir = np.array(e.verts[0].co - e.verts[1].co, dtype=np.float64)
        return abs(np.dot(loops_axis_vec, e_dir / np.linalg.norm(e_dir)))
    sorted_autocut_edges = sorted(filled_edges, key=edge_alignment_to_axis_vec, reverse=True)
    init_time = time.time() - init_start_time

    # Try to remove auto cuts
    edge : bmesh.types.BMEdge
    for iter, edge in enumerate(sorted_autocut_edges):
        auto_cut_progress = iter / len(sorted_autocut_edges)
        yield auto_cut_progress
        # boundary edges do not have to be cut
        if edge.is_boundary:
            paper_model.clear_edges([edge.index])
            continue
        # look ahead to see if we can remove the cut
        linked_face_ids = [f.index for f in edge.link_faces]
        assert len(linked_face_ids) == 2
        linked_component_ids = [paper_model.face_to_component_index_dict[f_index] for f_index in linked_face_ids]
        if linked_component_ids[0] == linked_component_ids[1]:
            continue  # component would be cyclic
        faces_in_component_1 : set = paper_model.connected_components[linked_component_ids[0]].face_index_set
        faces_in_component_2 : set = paper_model.connected_components[linked_component_ids[1]].face_index_set
        component_union = faces_in_component_1.union(faces_in_component_2)
        if len(component_union) > max_faces_per_component:
            continue # component would be to large!
        # check for overlapping mesh pieces
        halfedge_1 = tuple(edge.verts)
        halfedge_2 = tuple(reversed(halfedge_1))
        face_index_1 = paper_model.halfedge_to_face_dict[tuple([v.index for v in halfedge_1])].index
        face_index_2 = paper_model.halfedge_to_face_dict[tuple([v.index for v in halfedge_2])].index
        component_index_1 = paper_model.face_to_component_index_dict[face_index_1]
        component_index_2 = paper_model.face_to_component_index_dict[face_index_2]
        component_1 : ConnectedComponent = paper_model.connected_components[component_index_1]
        component_2 : ConnectedComponent = paper_model.connected_components[component_index_2]
        if len(component_1.unfolded_face_geometry.keys()) < len(component_2.unfolded_face_geometry.keys()):
            merging_produces_overlaps = paper_model.test_if_two_touching_unfolded_components_overlap(component_index_1, component_index_2, face_index_1, face_index_2, halfedge_1, halfedge_2)
        else:
            merging_produces_overlaps = paper_model.test_if_two_touching_unfolded_components_overlap(component_index_2, component_index_1, face_index_2, face_index_1, halfedge_2, halfedge_1)
        if merging_produces_overlaps and not quality_level == "ALL_OVERLAPS_ALLOWED":
            continue # component would have overlapping mesh pieces

        # remove the auto cut
        no_overlapping_glueflaps_introduced = paper_model.clear_edges([edge.index], quality_level != "ALL_OVERLAPS_ALLOWED")
        # check for overlapping glue flaps
        if not no_overlapping_glueflaps_introduced and quality_level == "NO_OVERLAPS_ALLOWED":
            # oh no, revert the decision
            paper_model.auto_cut_edges([edge.index], quality_level != "ALL_OVERLAPS_ALLOWED")

    # timings
    total_time = time.time() - algo_start_time
    print("POLYZAMBONI INFO: Greedy Auto Cuts total time:", total_time)
    if total_time > 0:
        print("POLYZAMBONI INFO: Init time: {:.2f}s ({:.1f}%)".format(init_time, 100 * init_time / total_time))

    yield 1.0
    