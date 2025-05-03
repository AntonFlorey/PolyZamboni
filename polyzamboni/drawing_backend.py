"""
Functions in this file provide all data necessary for rendering feedback.
"""

from bpy.types import Mesh
from bmesh.types import BMesh
import numpy as np
import mathutils
import math
from enum import Enum
import time

from . import io
from . import utils
from .geometry import AffineTransform2D, triangulate_3d_polygon, face_corner_convex_3d, solve_for_weird_intersection_point
from .utils import mesh_edge_is_cut, find_bmesh_edge_of_halfedge
from .glueflaps import component_has_overlapping_glue_flaps, flap_is_overlapping, compute_3d_glue_flap_triangles_inside_face
from .printprepper import ComponentPrintData, CutEdgeData, FoldEdgeData, FoldEdgeAtGlueFlapData, ColoredTriangleData

def make_dotted_lines(line_array, target_line_length, max_segments = 100, linestyle = (1,1)):
    if target_line_length <= 0:
        return line_array
    dotted_lines_array = []
    line_index = 0
    while line_index < len(line_array):
        v_from = np.asarray(line_array[line_index])
        v_to = np.asarray(line_array[line_index + 1])

        line_len = np.linalg.norm(v_from - v_to)
        segments = min(int(line_len / target_line_length), max_segments)
        
        total_segments = 2 * segments + 1
        segment_start_index = 0
        style_len = len(linestyle)
        assert (style_len % 2) == 0
        style_index = 0
        while segment_start_index < total_segments:
            t_from = segment_start_index / total_segments
            t_to = min((segment_start_index + linestyle[style_index]) / total_segments, 1.0)
            dotted_lines_array.append((1.0 - t_from) * v_from + t_from * v_to)
            dotted_lines_array.append((1.0 - t_to) * v_from + t_to * v_to)
            segment_start_index += linestyle[style_index] + linestyle[style_index + 1]
            style_index = (style_index + 2) % style_len
        line_index += 2
    return dotted_lines_array

def linear_to_srgb(linear_color):
    linear_color = np.array(linear_color)
    srgb_color = np.where(linear_color <= 0.0031308, 12.92 * linear_color, 1.055 * np.power(linear_color, 1/2.4) - 0.055)
    return srgb_color

class ComponentQuality(Enum):
    PERFECT_REGION = 0
    BAD_GLUE_FLAPS_REGION = 1
    OVERLAPPING_REGION = 2
    NOT_FOLDABLE_REGION = 3
    NOT_SELECTED_REGION = 4

class GlueFlapQuality(Enum):
    GLUE_FLAP_NO_OVERLAPS = 0
    GLUE_FLAP_WITH_OVERLAPS = 1
    GLUE_FLAP_TO_LARGE = 2

def mesh_edge_id_list_to_coordinate_list(bmesh : BMesh, edge_ids, normal_offset, world_matrix):
    cut_coordinates = []
    bmesh.edges.ensure_lookup_table()
    for edge_id in edge_ids:
        if edge_id >= len(bmesh.edges):
            return [] # something went wrong!
        vert1, vert2 = bmesh.edges[edge_id].verts
        cut_coordinates.append(world_matrix @ (vert1.co + normal_offset * vert1.normal))
        cut_coordinates.append(world_matrix @ (vert2.co + normal_offset * vert1.normal))
    return cut_coordinates

def __get_corner_points_all_interior_or_cut(corner_pos, prev_pos, next_pos, normal, dist_to_edges):
    ctp : mathutils.Vector = (prev_pos - corner_pos)
    ctn : mathutils.Vector = (next_pos - corner_pos)
    corner_angle = math.acos(np.clip(ctp.normalized().dot(ctn.normalized()), -1, 1))
    shell_factor = 1 / math.sin(corner_angle / 2)
    corner_is_convex = face_corner_convex_3d(prev_pos, corner_pos, next_pos, normal)
    rotation_to_angle_bisect_vec : mathutils.Matrix = mathutils.Matrix.Rotation((corner_angle if corner_is_convex else 2 * np.pi - corner_angle) / 2, 3, normal)
    angle_bisect_vec = rotation_to_angle_bisect_vec @ ctn.normalized()
    return [mathutils.Vector(corner_pos + dist_to_edges * shell_factor * angle_bisect_vec)]

def __get_corner_points_only_vertex_on_cut(corner_pos, prev_pos, next_pos, normal : mathutils.Vector, cut_dist, offset_dist):
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

def __get_corner_points_one_edge_cut(corner_pos, prev_pos, next_pos, normal : mathutils.Vector, cut_dist, offset_dist, prev_is_cut):
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

def compute_polygon_outline_for_face_drawing(bmesh : BMesh, face_index, large_dist, small_dist, edge_constraints):
    face = bmesh.faces[face_index]
    face_normal = face.normal
    verts = list(face.verts)
    cool_vertices = []
    for v_id in range(len(verts)):
        curr_v = verts[v_id]
        prev_v = verts[(v_id + len(verts) - 1) % len(verts)]
        next_v = verts[(v_id + 1) % len(verts)]

        # v_on_cutting_edge_or_boundary = curr_v.is_boundary or np.any([self.mesh_edge_is_cut(e) for e in curr_v.link_edges])
        v_on_cutting_edge = np.any([mesh_edge_is_cut(e.index, edge_constraints) for e in curr_v.link_edges])

        e_to_curr = find_bmesh_edge_of_halfedge(bmesh, (prev_v.index, curr_v.index))
        e_from_curr = find_bmesh_edge_of_halfedge(bmesh, (curr_v.index, next_v.index))

        e_prev_curr_is_cutting = mesh_edge_is_cut(e_to_curr.index, edge_constraints)
        e_curr_to_next_is_cutting = mesh_edge_is_cut(e_from_curr.index, edge_constraints)

        if not v_on_cutting_edge and not e_prev_curr_is_cutting and not e_curr_to_next_is_cutting:
            cool_vertices += __get_corner_points_all_interior_or_cut(curr_v.co, prev_v.co, next_v.co, face_normal, small_dist)
        elif v_on_cutting_edge and e_prev_curr_is_cutting and e_curr_to_next_is_cutting:
            cool_vertices += __get_corner_points_all_interior_or_cut(curr_v.co, prev_v.co, next_v.co, face_normal, large_dist)
        elif v_on_cutting_edge and not e_prev_curr_is_cutting and not e_curr_to_next_is_cutting:
            cool_vertices += __get_corner_points_only_vertex_on_cut(curr_v.co, prev_v.co, next_v.co, face_normal, large_dist, small_dist)
        elif e_prev_curr_is_cutting:
            cool_vertices += __get_corner_points_one_edge_cut(curr_v.co, prev_v.co, next_v.co, face_normal, large_dist, small_dist, True)
        else:
            cool_vertices += __get_corner_points_one_edge_cut(curr_v.co, prev_v.co, next_v.co, face_normal, large_dist, small_dist, False)

    return cool_vertices

def compute_and_update_connected_component_triangle_lists_for_drawing(mesh : Mesh, bmesh : BMesh, face_offset, edge_constraints, world_matrix,
                                                                      connected_components = None):
    if connected_components is None:
        connected_components = io.read_all_component_render_data(mesh)

    outdated_render_flags = io.read_outdated_render_data(mesh)
    verts_per_component, triangles_per_component = io.read_all_component_render_data(mesh)
    if verts_per_component is None:
        verts_per_component, triangles_per_component = {}, {}
        io.write_all_component_render_data(mesh, verts_per_component, triangles_per_component) # to make sure we can later update 

    bmesh.faces.ensure_lookup_table()

    render_ready_triangles_per_component = {}
    render_ready_vertex_positions = []

    min_edge_len = min(e.calc_length() for e in bmesh.edges) # i hope this does not take too long
    small_dist = 0.05 * min_edge_len

    for component_id, faces_in_component in connected_components.items():
        # only compute render data if it is missing or marked to be recomputed
        if component_id in outdated_render_flags or component_id not in verts_per_component.keys():
            component_cut_dist = 0.25 * min([min([e.calc_length() for e in bmesh.faces[f_index].edges]) for f_index in faces_in_component])
            component_triangles = []
            component_vertex_positions = []
            for face_index in faces_in_component:
                polygon_outline = compute_polygon_outline_for_face_drawing(bmesh, face_index, component_cut_dist, small_dist, edge_constraints)
                curr_offset = len(component_vertex_positions)
                _, curr_triangle_ids = triangulate_3d_polygon(polygon_outline, bmesh.faces[face_index].normal, list(range(curr_offset, curr_offset + len(polygon_outline))))
                component_vertex_positions += [(v, face_index) for v in polygon_outline]
                component_triangles += curr_triangle_ids
            verts_per_component[component_id] = component_vertex_positions
            triangles_per_component[component_id] = component_triangles
            outdated_render_flags.discard(component_id)
            if io.component_render_data_exists(mesh):
                io.write_render_data_of_one_component(mesh, component_id, component_vertex_positions, component_triangles)
        vertex_id_offset = len(render_ready_vertex_positions)
        render_ready_vertex_positions += [world_matrix @ (v + face_offset * bmesh.faces[f_index].normal) for (v, f_index) in verts_per_component[component_id]]
        render_ready_triangles_per_component[component_id] = [(vertex_id_offset + tri[0], vertex_id_offset + tri[1], vertex_id_offset + tri[2]) for tri in triangles_per_component[component_id]]
    io.write_outdated_render_data(mesh, outdated_render_flags)
    return render_ready_vertex_positions, render_ready_triangles_per_component

def get_triangle_list_per_cluster_quality(mesh : Mesh, bmesh : BMesh, face_offset, edge_constraints, world_matrix,
                                          connected_components = None, selected_component = None):
    if connected_components is None:
        connected_components = io.read_connected_component_sets(mesh)
        assert connected_components is not None
    all_v_positions, tris_per_component = compute_and_update_connected_component_triangle_lists_for_drawing(mesh, bmesh, face_offset, edge_constraints, world_matrix, connected_components)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    components_with_overlaps = io.read_components_with_overlaps(mesh)
    glue_flap_collisions = io.read_glue_flap_collisions_dict(mesh)

    quality_dict = {
        ComponentQuality.PERFECT_REGION : [],
        ComponentQuality.BAD_GLUE_FLAPS_REGION : [],
        ComponentQuality.OVERLAPPING_REGION : [],
        ComponentQuality.NOT_FOLDABLE_REGION : [],
        ComponentQuality.NOT_SELECTED_REGION : []
        }
    for component_id, triangles in tris_per_component.items():
        if selected_component is not None and component_id != selected_component:
            #quality_dict[ComponentQuality.NOT_SELECTED_REGION] += triangles
            continue
        if component_id in cyclic_components:
            quality_dict[ComponentQuality.NOT_FOLDABLE_REGION] += triangles
            continue
        if component_id in components_with_overlaps:
            quality_dict[ComponentQuality.OVERLAPPING_REGION] += triangles
            continue
        if component_has_overlapping_glue_flaps(mesh, component_id, glue_flap_collisions):
            quality_dict[ComponentQuality.BAD_GLUE_FLAPS_REGION] += triangles
            continue
        quality_dict[ComponentQuality.PERFECT_REGION] += triangles
    return all_v_positions, quality_dict

def get_lines_array_per_glue_flap_quality(mesh : Mesh, bmesh : BMesh, face_offset, world_matrix, flap_angle, flap_height,
                                          connected_components = None,
                                          face_to_component_dict = None):
    glue_flaps_dict = io.read_glueflap_halfedge_dict(mesh)
    glue_flap_collision_dict = io.read_glue_flap_collisions_dict(mesh)
    halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bmesh)
    if connected_components is None:
        connected_components, face_to_component_dict = io.read_connected_components(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)

    bmesh.edges.ensure_lookup_table()
    quality_dict = {
        GlueFlapQuality.GLUE_FLAP_NO_OVERLAPS : [],
        GlueFlapQuality.GLUE_FLAP_WITH_OVERLAPS : [],
        GlueFlapQuality.GLUE_FLAP_TO_LARGE : []
        }
    for edge_index, halfedge in glue_flaps_dict.items():
        opp_halfedge = (halfedge[1], halfedge[0])
        component_id_of_halfedge = face_to_component_dict[halfedge_to_face_dict[halfedge].index]
        face_of_opp_halfedge = halfedge_to_face_dict[opp_halfedge]
        component_id_of_opp_halfedge = face_to_component_dict[face_of_opp_halfedge.index]

        if component_id_of_opp_halfedge in cyclic_components:
            continue # there is no valid unfolding of the opposite component
        flap_triangles_3d = compute_3d_glue_flap_triangles_inside_face(mesh, face_of_opp_halfedge.index, utils.find_bmesh_edge_of_halfedge(bmesh, halfedge), flap_angle, flap_height)
        flap_line_array = []
        if len(flap_triangles_3d) == 1:
            tri = flap_triangles_3d[0]
            flap_line_array = [tri[0], tri[1], tri[1], tri[2]]
        else:
            tri_0 = flap_triangles_3d[0]
            tri_1 = flap_triangles_3d[1]
            flap_line_array = [tri_0[0], tri_0[1], tri_0[1], tri_0[2], tri_1[0], tri_1[1]]
        # apply offset
        flap_line_array = [world_matrix @ (mathutils.Vector(v) + 1.5 * face_offset * face_of_opp_halfedge.normal) for v in flap_line_array]
        if flap_is_overlapping(mesh, component_id_of_halfedge, edge_index, glue_flap_collision_dict):
            quality_dict[GlueFlapQuality.GLUE_FLAP_WITH_OVERLAPS] += flap_line_array
            continue
        # TODO Flaps that are to large... maybe not needed ... maybe add this in the future
        quality_dict[GlueFlapQuality.GLUE_FLAP_NO_OVERLAPS] += flap_line_array
    return quality_dict

def compute_backgound_paper_render_data(num_pages, paper_size, selected_page = None, margin_between_pages = 1.0, pages_per_row = 2.0):
    selected_page_lines = []
    page_verts = []
    other_page_lines = []
    for page_index in range(num_pages):
        row_index = page_index % pages_per_row
        col_index = page_index // pages_per_row
        page_anchor = np.array([row_index * (paper_size[0] + margin_between_pages), -col_index * (paper_size[1] + margin_between_pages)])
        ll = page_anchor 
        lr = page_anchor + np.array([paper_size[0], 0.0])
        ur = page_anchor + np.array([paper_size[0], paper_size[1]])
        ul = page_anchor + np.array([0.0, paper_size[1]])
        current_page_line_coords = [ll, lr, lr, ur, ur, ul, ul, ll]
        if selected_page is not None and selected_page == page_index:
            selected_page_lines += current_page_line_coords
        else:
            other_page_lines += current_page_line_coords
        page_verts += [ll, lr, ur, ll, ur, ul]
    return page_verts, selected_page_lines, other_page_lines

class LayoutRenderData(Enum):
    FULL_LINES = 0
    CONVEX_LINES = 1
    CONCAVE_LINES = 2
    BG_VERTS = 3
    BG_COLORS = 4
    STEP_NUMBER = 5

def compute_page_layout_render_data_of_component(component : ComponentPrintData, paper_size, fold_angle_th, page_index, pages_per_row = 2.0, margin_between_pages = 1.0):
    row_index = page_index % pages_per_row
    col_index = page_index // pages_per_row
    page_anchor = np.array([row_index * (paper_size[0] + margin_between_pages), -col_index * (paper_size[1] + margin_between_pages)])
    full_transform = AffineTransform2D(affine_part=page_anchor) @ component.page_transform

    render_data = {}
    full_lines = []
    convex_lines = []
    concave_lines = []
    bg_verts = []
    bg_colors = []

    for cut_edge_data in component.cut_edges + component.glue_flap_edges:
        coords = [full_transform * coord for coord in cut_edge_data.coords]
        full_lines += coords
    fold_edge_data : FoldEdgeData
    for fold_edge_data in component.fold_edges:
        if fold_edge_data.fold_angle <= fold_angle_th:
            continue
        coords = [full_transform * coord for coord in fold_edge_data.coords]
        if fold_edge_data.is_convex:
            convex_lines += coords
        else:
            concave_lines += coords
    fold_edge_at_flap : FoldEdgeAtGlueFlapData
    for fold_edge_at_flap in component.fold_edges_at_flaps:
        coords = [full_transform * coord for coord in fold_edge_at_flap.coords]
        if fold_edge_at_flap.fold_angle <= fold_angle_th:
            full_lines += coords
            continue
        if fold_edge_at_flap.is_convex:
            convex_lines += coords
        else:
            concave_lines += coords
    view_coords = full_transform * component.build_step_number_position
    render_data[LayoutRenderData.STEP_NUMBER] = (component.build_step_number, view_coords)
    tri_data : ColoredTriangleData
    for tri_data in component.colored_triangles:
        bg_verts += [full_transform * coord for coord in tri_data.coords]
        bg_colors += [linear_to_srgb(tri_data.color)] * 3

    render_data[LayoutRenderData.FULL_LINES] = full_lines
    render_data[LayoutRenderData.CONVEX_LINES] = make_dotted_lines(convex_lines, 0.2)
    render_data[LayoutRenderData.CONCAVE_LINES] = make_dotted_lines(concave_lines, 0.2, linestyle=[1,1,4,1])
    render_data[LayoutRenderData.BG_VERTS] = bg_verts
    render_data[LayoutRenderData.BG_COLORS] = bg_colors

    return render_data

def compute_page_layout_render_data_of_all_components(components_per_page, paper_size, margin_between_pages = 1.0, pages_per_row = 2, fold_angle_th = 0.0,):
    render_data_per_component = {}
    for page_index, components_on_page in enumerate(components_per_page):
        current_component : ComponentPrintData
        for current_component in components_on_page:
            render_data_per_component[current_component.og_component_id] = compute_page_layout_render_data_of_component(current_component, paper_size, fold_angle_th,
                                                                                                                        page_index, pages_per_row, margin_between_pages)
    return render_data_per_component

def combine_layout_render_data_of_all_components(render_data_per_component, selected_component = None, color_components = True, show_step_numbers = True):
    component_bg_verts = []
    component_bg_colors = []
    selected_component_lines = []
    other_component_lines = []
    build_step_numbers = []

    for component_id, render_data in render_data_per_component.items():
        if selected_component is not None and component_id == selected_component:
            selected_component_lines += render_data[LayoutRenderData.FULL_LINES]
        else:
            other_component_lines += render_data[LayoutRenderData.FULL_LINES]
        other_component_lines += render_data[LayoutRenderData.CONCAVE_LINES]
        other_component_lines += render_data[LayoutRenderData.CONVEX_LINES]
        if color_components:
            component_bg_verts += render_data[LayoutRenderData.BG_VERTS]
            component_bg_colors += render_data[LayoutRenderData.BG_COLORS]
        if show_step_numbers:
            build_step_numbers.append(render_data[LayoutRenderData.STEP_NUMBER])

    return component_bg_verts, component_bg_colors, other_component_lines, selected_component_lines, build_step_numbers
