"""
Functions in this file provide all data necessary for rendering feedback.
"""

import bpy
from bpy.types import Mesh
from bmesh.types import BMesh, BMFace
import numpy as np
import mathutils
import math
import gpu
from enum import Enum
import time
from dataclasses import dataclass
import typing
import numpy.typing

from . import io
from . import utils
from .colors import *
from .geometry import AffineTransform2D, triangulate_3d_polygon, face_corner_convex_3d, solve_for_weird_intersection_point
from .utils import mesh_edge_is_cut, find_bmesh_edge_of_halfedge
from .glueflaps import component_has_overlapping_glue_flaps, flap_is_overlapping, compute_3d_glue_flap_coords_in_glued_face
from .printprepper import ComponentPrintData, FoldEdgeData, FoldEdgeAtGlueFlapData, ColoredTriangleData

def make_arc_length_array(vertex_position_array):
    res = []
    for i in range(0,len(vertex_position_array),2): 
        v_from = np.asarray(vertex_position_array[i])
        v_to = np.asarray(vertex_position_array[i + 1])
        res.append(0)
        res.append(np.linalg.norm(v_to - v_from))
    return res

def linear_to_srgb(linear_color):
    if linear_color is None:
        return None
    linear_rgb = np.array(linear_color[:3], dtype=np.float64)
    srgb_color = np.where(linear_rgb <= 0.0031308, 12.92 * linear_rgb, 1.055 * np.power(linear_rgb, 1/2.4) - 0.055)
    return np.array((*srgb_color, linear_color[3]), dtype=np.float64)

def srgb_to_linear(srgba_color):
    if srgba_color is None:
        return None
    srgba_rgb = np.array(srgba_color[:3], dtype=np.float64)
    linear_rgb = np.where(srgba_rgb <= 0.040449936, srgba_rgb / 12.92, np.power((srgba_rgb + 0.055) / 1.055, 2.4))
    return np.array((*linear_rgb, srgba_color[3]))

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
                                          connected_components = None, selected_component = None, components_in_selected_section = set()):
    if connected_components is None:
        connected_components = io.read_connected_component_sets(mesh)
        assert connected_components is not None
    all_v_positions, tris_per_component = compute_and_update_connected_component_triangle_lists_for_drawing(mesh, bmesh, face_offset, edge_constraints, world_matrix, connected_components)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    components_with_overlaps = io.read_components_with_overlaps(mesh)
    glue_flap_collisions = io.read_glue_flap_collisions_dict(mesh)

    quality_dict = { "default": {
                        ComponentQuality.PERFECT_REGION : [],
                        ComponentQuality.BAD_GLUE_FLAPS_REGION : [],
                        ComponentQuality.OVERLAPPING_REGION : [],
                        ComponentQuality.NOT_FOLDABLE_REGION : [],
                        ComponentQuality.NOT_SELECTED_REGION : []},
                    "selected": {
                        ComponentQuality.PERFECT_REGION : [],
                        ComponentQuality.BAD_GLUE_FLAPS_REGION : [],
                        ComponentQuality.OVERLAPPING_REGION : [],
                        ComponentQuality.NOT_FOLDABLE_REGION : [],
                        ComponentQuality.NOT_SELECTED_REGION : []}}

    for component_id, triangles in tris_per_component.items():
        if selected_component is not None and component_id != selected_component:
            #quality_dict[ComponentQuality.NOT_SELECTED_REGION] += triangles
            continue
        if component_id in components_in_selected_section:
            if component_id in cyclic_components:
                quality_dict["selected"][ComponentQuality.NOT_FOLDABLE_REGION] += triangles
                continue
            if component_id in components_with_overlaps:
                quality_dict["selected"][ComponentQuality.OVERLAPPING_REGION] += triangles
                continue
            if component_has_overlapping_glue_flaps(mesh, component_id, glue_flap_collisions):
                quality_dict["selected"][ComponentQuality.BAD_GLUE_FLAPS_REGION] += triangles
                continue
            quality_dict["selected"][ComponentQuality.PERFECT_REGION] += triangles
        else:
            if component_id in cyclic_components:
                quality_dict["default"][ComponentQuality.NOT_FOLDABLE_REGION] += triangles
                continue
            if component_id in components_with_overlaps:
                quality_dict["default"][ComponentQuality.OVERLAPPING_REGION] += triangles
                continue
            if component_has_overlapping_glue_flaps(mesh, component_id, glue_flap_collisions):
                quality_dict["default"][ComponentQuality.BAD_GLUE_FLAPS_REGION] += triangles
                continue
            quality_dict["default"][ComponentQuality.PERFECT_REGION] += triangles
    return all_v_positions, quality_dict

def get_lines_array_per_glue_flap_quality(mesh : Mesh, bmesh : BMesh, face_offset, world_matrix, flap_angle, flap_height,
                                          connected_components = None,
                                          face_to_component_dict = None):
    glue_flaps_dict = io.read_glueflap_halfedge_dict(mesh)
    glueflap_geometry = io.read_glue_flap_geometry_per_edge_per_component(mesh)
    glue_flap_collision_dict = io.read_glue_flap_collisions_dict(mesh)
    unfolding_affine_transforms = io.read_affine_transforms_to_roots(mesh)
    local_coord_system_per_face = io.read_local_coordinate_systems_per_face(mesh)
    if connected_components is None:
        connected_components, face_to_component_dict = io.read_connected_components(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    halfedge_to_face_dict = utils.construct_halfedge_to_face_dict(bmesh)

    bmesh.edges.ensure_lookup_table()
    quality_dict = {
        GlueFlapQuality.GLUE_FLAP_NO_OVERLAPS : [],
        GlueFlapQuality.GLUE_FLAP_WITH_OVERLAPS : [],
        GlueFlapQuality.GLUE_FLAP_TO_LARGE : []
        }
    for edge_index, halfedge in glue_flaps_dict.items():
        opp_halfedge = (halfedge[1], halfedge[0])
        face_of_halfedge : BMFace = halfedge_to_face_dict[halfedge]
        component_id_of_halfedge = face_to_component_dict[face_of_halfedge.index]
        face_of_opp_halfedge : BMFace = halfedge_to_face_dict[opp_halfedge]
        component_id_of_opp_halfedge = face_to_component_dict[face_of_opp_halfedge.index]

        if component_id_of_opp_halfedge in cyclic_components:
            continue # there is no valid unfolding of the opposite component
        flap_triangles_3d = compute_3d_glue_flap_coords_in_glued_face(bmesh, component_id_of_halfedge, face_of_halfedge.index, bmesh.edges[edge_index], halfedge, component_id_of_opp_halfedge, face_of_opp_halfedge.index,
                                                                      glueflap_geometry, unfolding_affine_transforms, local_coord_system_per_face)
        flap_line_array = []
        if len(flap_triangles_3d) == 1:
            tri = flap_triangles_3d[0]
            flap_line_array = [tri[0], tri[1], tri[1], tri[2]]
        else:
            tri_0 = flap_triangles_3d[0]
            tri_1 = flap_triangles_3d[1]
            flap_line_array = [tri_0[0], tri_0[1], tri_0[1], tri_0[2], tri_1[1], tri_1[2]]
        # apply offset
        flap_line_array = [world_matrix @ (mathutils.Vector(v) + 1.5 * face_offset * face_of_opp_halfedge.normal) for v in flap_line_array]
        if flap_is_overlapping(mesh, component_id_of_halfedge, edge_index, glue_flap_collision_dict):
            quality_dict[GlueFlapQuality.GLUE_FLAP_WITH_OVERLAPS] += flap_line_array
            continue
        # TODO Flaps that are to large... maybe not needed ... maybe add this in the future
        quality_dict[GlueFlapQuality.GLUE_FLAP_NO_OVERLAPS] += flap_line_array
    return quality_dict

#################################
#          Page Preview         #
#################################

def compute_backgound_paper_render_data(num_pages, paper_size, selected_page = None, margin_between_pages = 1.0, pages_per_row = 2.0):
    selected_page_lines = []
    page_verts = []
    other_page_lines = []
    for page_index in range(num_pages):
        row_index = page_index % pages_per_row
        col_index = page_index // pages_per_row
        page_anchor = np.array([row_index * (paper_size[0] + margin_between_pages), -col_index * (paper_size[1] + margin_between_pages)], dtype=np.float64)
        ll = page_anchor 
        lr = page_anchor + np.array([paper_size[0], 0.0], dtype=np.float64)
        ur = page_anchor + np.array([paper_size[0], paper_size[1]], dtype=np.float64)
        ul = page_anchor + np.array([0.0, paper_size[1]], dtype=np.float64)
        current_page_line_coords = [ll, lr, lr, ur, ur, ul, ul, ll]
        if selected_page is not None and selected_page == page_index:
            selected_page_lines += current_page_line_coords
        else:
            other_page_lines += current_page_line_coords
        page_verts += [ll, lr, ur, ll, ur, ul]
    return page_verts, selected_page_lines, other_page_lines

class LayoutRenderData(Enum):
    OUTLINE_LINES = 0
    CONCAVE_FOLDS = 1
    CONVEX_FOLDS = 2
    BG_VERTS = 3
    BG_COLORS = 4
    STEP_NUMBER = 5
    BG_UVS = 6
    BG_TEXTURES = 7

def compute_page_layout_render_data_of_component(component : ComponentPrintData, paper_size, fold_angle_th, page_index, pages_per_row = 2.0, margin_between_pages = 1.0):
    row_index = page_index % pages_per_row
    col_index = page_index // pages_per_row
    page_anchor = np.array([row_index * (paper_size[0] + margin_between_pages), -col_index * (paper_size[1] + margin_between_pages)], dtype=np.float64)
    full_transform = AffineTransform2D(affine_part=page_anchor) @ component.page_transform

    render_data = {}
    full_lines = []
    convex_lines = []
    concave_lines = []
    bg_verts = []
    bg_colors = []
    bg_uvs = []
    bg_texture_paths = []

    cut_edges_coords = [cut_edge.coords for cut_edge in component.cut_edges]
    for glue_flap_data in component.glue_flaps:
        cut_edges_coords += glue_flap_data.edge_coords
    for one_edge_coords in cut_edges_coords:
        coords = [full_transform * coord for coord in one_edge_coords]
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
            continue
        if fold_edge_at_flap.is_convex:
            convex_lines += coords
        else:
            concave_lines += coords

    section_plus_step_number = ""
    if component.build_section_name is not None:
        section_plus_step_number += component.build_section_name + " "
    section_plus_step_number += str(component.build_step_number)
    view_coords = full_transform * component.build_step_number_position
    render_data[LayoutRenderData.STEP_NUMBER] = (section_plus_step_number, view_coords)
    tri_data : ColoredTriangleData
    for tri_data in component.colored_triangles:
        bg_verts += [full_transform * coord for coord in tri_data.coords]
        bg_colors.append(linear_to_srgb(tri_data.color if tri_data.color is not None else (1,1,1,1)))
        bg_uvs += tri_data.uvs
        bg_texture_paths.append(str(tri_data.absolute_texture_path))

    render_data[LayoutRenderData.OUTLINE_LINES] = full_lines
    render_data[LayoutRenderData.CONVEX_FOLDS] = convex_lines
    render_data[LayoutRenderData.CONCAVE_FOLDS] = concave_lines
    render_data[LayoutRenderData.BG_VERTS] = bg_verts
    render_data[LayoutRenderData.BG_COLORS] = bg_colors
    render_data[LayoutRenderData.BG_UVS] = bg_uvs
    render_data[LayoutRenderData.BG_TEXTURES] = bg_texture_paths

    return render_data

def compute_page_layout_render_data_of_all_components(components_per_page, paper_size, margin_between_pages = 1.0, pages_per_row = 2, fold_angle_th = 0.0,):
    render_data_per_component = {}
    for page_index, components_on_page in enumerate(components_per_page):
        current_component : ComponentPrintData
        for current_component in components_on_page:
            render_data_per_component[current_component.og_component_id] = compute_page_layout_render_data_of_component(current_component, paper_size, fold_angle_th,
                                                                                                                        page_index, pages_per_row, margin_between_pages)
    return render_data_per_component

class PageLayoutLinestyle(Enum):
    FULL = 0
    DASHED = 1
    DASH_DOT = 2

@dataclass
class PageLayoutPreviewIslandTriangleData:
    texture : gpu.types.GPUTexture | None
    coords : list[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]
    uvs : list[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]
    colors : list[tuple[float, float, float , float]]

@dataclass
class PageLayoutPreviewIslandEdgeData:
    linestyle : PageLayoutLinestyle
    thickness : float
    colors : list[tuple[float, float, float , float]]
    coords : list[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]

@dataclass
class PageLayoutPreviewIslandNumbersData:
    nums_with_positions : list[tuple[str, typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]]    
    transparency : list[float]

@dataclass
class PageLayoutPreviewIslandRenderData:
    triangles : list[PageLayoutPreviewIslandTriangleData]
    edges : list[PageLayoutPreviewIslandEdgeData]
    page_numbers : list[PageLayoutPreviewIslandNumbersData]

def combine_layout_render_data_of_all_components(render_data_per_component, 
                                                 selected_component = None, 
                                                 components_in_selected_section : set = set(), 
                                                 unselected_component_transparency = 1.0, 
                                                 show_step_numbers = True):
    triangle_batches : dict[str, PageLayoutPreviewIslandTriangleData] = {}
    convex_fold_lines = PageLayoutPreviewIslandEdgeData(PageLayoutLinestyle.DASHED, 1.5, [], [])
    concave_fold_lines = PageLayoutPreviewIslandEdgeData(PageLayoutLinestyle.DASH_DOT, 1.5, [], [])
    outlines = PageLayoutPreviewIslandEdgeData(PageLayoutLinestyle.FULL, 1.5, [], [])
    thick_lines = PageLayoutPreviewIslandEdgeData(PageLayoutLinestyle.FULL, 2.0, [], [])
    step_numbers = PageLayoutPreviewIslandNumbersData([], [])

    for component_id, render_data in render_data_per_component.items():
        component_transparency = 1.0 if component_id in components_in_selected_section else unselected_component_transparency
        # triangles
        for triangle_id, texture_path in enumerate(render_data[LayoutRenderData.BG_TEXTURES]):
            if texture_path not in triangle_batches:
                texture = None
                if texture_path != "None":
                    texture = gpu.texture.from_image(bpy.data.images.load(filepath=texture_path, check_existing=True))
                triangle_batches[texture_path] = PageLayoutPreviewIslandTriangleData(texture, [], [], [])
            current_batch = triangle_batches[texture_path]
            current_color = render_data[LayoutRenderData.BG_COLORS][triangle_id]
            current_batch.colors += [(*current_color[:3], component_transparency)] * 3
            i = triangle_id * 3
            current_batch.coords += render_data[LayoutRenderData.BG_VERTS][i:i + 3]
            current_batch.uvs += render_data[LayoutRenderData.BG_UVS][i:i + 3]
        
        # edges
        def fill_edge_data(container : PageLayoutPreviewIslandEdgeData, position_array, color):
            container.coords += position_array
            container.colors += [(*color[:3], component_transparency)] * len(position_array)

        fill_edge_data(convex_fold_lines, render_data[LayoutRenderData.CONVEX_FOLDS], BLACK)
        fill_edge_data(concave_fold_lines, render_data[LayoutRenderData.CONCAVE_FOLDS], BLACK)
        if selected_component is not None and component_id == selected_component:
            fill_edge_data(thick_lines, render_data[LayoutRenderData.OUTLINE_LINES], ORANGE)
        else:
            fill_edge_data(outlines, render_data[LayoutRenderData.OUTLINE_LINES], BLACK)
        
        # step numbers
        if not show_step_numbers:
            continue

        step_numbers.nums_with_positions.append(render_data[LayoutRenderData.STEP_NUMBER])
        step_numbers.transparency.append(component_transparency)

    return triangle_batches, [convex_fold_lines, concave_fold_lines, outlines, thick_lines], step_numbers
