"""
Functions in this file provide all data necessary for rendering feedback.
"""

import bpy
from bpy.types import Mesh
from bmesh.types import BMesh
import numpy as np
import mathutils
import math
from enum import Enum

from . import io
from . import utils
from .geometry import triangulate_3d_polygon, signed_point_dist_to_line, face_corner_convex_3d, solve_for_weird_intersection_point
from .utils import mesh_edge_is_cut, find_bmesh_edge_of_halfedge
from .glueflaps import component_has_overlapping_glue_flaps, flap_is_overlapping

class ComponentQuality(Enum):
    PERFECT_REGION = 0
    BAD_GLUE_FLAPS_REGION = 1
    OVERLAPPING_REGION = 2
    NOT_FOLDABLE_REGION = 3

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

def compute_polygon_outline_for_face_drawing(bmesh : BMesh, face_index, large_dist, small_dist, edge_constraints, use_auto_cuts):
    face = bmesh.faces[face_index]
    face_normal = face.normal
    verts = list(face.verts)
    cool_vertices = []
    for v_id in range(len(verts)):
        curr_v = verts[v_id]
        prev_v = verts[(v_id + len(verts) - 1) % len(verts)]
        next_v = verts[(v_id + 1) % len(verts)]

        # v_on_cutting_edge_or_boundary = curr_v.is_boundary or np.any([self.mesh_edge_is_cut(e) for e in curr_v.link_edges])
        v_on_cutting_edge = np.any([mesh_edge_is_cut(e.index, edge_constraints, use_auto_cuts) for e in curr_v.link_edges])

        e_to_curr = find_bmesh_edge_of_halfedge(bmesh, (prev_v.index, curr_v.index))
        e_from_curr = find_bmesh_edge_of_halfedge(bmesh, (curr_v.index, next_v.index))

        e_prev_curr_is_cutting = mesh_edge_is_cut(e_to_curr.index, edge_constraints, use_auto_cuts)
        e_curr_to_next_is_cutting = mesh_edge_is_cut(e_from_curr.index, edge_constraints, use_auto_cuts)

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

def compute_and_update_connected_component_triangle_lists_for_drawing(mesh : Mesh, bmesh : BMesh, face_offset, edge_constraints, use_auto_cuts, world_matrix,
                                                                      connected_components = None):
    if connected_components is None:
        connected_components = io.read_all_component_render_data(mesh)

    outdated_render_flags = io.read_outdated_render_data_flags(mesh)
    verts_per_component, triangles_per_component = io.read_all_component_render_data(mesh)
    if verts_per_component is None:
        verts_per_component, triangles_per_component = {}, {}
        io.write_all_component_render_data(mesh, verts_per_component, triangles_per_component) # to make sure we can later update 

    bmesh.faces.ensure_lookup_table()

    triangles_per_component = {}
    all_vertex_positions = []

    min_edge_len = min(e.calc_length() for e in mesh.edges) # i hope this does not take too long
    small_dist = 0.05 * min_edge_len

    for component_id, faces_in_component in connected_components.items():
        # only compute render data if it is missing or marked to be recomputed
        if outdated_render_flags[component_id] or component_id not in verts_per_component.keys():
            component_cut_dist = 0.25 * min([min([e.calc_length() for e in bmesh.faces[f_index].edges]) for f_index in faces_in_component])
            component_triangles = []
            component_vertex_positions = []
            for face_index in faces_in_component:
                polygon_outline = compute_polygon_outline_for_face_drawing(bmesh, face_index, component_cut_dist, small_dist, edge_constraints, use_auto_cuts)
                curr_offset = len(component_vertex_positions)
                _, curr_triangle_ids = triangulate_3d_polygon(polygon_outline, bmesh.faces[face_index].normal, list(range(curr_offset, curr_offset + len(polygon_outline))))
                component_vertex_positions += [(v, face_index) for v in polygon_outline]
                component_triangles += curr_triangle_ids
            verts_per_component[component_id] = component_vertex_positions
            triangles_per_component[component_id] = component_triangles
            outdated_render_flags[component_id] = False
            if io.component_render_data_exists(mesh):
                io.write_render_data_of_one_component(mesh, component_id, component_vertex_positions, component_triangles)

        vertex_id_offset = len(all_vertex_positions)
        all_vertex_positions += [world_matrix @ (v + face_offset * bmesh.faces[f_index].normal) for (v, f_index) in verts_per_component[component_id]]
        triangles_per_component[component_id] = [(vertex_id_offset + tri[0], vertex_id_offset + tri[1], vertex_id_offset + tri[2]) for tri in triangles_per_component[component_id]]

    io.write_outdated_render_data_flags(mesh, outdated_render_flags)
    return all_vertex_positions, triangles_per_component

def get_triangle_list_per_cluster_quality(mesh : Mesh, bmesh : BMesh, face_offset, edge_constraints, use_auto_cuts, world_matrix,
                                          connected_components = None):
    all_v_positions, tris_per_component = compute_and_update_connected_component_triangle_lists_for_drawing(mesh, bmesh, face_offset, edge_constraints, use_auto_cuts, world_matrix, connected_components)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    components_with_overlaps = io.read_components_with_overlaps(mesh)
    glue_flap_collisions = io.read_glue_flap_collisions_dict(mesh)

    quality_dict = {
        ComponentQuality.PERFECT_REGION : [],
        ComponentQuality.BAD_GLUE_FLAPS_REGION : [],
        ComponentQuality.OVERLAPPING_REGION : [],
        ComponentQuality.NOT_FOLDABLE_REGION : []
        }
    for component_id, triangles in tris_per_component.items():
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

def get_lines_array_per_glue_flap_quality(mesh : Mesh, bmesh : BMesh, face_offset, edge_constraints, use_auto_cuts, world_matrix,
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
        component_id_of_halfedge = face_to_component_dict[halfedge_to_face_dict[halfedge]]
        face_of_opp_halfedge = halfedge_to_face_dict[opp_halfedge]
        component_id_of_opp_halfedge = face_to_component_dict[face_of_opp_halfedge.index]

        if component_id_of_opp_halfedge in cyclic_components:
            continue # there is no valid unfolding of the opposite component
        flap_triangles_3d = opp_unfolding.compute_3d_glue_flap_triangles_inside_face(opp_face.index, self.halfedge_to_edge[halfedge], self.flap_angle, self.flap_height)
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
