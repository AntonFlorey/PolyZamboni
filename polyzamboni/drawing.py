"""
Handling of all the user feedback rendering. 
"""

import bpy
import bmesh
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
import time

from . import drawing_backend
from . import io
from .zambonipolice import check_if_polyzamobni_data_exists_and_fits_to_bmesh, check_if_page_numbers_and_transforms_exist_for_all_components
from .exporters import paper_sizes
from .printprepper import ComponentPrintData, ColoredTriangleData, CutEdgeData, FoldEdgeData, GlueFlapEdgeData, FoldEdgeAtGlueFlapData, create_print_data_for_all_components
from .geometry import AffineTransform2D

# colors (RWTH)
BLUE = (  0 / 255,  84 / 255, 159 / 255, 1.0)
MAGENTA = (227 / 255,   0 / 255, 102 / 255, 1.0)
YELLOW = (255 / 255, 237 / 255,   0 / 255, 1.0)
PETROL = (  0 / 255,  97 / 255, 101 / 255, 1.0)
TEAL = (  0 / 255, 152 / 255, 161 / 255, 1.0)
GREEN = ( 87 / 255, 171 / 255,  39 / 255, 1.0)
MAY_GREEN = (189 / 255, 205 / 255,   0 / 255, 1.0)
ORANGE = (246 / 255, 168 / 255,   0 / 255, 1.0)
RED = (204 / 255,   7 / 255,  30 / 255, 1.0)
BORDEAUX = (161 / 255,  16 / 255,  53 / 255, 1.0)
PURPLE = ( 97 / 255,  33 / 255,  88 / 255, 1.0)
LILAC = (122 / 255, 111 / 255, 172 / 255, 1.0)

# colors (POLYZAMBONI)
POLYZAMBONI_RED = (  191 / 255,  77 / 255, 77 / 255, 1.0)
POLYZAMBONI_ORANGE = (  191 / 255,  124 / 255, 77 / 255, 1.0)
POLYZAMBONI_YELLOW = (  191 / 255,  191 / 255, 77 / 255, 1.0)
POLYZAMBONI_GREEN = (  124 / 255,  191 / 255, 77 / 255, 1.0)
POLYZAMBONI_TEAL = (  74 / 255,  161 / 255, 129 / 255, 1.0)
POLYZAMBONI_BLUE = (  71 / 255,  141 / 255, 179 / 255, 1.0)
POLYZAMBONI_LILA = (  134 / 255,  71 / 255, 179 / 255, 1.0)
POLYZAMBONI_ROSE = (  179 / 255,  71 / 255, 116 / 255, 1.0)

BLACK = (0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)

# Keep track of active draw callbacks
_drawing_handle_user_provided_cuts = None
_drawing_handle_locked_edges = None
_drawing_handle_auto_completed_cuts = None
_drawing_handle_region_quality_triangles = None
_drawing_handle_glue_flaps = None
_drawing_handle_pages = None

class ColorGenerator():
    """ A simple color generator"""

    def __init__(self):
        self.colors = [
            BLUE,
            MAGENTA,
            YELLOW,
            PETROL,
            TEAL,
            GREEN,
            MAY_GREEN,
            ORANGE,
            RED,
            BORDEAUX,
            PURPLE,
            LILAC
        ]
        self.index = 0

    def next_color(self):
        col = self.colors[self.index]
        self.index = (self.index + 1) % len(self.colors)
        return col
    
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

#################################
#     General Draw Callbacks    #
#################################

def deactivate_draw_callback(callback_handle, region_type='WINDOW'):
    if callback_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(callback_handle, region_type)

def lines_2D_draw_callback(line_array, color, width=3):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    prev_line_width = gpu.state.line_width_get()
    gpu.state.line_width_set(width)
    batch = batch_for_shader(shader, 'LINES', {"pos" : line_array})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(prev_line_width)

def multicolored_triangles_2D_draw_callback(vertex_positions, vertex_colors):
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    # prepare and draw batch
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions, "color" : vertex_colors})
    shader.bind()
    batch.draw(shader)

def triangles_2D_draw_callback(vertex_positions, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    # prepare and draw batch
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions})
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)

def lines_draw_callback(line_array, color, width=3):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    prev_line_width = gpu.state.line_width_get()
    gpu.state.line_width_set(width)
    batch = batch_for_shader(shader, 'LINES', {"pos": line_array})
    shader.bind()
    shader.uniform_float("color", color)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    batch.draw(shader)
    # restore opengl defaults
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(prev_line_width)

def triangles_draw_callback(vertex_positions, triangle_indices, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.face_culling_set("BACK")
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    # prepare and draw batch
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions}, indices=triangle_indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    # restore opengl defaults
    gpu.state.depth_mask_set(False)

#################################
#        Handmade Cuts          #
#################################

def hide_user_provided_cuts():
    global _drawing_handle_user_provided_cuts
    deactivate_draw_callback(_drawing_handle_user_provided_cuts)
    _drawing_handle_user_provided_cuts = None

def show_user_provided_cuts(cuts_as_line_array, dotted_line_length = 0.1):
    hide_user_provided_cuts()
    global _drawing_handle_user_provided_cuts
    user_cuts_color = RED
    dotted_cuts = make_dotted_lines(cuts_as_line_array, dotted_line_length)
    _drawing_handle_user_provided_cuts = bpy.types.SpaceView3D.draw_handler_add(lines_draw_callback, (dotted_cuts, user_cuts_color), "WINDOW", "POST_VIEW")

#################################
#         Locked Edges          #
#################################

def hide_locked_edges():
    global _drawing_handle_locked_edges
    deactivate_draw_callback(_drawing_handle_locked_edges)
    _drawing_handle_locked_edges = None

def show_locked_edges(cuts_as_line_array, dotted_line_length = 0.1):
    hide_locked_edges()
    global _drawing_handle_locked_edges
    locked_edges_color = GREEN
    dotted_cuts = make_dotted_lines(cuts_as_line_array, dotted_line_length)
    _drawing_handle_locked_edges = bpy.types.SpaceView3D.draw_handler_add(lines_draw_callback, (dotted_cuts, locked_edges_color), "WINDOW", "POST_VIEW")

#################################
#           Auto Cuts           #
#################################

def hide_auto_completed_cuts():
    global _drawing_handle_auto_completed_cuts
    deactivate_draw_callback(_drawing_handle_auto_completed_cuts)
    _drawing_handle_auto_completed_cuts = None

def show_auto_completed_cuts(cuts_as_line_array, dotted_line_length = 0.1):
    hide_auto_completed_cuts() 
    global _drawing_handle_auto_completed_cuts
    auto_cuts_color = TEAL
    dotted_auto_cuts = make_dotted_lines(cuts_as_line_array, dotted_line_length)
    _drawing_handle_auto_completed_cuts = bpy.types.SpaceView3D.draw_handler_add(lines_draw_callback, (dotted_auto_cuts, auto_cuts_color), "WINDOW", "POST_VIEW")

#################################
#    Region Quality Triangles   #
#################################

quality_color_mapping = {
    drawing_backend.ComponentQuality.PERFECT_REGION : POLYZAMBONI_GREEN,
    drawing_backend.ComponentQuality.BAD_GLUE_FLAPS_REGION : POLYZAMBONI_YELLOW,
    drawing_backend.ComponentQuality.OVERLAPPING_REGION : POLYZAMBONI_ORANGE,
    drawing_backend.ComponentQuality.NOT_FOLDABLE_REGION : POLYZAMBONI_RED
}

def hide_region_quality_triangles():
    global _drawing_handle_region_quality_triangles
    deactivate_draw_callback(_drawing_handle_region_quality_triangles)
    _drawing_handle_region_quality_triangles = None

def region_quality_triangles_draw_callback(vertex_positions, regions_by_quality):
    for quality, triangle_indices in regions_by_quality.items():
        if quality not in quality_color_mapping:
            print("POLYZAMBONI WARNING: Quality of provided region is not known!")
            continue
        triangles_draw_callback(vertex_positions, triangle_indices, quality_color_mapping[quality])

def show_region_quality_triangles(vertex_positions, regions_by_quality):
    hide_region_quality_triangles()
    global _drawing_handle_region_quality_triangles
    _drawing_handle_region_quality_triangles = bpy.types.SpaceView3D.draw_handler_add(region_quality_triangles_draw_callback, (vertex_positions, regions_by_quality), "WINDOW", "POST_VIEW")

#################################
#           Glue Flaps          #
#################################

flap_quality_color_mapping = {
    drawing_backend.GlueFlapQuality.GLUE_FLAP_NO_OVERLAPS : PETROL,
    drawing_backend.GlueFlapQuality.GLUE_FLAP_TO_LARGE : POLYZAMBONI_LILA,
    drawing_backend.GlueFlapQuality.GLUE_FLAP_WITH_OVERLAPS : BORDEAUX,
}

def hide_glue_flaps():
    global _drawing_handle_glue_flaps
    deactivate_draw_callback(_drawing_handle_glue_flaps)
    _drawing_handle_glue_flaps = None

def glue_flaps_draw_callback(flaps_by_quality):
    for quality, glue_flap_lines in flaps_by_quality.items():
        if quality not in flap_quality_color_mapping:
            print("POLYZAMBONI WARNING: Quality of provided glue flaps is not known!")
        lines_draw_callback(glue_flap_lines, flap_quality_color_mapping[quality], width=2.5)

def show_glue_flaps(flaps_by_quality):
    hide_glue_flaps()
    global _drawing_handle_glue_flaps
    _drawing_handle_glue_flaps = bpy.types.SpaceView3D.draw_handler_add(glue_flaps_draw_callback, (flaps_by_quality,), "WINDOW", "POST_VIEW")

#################################
#              Pages            #
#################################

def hide_pages():
    global _drawing_handle_pages
    if _drawing_handle_pages is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_drawing_handle_pages, "WINDOW")
        _drawing_handle_pages = None

def page_bg_draw_callback(page_verts, selected_page_lines, other_pages_lines):
    triangles_2D_draw_callback(page_verts, WHITE)
    if selected_page_lines is not None:
        lines_2D_draw_callback(selected_page_lines, ORANGE)
    lines_2D_draw_callback(other_pages_lines, BLACK)

def component_draw_callback(component_triangle_verts, component_triangle_colors, component_lines):
    multicolored_triangles_2D_draw_callback(component_triangle_verts, component_triangle_colors)
    lines_2D_draw_callback(component_lines, BLACK, 1.5)

def pages_draw_callback(page_verts, selected_page_lines, other_pages_lines, component_triangle_verts, component_triangle_colors, component_lines):
    # pages in the backgound
    page_bg_draw_callback(page_verts, selected_page_lines, other_pages_lines)
    # connected components
    component_draw_callback(component_triangle_verts, component_triangle_colors, component_lines)

def show_pages(num_pages, components_per_page, paper_size = paper_sizes["A4"], selected_page = None, margin_between_pages = 1.0, pages_per_row = 2, fold_angle_th = 0.0):
    selected_page_lines = None if selected_page is None else []
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
    
    # components on pages
    full_lines = []
    convex_lines = []
    concave_lines = []
    component_bg_verts = []
    component_bg_colors = []

    assert num_pages >= len(components_per_page)
    for page_index, components_on_page in enumerate(components_per_page):
        row_index = page_index % pages_per_row
        col_index = page_index // pages_per_row
        page_anchor = np.array([row_index * (paper_size[0] + margin_between_pages), -col_index * (paper_size[1] + margin_between_pages)])

        current_component : ComponentPrintData
        for current_component in components_on_page:
            page_transform = current_component.page_transform
            for cut_edge_data in current_component.cut_edges + current_component.glue_flap_edges:
                full_lines += [page_anchor + page_transform * coord for coord in cut_edge_data.coords]
            fold_edge_data : FoldEdgeData
            for fold_edge_data in current_component.fold_edges:
                if fold_edge_data.fold_angle <= fold_angle_th:
                    continue
                coords = [page_anchor + page_transform * coord for coord in fold_edge_data.coords]
                if fold_edge_data.is_convex:
                    convex_lines += coords
                else:
                    concave_lines += coords
            fold_edge_at_flap : FoldEdgeAtGlueFlapData
            for fold_edge_at_flap in current_component.fold_edges_at_flaps:
                coords = [page_anchor + page_transform * coord for coord in fold_edge_at_flap.coords]
                if fold_edge_at_flap.fold_angle <= fold_angle_th:
                    full_lines += coords
                    continue
                if fold_edge_at_flap.is_convex:
                    convex_lines += coords
                else:
                    concave_lines += coords
            tri_data : ColoredTriangleData
            for tri_data in current_component.colored_triangles:
                component_bg_verts += [page_anchor + page_transform * coord for coord in tri_data.coords]
                component_bg_colors += [tri_data.color] * 3
    convex_lines = make_dotted_lines(convex_lines, 0.2)
    concave_lines = make_dotted_lines(concave_lines, 0.2, linestyle=[1,1,4,1])
    component_lines = full_lines + convex_lines + concave_lines

    hide_pages()
    global _drawing_handle_pages
    _drawing_handle_pages = bpy.types.SpaceImageEditor.draw_handler_add(pages_draw_callback, (page_verts, selected_page_lines, other_page_lines, component_bg_verts, component_bg_colors, component_lines), "WINDOW", "POST_VIEW")

#################################
#          Update all           #
#################################

def hide_all_drawings():
    hide_user_provided_cuts()
    hide_locked_edges()
    hide_auto_completed_cuts()
    hide_region_quality_triangles()
    hide_glue_flaps()

def redraw_image_editor(context : bpy.types.Context):
    for area in context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            area.tag_redraw()

def redraw_view_3d(context : bpy.types.Context):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def update_all_page_layout_drawings(self, context):
    try:
        draw_settings = context.scene.polyzamboni_drawing_settings
    except AttributeError:
        print("POLYZAMBONI WARNING: Context not yet available...")
        return
    
    # hide everything
    hide_pages()

    ao : bpy.types.Object = context.active_object
    if not draw_settings.show_page_layout or ao.type != 'MESH' or not ao.data.polyzamboni_general_mesh_props.has_attached_paper_model:
        redraw_image_editor(context)
        return
    
    #obtain selected paper model
    active_mesh = ao.data
    general_mesh_props = active_mesh.polyzamboni_general_mesh_props

    if not check_if_page_numbers_and_transforms_exist_for_all_components(active_mesh):
        print("DEBUG: No page layout to draw found!")
        redraw_image_editor(context)
        return
    
    # collect all component print data
    component_print_data = create_print_data_for_all_components(ao, general_mesh_props.model_scale)

    # read and set correct page transforms
    page_transforms_per_component = io.read_page_transforms(active_mesh)
    current_component_print_data : ComponentPrintData
    for current_component_print_data in component_print_data:
        current_component_print_data.page_transform = page_transforms_per_component[current_component_print_data.og_component_id]

    # create page layout
    page_numbers_per_components = io.read_page_numbers(active_mesh)
    num_pages = max(page_numbers_per_components.values()) + 1 if len(page_numbers_per_components) > 0 else 0
    components_on_pages = [[] for _ in range(num_pages)]
    for current_component_print_data in component_print_data:
        components_on_pages[page_numbers_per_components[current_component_print_data.og_component_id]].append(current_component_print_data)

    # erstmal so
    show_pages(num_pages, components_on_pages, paper_sizes[general_mesh_props.paper_size], None, fold_angle_th=draw_settings.hide_fold_edge_angle_th)
    redraw_image_editor(context)

def update_all_polyzamboni_drawings(self, context):
    try:
        draw_settings = context.scene.polyzamboni_drawing_settings
    except AttributeError:
        print("POLYZAMBONI WARNING: Context not yet available...")
        return
    
    # hide everything
    hide_all_drawings()

    ao : bpy.types.Object = context.active_object
    # draw user provided cuts
    if not draw_settings.drawing_enabled or ao.type != 'MESH' or not ao.data.polyzamboni_general_mesh_props.has_attached_paper_model:
        redraw_view_3d(context)
        return
    
    # obtain selected paper model to draw
    active_mesh = ao.data
    general_mesh_props = active_mesh.polyzamboni_general_mesh_props
    bm = bmesh.new()
    bm.from_mesh(active_mesh)
    if not check_if_polyzamobni_data_exists_and_fits_to_bmesh(active_mesh, bm):
        print("POLYZAMBONI WARNING: Attached paper model data invalid! CAN NOT DRAW D:")
        redraw_view_3d(context)
        return
    world_matrix = ao.matrix_world
    edge_constraints = io.read_edge_constraints_dict(active_mesh)
    connected_components, face_to_component_dict =io.read_connected_components(active_mesh)

    # draw user provided cuts
    show_user_provided_cuts(drawing_backend.mesh_edge_id_list_to_coordinate_list(bm, io.read_manual_cut_edges(active_mesh), draw_settings.normal_offset, world_matrix), dotted_line_length=draw_settings.dotted_line_length)
    show_locked_edges(drawing_backend.mesh_edge_id_list_to_coordinate_list(bm, io.read_locked_edges(active_mesh), draw_settings.normal_offset, world_matrix), dotted_line_length=draw_settings.dotted_line_length)
    show_auto_completed_cuts(drawing_backend.mesh_edge_id_list_to_coordinate_list(bm, io.read_auto_cut_edges(active_mesh), draw_settings.normal_offset, world_matrix), dotted_line_length=draw_settings.dotted_line_length)

    if draw_settings.color_faces_by_quality:
        all_v_positions, quality_dict = drawing_backend.get_triangle_list_per_cluster_quality(active_mesh, bm, draw_settings.normal_offset, edge_constraints, 
                                                                                              world_matrix, connected_components)
        show_region_quality_triangles(all_v_positions, quality_dict)

    if draw_settings.show_glue_flaps:
        # so far, the glue flap geometry gets computed on the fly each time this function is called
        glue_flap_quality_dict = drawing_backend.get_lines_array_per_glue_flap_quality(active_mesh, bm, draw_settings.normal_offset, world_matrix, general_mesh_props.glue_flap_angle, 
                                                                                       general_mesh_props.glue_flap_height, connected_components, face_to_component_dict)
        show_glue_flaps(glue_flap_quality_dict)

    # Trigger a redraw of all screen areas
    redraw_view_3d(context)
