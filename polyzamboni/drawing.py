"""
Handling of all the user feedback rendering. 
"""

import bpy
import bmesh
import gpu
import blf
import numpy as np
from gpu_extras.batch import batch_for_shader
import time
from enum import Enum

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
GRAY = (0.5, 0.5, 0.5, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)

# Keep track of active draw callbacks
_drawing_handle_user_provided_cuts = None
_drawing_handle_locked_edges = None
_drawing_handle_auto_completed_cuts = None
_drawing_handle_region_quality_triangles = None
_drawing_handle_glue_flaps = None
_drawing_handle_pages = None
_drawing_handle_step_numbers = None

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

#################################
#     General Draw Callbacks    #
#################################

def deactivate_draw_callback(callback_handle, region_type='WINDOW'):
    if callback_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(callback_handle, region_type)

def lines_2D_draw_callback(line_array, color, width=3):
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    prev_line_width = gpu.state.line_width_get()
    prev_blend = gpu.state.blend_get()
    gpu.state.line_width_set(width)
    gpu.state.blend_set('ALPHA')
    batch = batch_for_shader(shader, 'LINES', {"pos" : line_array})
    shader.bind()   
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.line_width_set(prev_line_width)
    gpu.state.blend_set(prev_blend)

def multicolored_triangles_2D_draw_callback(vertex_positions, vertex_colors):
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    prev_blend = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')
    # prepare and draw batch
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions, "color" : vertex_colors})
    shader.bind()
    batch.draw(shader)
    gpu.state.blend_set(prev_blend)

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
    dotted_cuts = drawing_backend.make_dotted_lines(cuts_as_line_array, dotted_line_length)
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
    dotted_cuts = drawing_backend.make_dotted_lines(cuts_as_line_array, dotted_line_length)
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
    dotted_auto_cuts = drawing_backend.make_dotted_lines(cuts_as_line_array, dotted_line_length)
    _drawing_handle_auto_completed_cuts = bpy.types.SpaceView3D.draw_handler_add(lines_draw_callback, (dotted_auto_cuts, auto_cuts_color), "WINDOW", "POST_VIEW")

#################################
#    Region Quality Triangles   #
#################################

quality_color_mapping = {
    drawing_backend.ComponentQuality.PERFECT_REGION : POLYZAMBONI_GREEN,
    drawing_backend.ComponentQuality.BAD_GLUE_FLAPS_REGION : POLYZAMBONI_YELLOW,
    drawing_backend.ComponentQuality.OVERLAPPING_REGION : POLYZAMBONI_ORANGE,
    drawing_backend.ComponentQuality.NOT_FOLDABLE_REGION : POLYZAMBONI_RED,
    drawing_backend.ComponentQuality.NOT_SELECTED_REGION : GRAY
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
    global _drawing_handle_step_numbers
    if _drawing_handle_pages is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_drawing_handle_pages, "WINDOW")
        _drawing_handle_pages = None
    if _drawing_handle_step_numbers is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(_drawing_handle_step_numbers, "WINDOW")
        _drawing_handle_step_numbers = None

def page_bg_draw_callback(page_verts, selected_page_lines, other_pages_lines):
    triangles_2D_draw_callback(page_verts, WHITE)
    if selected_page_lines is not None:
        lines_2D_draw_callback(selected_page_lines, PETROL)
    lines_2D_draw_callback(other_pages_lines, BLACK)

def component_draw_callback(component_triangle_verts, component_triangle_colors, component_lines, selected_component_lines = None):
    multicolored_triangles_2D_draw_callback(component_triangle_verts, component_triangle_colors)
    lines_2D_draw_callback(component_lines, BLACK, 1.5)
    if selected_component_lines is not None:
        lines_2D_draw_callback(selected_component_lines, ORANGE, 2)

def build_step_numbers_draw_callback(numbers_with_positions, font_size):
    scale = 0.01
    blf.color(0, 0, 0, 0, 1)
    blf.size(0, font_size / scale)
    blf.shadow(0, 0, 0, 0, 0, 1)
    with gpu.matrix.push_pop():
        gpu.matrix.scale_uniform(scale)
        for num_with_pos in numbers_with_positions:
            step_number = num_with_pos[0]
            dim = blf.dimensions(0, str(step_number))
            position = num_with_pos[1] / scale - np.array([dim[0] / 2, dim[1] / 2])
            blf.position(0, position[0], position[1], 0)
            blf.draw(0, str(step_number))

def pages_draw_callback(page_verts, selected_page_lines, other_pages_lines, component_triangle_verts, component_triangle_colors, component_lines, 
                        selected_component_lines, numbers_with_positions, font_size):
    # pages in the backgound
    page_bg_draw_callback(page_verts, selected_page_lines, other_pages_lines)
    # connected components
    component_draw_callback(component_triangle_verts, component_triangle_colors, component_lines, selected_component_lines)
    # build step numbers
    build_step_numbers_draw_callback(numbers_with_positions, font_size)

def show_pages_with_procomputed_render_data(render_data_per_component, num_pages, paper_size = paper_sizes["A4"], selected_page = None, selected_component = None, 
                                            margin_between_pages = 1.0, pages_per_row = 2, color_components = True, show_step_numbers = True):
    page_verts, selected_page_lines, other_page_lines = drawing_backend.compute_backgound_paper_render_data(num_pages, paper_size, selected_page, margin_between_pages, pages_per_row)
    combined_render_data = drawing_backend.combine_layout_render_data_of_all_components(render_data_per_component, selected_component, color_components, show_step_numbers)

    hide_pages()
    global _drawing_handle_pages
    _drawing_handle_pages = bpy.types.SpaceImageEditor.draw_handler_add(pages_draw_callback, 
                                                                        (page_verts, selected_page_lines, other_page_lines, *combined_render_data, 1), 
                                                                        "WINDOW", "POST_VIEW")

def show_pages(num_pages, components_per_page, paper_size = paper_sizes["A4"], selected_page = None, selected_component = None, margin_between_pages = 1.0, 
               pages_per_row = 2, fold_angle_th = 0.0, color_components = True, show_step_numbers = True):

    page_verts, selected_page_lines, other_page_lines = drawing_backend.compute_backgound_paper_render_data(num_pages, paper_size, selected_page, margin_between_pages, pages_per_row)
    
    # components on pages
    render_data_per_component = drawing_backend.compute_page_layout_render_data_of_all_components(components_per_page, paper_size, margin_between_pages, pages_per_row, fold_angle_th)
    combined_render_data = drawing_backend.combine_layout_render_data_of_all_components(render_data_per_component, selected_component, color_components, show_step_numbers)

    hide_pages()
    global _drawing_handle_pages
    _drawing_handle_pages = bpy.types.SpaceImageEditor.draw_handler_add(pages_draw_callback, 
                                                                        (page_verts, selected_page_lines, other_page_lines, *combined_render_data, 1), 
                                                                        "WINDOW", "POST_VIEW")

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
    show_pages(num_pages, components_on_pages, paper_sizes[general_mesh_props.paper_size], None, 
               fold_angle_th=draw_settings.hide_fold_edge_angle_th, 
               color_components=draw_settings.show_component_colors, 
               show_step_numbers=draw_settings.show_build_step_numbers)
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
        selected_component_id = general_mesh_props.selected_component_id if general_mesh_props.selected_component_id != -1 else None
        all_v_positions, quality_dict = drawing_backend.get_triangle_list_per_cluster_quality(active_mesh, bm, draw_settings.normal_offset, edge_constraints, 
                                                                                              world_matrix, connected_components, selected_component_id)
        show_region_quality_triangles(all_v_positions, quality_dict)

    if draw_settings.show_glue_flaps:
        # so far, the glue flap geometry gets computed on the fly each time this function is called
        glue_flap_quality_dict = drawing_backend.get_lines_array_per_glue_flap_quality(active_mesh, bm, draw_settings.normal_offset, world_matrix, general_mesh_props.glue_flap_angle, 
                                                                                       general_mesh_props.glue_flap_height, connected_components, face_to_component_dict)
        show_glue_flaps(glue_flap_quality_dict)

    # Trigger a redraw of all screen areas
    redraw_view_3d(context)
