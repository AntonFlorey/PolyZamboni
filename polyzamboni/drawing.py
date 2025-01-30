import bpy
import gpu
import bgl
import numpy as np
from gpu_extras.batch import batch_for_shader
from . import globals
from . import cutgraph
from .constants import PERFECT_REGION, BAD_GLUE_FLAPS_REGION, OVERLAPPING_REGION, NOT_FOLDABLE_REGION, GLUE_FLAP_NO_OVERLAPS, GLUE_FLAP_WITH_OVERLAPS, GLUE_FLAP_TO_LARGE

# colors
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

# Keep track of active draw callbacks 
_drawing_handle_errors = None
_drawing_handle_user_provided_cuts = None
_drawing_handle_locked_edges = None
_drawing_handle_auto_completed_cuts = None
_drawing_handle_region_quality_triangles = None
_drawing_handle_glue_flaps = None

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
    
def make_dotted_lines(line_array, target_line_length, max_segments = 100):
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
        for segment_i in range(1, total_segments, 2):
            t_from = segment_i / total_segments
            t_to = (segment_i + 1) / total_segments
            dotted_lines_array.append((1.0 - t_from) * v_from + t_from * v_to)
            dotted_lines_array.append((1.0 - t_to) * v_from + t_to * v_to)
        line_index += 2
    return dotted_lines_array

#################################
#     General Draw Callbacks    #
#################################

def deactivate_draw_callback(callback_handle, region_type='WINDOW'):
    if callback_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(callback_handle, region_type)

def lines_draw_callback(line_array, color, width=3.0):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.line_width_set(width)
    batch = batch_for_shader(shader, 'LINES', {"pos": line_array})
    shader.uniform_float("color", color)
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    batch.draw(shader)
    # restore opengl defaults
    gpu.state.depth_mask_set(False)
    gpu.state.line_width_set(1.0)

def triangles_draw_callback(vertex_positions, triangle_indices, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.face_culling_set("BACK")
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(True)
    # prepare and draw batch
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertex_positions}, indices=triangle_indices)
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
    auto_cuts_color = BLUE
    dotted_auto_cuts = make_dotted_lines(cuts_as_line_array, dotted_line_length)
    _drawing_handle_auto_completed_cuts = bpy.types.SpaceView3D.draw_handler_add(lines_draw_callback, (dotted_auto_cuts, auto_cuts_color), "WINDOW", "POST_VIEW")

#################################
#    Region Quality Triangles   #
#################################

quality_color_mapping = {
    PERFECT_REGION : GREEN,
    BAD_GLUE_FLAPS_REGION : YELLOW,
    OVERLAPPING_REGION : ORANGE,
    NOT_FOLDABLE_REGION : RED
}

def hide_region_quality_triangles():
    global _drawing_handle_region_quality_triangles
    deactivate_draw_callback(_drawing_handle_region_quality_triangles)
    _drawing_handle_region_quality_triangles = None

def region_quality_triangles_draw_callback(vertex_positions, regions_by_quality):
    for quality, triangle_indices in regions_by_quality.items():
        if quality not in quality_color_mapping:
            print("WARNING: Quality of provided region is not known!")
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
    GLUE_FLAP_NO_OVERLAPS : MAY_GREEN,
    GLUE_FLAP_TO_LARGE : TEAL,
    GLUE_FLAP_WITH_OVERLAPS : MAGENTA,
}

def hide_glue_flaps():
    global _drawing_handle_glue_flaps
    deactivate_draw_callback(_drawing_handle_glue_flaps)
    _drawing_handle_glue_flaps = None

def glue_flaps_draw_callback(flaps_by_quality):
    for quality, glue_flap_lines in flaps_by_quality.items():
        if quality not in flap_quality_color_mapping:
            print("WARNING: Quality of provided glue flaps is not known!")
        lines_draw_callback(glue_flap_lines, flap_quality_color_mapping[quality])

def show_glue_flaps(flaps_by_quality):
    hide_glue_flaps()
    global _drawing_handle_glue_flaps
    _drawing_handle_glue_flaps = bpy.types.SpaceView3D.draw_handler_add(glue_flaps_draw_callback, (flaps_by_quality,), "WINDOW", "POST_VIEW")

#################################
#          Update all           #
#################################

def hide_all_drawings():
    hide_user_provided_cuts()
    hide_locked_edges()
    hide_auto_completed_cuts()
    hide_region_quality_triangles()
    hide_glue_flaps()

def update_all_polyzamboni_drawings(self, context):
    try:
        draw_settings = context.scene.polyzamboni_drawing_settings
    except AttributeError:
        print("Context not yet available...")
        # Trigger a redraw of all screen areas
        return
    
    # hide everything
    hide_all_drawings()

    # draw user provided cuts
    if not draw_settings.drawing_enabled or globals.PZ_CURRENT_CUTGRAPH_ID is None:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return
    
    # obtain current cutgraph to draw
    cutgraph_to_draw : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[globals.PZ_CURRENT_CUTGRAPH_ID]

    # draw user provided cuts
    show_user_provided_cuts(cutgraph_to_draw.mesh_edge_id_list_to_coordinate_list(cutgraph_to_draw.get_manual_cuts_list(), draw_settings.normal_offset), dotted_line_length=draw_settings.dotted_line_length)
    show_locked_edges(cutgraph_to_draw.mesh_edge_id_list_to_coordinate_list(cutgraph_to_draw.get_locked_edges_list(), draw_settings.normal_offset), dotted_line_length=draw_settings.dotted_line_length)

    if draw_settings.show_auto_completed_cuts:
        pass

    if draw_settings.color_faces_by_quality:
        all_v_positions, quality_dict = cutgraph_to_draw.get_triangle_list_per_cluster_quality(draw_settings.normal_offset)
        show_region_quality_triangles(all_v_positions, quality_dict)

    if draw_settings.show_glue_flaps:
        glue_flap_quality_dict = cutgraph_to_draw.get_lines_array_per_glue_flap_quality(draw_settings.normal_offset)
        show_glue_flaps(glue_flap_quality_dict)

    # Trigger a redraw of all screen areas
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def draw_errors():
    print("drawing not implemented yet")