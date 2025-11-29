"""
This file contains several high level functions that meaningful combine multiple polyzamboni operations.
"""

import bpy
from bpy.types import Mesh, Object
from bmesh.types import BMesh
from enum import Enum
import math
import numpy as np

from . import geometry
from . import io
from . import units
from . import exporters 
from . import printprepper
from . import utils
from .papermodel import PaperModel
from .printprepper import create_print_data_for_all_components, fit_components_on_pages, ComponentPrintData, ColoredTriangleData, GlueFlapData
from .exporters import paper_sizes, ExportSettings


#################################
#    Init, delete and update    #
#################################

def get_indices_of_non_manifold_vertices(bm : BMesh):
    non_manifold_vertices = set()
    for vertex in bm.verts:
        if not vertex.is_manifold:
            non_manifold_vertices.add(vertex.index)
    for edge in bm.edges:
        if not (edge.is_boundary or edge.is_manifold):
            for vertex in edge.verts:
                non_manifold_vertices.add(vertex.index)
    return non_manifold_vertices

def get_indices_of_multi_touching_faces(bm : BMesh):
    face_pair_set = set()
    multi_touching_face_indices = set()
    for edge in bm.edges:
        if edge.is_boundary:
            continue
        assert len(edge.link_faces) == 2
        face_pair_key = tuple(sorted([f.index for f in edge.link_faces]))
        if face_pair_key in face_pair_set:
            for f in edge.link_faces:
                multi_touching_face_indices.add(f.index)
        face_pair_set.add(face_pair_key)
    return multi_touching_face_indices

def get_indices_of_not_triangulatable_faces(bm : BMesh):
    non_triangulatable_faces = set()
    for face in bm.faces:
        _, tri_ids = geometry.triangulate_3d_polygon([v.co for v in face.verts], face.normal)
        if len(tri_ids) != len(face.verts) - 2:
            non_triangulatable_faces.add(face.index)
    return non_triangulatable_faces

def initialize_paper_model(mesh : Mesh):
    new_papermodel = PaperModel.new_from_mesh(mesh)
    new_papermodel.close()

def delete_paper_model(mesh : Mesh):
    zamboni_props = mesh.polyzamboni_general_mesh_props
    zamboni_props.has_attached_paper_model = False
    io.remove_all_polyzamboni_data(mesh)

def sync_paper_model_with_mesh_geometry(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.apply_mesh_geometry_changes()

#################################
#   Cut, clear and glue edges   #
#################################

def cut_edges(mesh : Mesh, indices_of_edges_to_cut):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.cut_edges(indices_of_edges_to_cut)

def glue_edges(mesh : Mesh, indices_of_edges_to_lock):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.glue_edges(indices_of_edges_to_lock)

def clear_edges(mesh : Mesh, indices_of_edges_to_clear):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.clear_edges(indices_of_edges_to_clear)

def add_cutout_region(mesh : Mesh, face_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.cut_out_face_region(face_indices)

def add_cuts_between_different_materials(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.add_cuts_between_different_materials()

def remove_auto_cuts(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.remove_all_auto_cuts()

#################################
#           Glue Flaps          #
#################################

def update_all_flap_geometry(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.update_all_flap_geometry()

def add_glue_flaps(mesh : Mesh, edge_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.add_glue_flaps_around_edges(edge_indices)

def remove_glue_flaps(mesh : Mesh, edge_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.remove_glue_flaps_around_edges(edge_indices)

def flip_glue_flaps(mesh : Mesh, edge_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.flip_glue_flaps_around_edges(edge_indices)

def recompute_all_glue_flaps(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.compute_all_glueflaps_greedily()

def smart_trim_glue_flaps(mesh : Mesh, edge_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.smart_trim_glue_edges_at_selected_edges(edge_indices)

#################################
#           Build Steps         #
#################################

def compute_build_step_numbers(mesh : Mesh, selected_start_face_ids):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.compute_build_step_numbers(selected_start_face_ids)

#################################
#          Exporting            #
#################################

def write_custom_split_property_row(layout : bpy.types.UILayout, text, data, prop_name, split_factor, active=True):
    custom_row = layout.row().split(factor=split_factor, align=True)
    col_1, col_2 = (custom_row.column(), custom_row.column())
    col_1.label(text=text)
    col_2.prop(data, prop_name, text="")
    custom_row.active = active

def line_style_property_draw(layout : bpy.types.UILayout, text, data, prop_name, split_factor, linestyles):
    ls_row = layout.row()
    #ls_row.split(factor=split_factor, align=True)
    ls_text_col, ls_enum_col = (ls_row.column(), ls_row.column())
    ls_text_col.row().label(text=text)
    ls_flow = ls_enum_col.column_flow(columns=3, align=True)#len(self.linestyles), align=True)
    for style in linestyles:
        ls_flow.column(align=True).prop_enum(data, prop_name, style[0])

def export_draw_func(operator : bpy.types.Operator):
    layout = operator.layout
    
    general_settings_props = operator.properties.general_settings
    line_settings_props = operator.properties.line_settings
    texture_settings_props = operator.properties.texture_settings

    # General settings
    layout.label(text="General print settings", icon="TOOL_SETTINGS")
    general_settings = layout.box()

    # use custom layout
    if operator.user_defined_page_layout_exists:
        write_custom_split_property_row(general_settings, "Use custom layout", general_settings_props, "use_custom_layout", 0.6)
        general_settings.separator(type="LINE")

    if not (operator.user_defined_page_layout_exists and general_settings_props.use_custom_layout):
        # paper size
        write_custom_split_property_row(general_settings, "Paper size", general_settings_props, "paper_size", 0.6)
        if general_settings_props.paper_size == "Custom":
            page_size_box = general_settings.box()
            page_size_box.label(text="Custom page size")
            sizing_row = page_size_box.row()
            sizing_row.column().prop(general_settings_props, "custom_page_width")
            sizing_row.column().prop(general_settings_props, "custom_page_height")

        # scaling mode
        scaling_mode_row = general_settings.row().column_flow(columns=2, align=True)
        scaling_mode_row.column(align=True).prop_enum(general_settings_props, "scaling_mode", "HEIGHT")
        scaling_mode_row.column(align=True).prop_enum(general_settings_props, "scaling_mode", "SCALE")

        if general_settings_props.scaling_mode == "HEIGHT":
            # target model height
            write_custom_split_property_row(general_settings, "Target height", general_settings_props, "target_model_height", 0.6)
            curr_scale_factor = general_settings_props.target_model_height / operator.mesh_height
            # set correct scaling
            general_settings_props.sizing_scale = curr_scale_factor 
        elif general_settings_props.scaling_mode == "SCALE":
            write_custom_split_property_row(general_settings, "Model scale", general_settings_props, "sizing_scale", 0.6)
            curr_scale_factor = general_settings_props.sizing_scale
            # set target model height
            general_settings_props.target_model_height = curr_scale_factor * operator.mesh_height
        
        if general_settings_props.sizing_scale > operator.max_fit_scaling:
            general_settings.row().label(icon="ERROR", text="A piece does not fit on one page!")
        # margin
        write_custom_split_property_row(general_settings, "Page margin", general_settings_props, "page_margin", 0.6)
        # island spacing
        write_custom_split_property_row(general_settings, "Island spacing", general_settings_props, "space_between_components", 0.6)
        # one mat per page
        write_custom_split_property_row(general_settings, "One material per page", general_settings_props, "one_material_per_page", 0.6)
    # side of prints
    write_custom_split_property_row(general_settings, "Prints inside", general_settings_props, "print_on_inside", 0.6)
    # font settings
    general_settings.separator(factor=0.2)
    text_settings_row = general_settings.row().split(factor=0.55)
    text_settings_left_col, text_settings_size_col, text_settings_color_col = (text_settings_row.column(), text_settings_row.column(), text_settings_row.column())
    text_settings_left_col.row().label(text="Print?")
    text_settings_size_col.row().label(text="Size")
    text_settings_color_col.row().label(text="Color")
    # edge numbers
    text_settings_left_col.row().prop(general_settings_props, "show_edge_numbers", toggle=1, text="Edge numbers")
    edge_number_size_row = text_settings_size_col.row()
    edge_number_size_row.active = general_settings_props.show_edge_numbers
    edge_number_size_row.prop(general_settings_props, "edge_number_font_size", text="")
    edge_number_color_row = text_settings_color_col.row()
    edge_number_color_row.active = general_settings_props.show_edge_numbers
    edge_number_color_row.prop(general_settings_props, "edge_number_color", text="")
    # step numbers
    text_settings_left_col.row().prop(general_settings_props, "show_step_numbers", toggle=1, text="Step numbers")
    step_number_size_row = text_settings_size_col.row()
    step_number_size_row.active = general_settings_props.show_step_numbers
    step_number_size_row.prop(general_settings_props, "build_steps_font_size", text="")
    step_number_color_row = text_settings_color_col.row()
    step_number_color_row.active = general_settings_props.show_step_numbers
    step_number_color_row.prop(general_settings_props, "steps_color", text="")
    if general_settings_props.show_step_numbers and not operator.build_steps_valid:
        step_num_warning_row = general_settings.row()
        step_num_warning_row.label(icon="ERROR", text="Sure about the step numbers?")

    # Line settings
    layout.label(text="Detailed line settings", icon="LINE_DATA")
    line_settings = layout.box()
    # line width
    write_custom_split_property_row(line_settings, "Line width (pt)", line_settings_props, "line_width", 0.6)
    # lines color
    write_custom_split_property_row(line_settings, "Cut edges color", line_settings_props, "lines_color", 0.6)
    write_custom_split_property_row(line_settings, "Convex folds color", line_settings_props, "convex_fold_edges_color", 0.6)
    write_custom_split_property_row(line_settings, "Concave folds color", line_settings_props, "concave_fold_edges_color", 0.6)
    # hide fold edge threshold
    write_custom_split_property_row(line_settings, "Fold edge threshold", line_settings_props, "hide_fold_edge_angle_th", 0.6)
    # edge number offset
    write_custom_split_property_row(line_settings, "Edge number offset", line_settings_props, "edge_number_offset", 0.6, general_settings_props.show_edge_numbers)
    # linestyles
    line_settings.separator(factor=0.2)
    line_settings.row().label(text="Choose linestyles of:")
    write_custom_split_property_row(line_settings, "Cut edges", line_settings_props, "cut_edge_ls", 0.6)
    write_custom_split_property_row(line_settings, "Convex fold edges", line_settings_props, "convex_fold_edge_ls", 0.6)
    write_custom_split_property_row(line_settings, "Concave fold edges", line_settings_props, "concave_fold_edge_ls", 0.6)
    write_custom_split_property_row(line_settings, "Glue flap edges", line_settings_props, "glue_flap_ls", 0.6)

    # Coloring / Texturing
    layout.label(text="Texture settings", icon="TEXTURE")
    glue_flap_color_settings = layout.box()
    glue_flap_color_row = glue_flap_color_settings.row().split(factor=0.6, align=True)
    glue_flap_col_1, glue_flap_col_2 = (glue_flap_color_row.column(), glue_flap_color_row.column())
    glue_flap_col_1.prop(texture_settings_props, "apply_glue_flap_color", toggle=1, text="Color glue flaps")
    glue_flap_col_2.prop(texture_settings_props, "glue_flap_color", text="")
    glue_flap_color_settings.active = texture_settings_props.apply_glue_flap_color

    texture_settings = layout.box()
    texture_row = texture_settings.row().column_flow(columns=2, align=True)
    show_textures_col, double_sided_col = (texture_row.column(align=True), texture_row.column(align=True))
    show_textures_col.prop(texture_settings_props, "apply_textures", toggle=1)
    double_sided_col.prop(texture_settings_props, "print_two_sided", toggle=1)
    double_sided_col.active = texture_settings_props.apply_textures
    write_custom_split_property_row(texture_settings, "Triangle bleed", texture_settings_props, "triangle_bleed", 0.6, texture_settings_props.apply_textures)

def page_layout_draw_func(operator : bpy.types.Operator):
    layout : bpy.types.UILayout = operator.layout
    # paper size
    write_custom_split_property_row(layout, "Paper size", operator.page_layout_options, "paper_size", 0.6)
    if operator.page_layout_options.paper_size == "Custom":
        page_size_box = layout.box()
        page_size_box.label(text="Custom page size")
        sizing_row = page_size_box.row()
        sizing_row.column().prop(operator.page_layout_options, "custom_page_width")
        sizing_row.column().prop(operator.page_layout_options, "custom_page_height")
    # model scale
    # scaling mode
    scaling_mode_row = layout.row().column_flow(columns=2, align=True)
    scaling_mode_row.column(align=True).prop_enum(operator.page_layout_options, "scaling_mode", "HEIGHT")
    scaling_mode_row.column(align=True).prop_enum(operator.page_layout_options, "scaling_mode", "SCALE")

    if operator.page_layout_options.scaling_mode == "HEIGHT":
        # target model height
        write_custom_split_property_row(layout, "Target height", operator.page_layout_options, "target_model_height", 0.6)
        curr_scale_factor = operator.page_layout_options.target_model_height / operator.mesh_height
        # set correct scaling
        operator.page_layout_options.sizing_scale = curr_scale_factor 
    elif operator.page_layout_options.scaling_mode == "SCALE":
        write_custom_split_property_row(layout, "Model scale", operator.page_layout_options, "sizing_scale", 0.6)
        curr_scale_factor = operator.page_layout_options.sizing_scale
        # set target model height
        operator.page_layout_options.target_model_height = curr_scale_factor * operator.mesh_height
    
    if operator.page_layout_options.sizing_scale > operator.max_fit_scaling:
        layout.row().label(icon="ERROR", text="A piece does not fit on one page!")

    # one mat per page
    write_custom_split_property_row(layout, "One material per page", operator.page_layout_options, "one_material_per_page", 0.6)
    # margin
    write_custom_split_property_row(layout, "Page margin", operator.page_layout_options, "page_margin", 0.6)
    # island spacing
    write_custom_split_property_row(layout, "Island spacing", operator.page_layout_options, "space_between_components", 0.6)

def create_exporter_for_operator(operator, output_format="pdf"):
    general_settings = operator.properties.general_settings
    line_settings = operator.properties.line_settings
    texture_settings = operator.properties.texture_settings

    export_settings = ExportSettings(
        paper_size=paper_sizes[general_settings.paper_size] if general_settings.paper_size != "Custom" else (units.blender_distance_to_cm(general_settings.custom_page_width), units.blender_distance_to_cm(general_settings.custom_page_height)), 
        line_width=line_settings.line_width,
        cut_edge_ls=line_settings.cut_edge_ls,
        convex_fold_edge_ls=line_settings.convex_fold_edge_ls,
        concave_fold_edge_ls=line_settings.concave_fold_edge_ls,
        glue_flap_ls=line_settings.glue_flap_ls,
        fold_hide_threshold_angle=line_settings.hide_fold_edge_angle_th,
        show_edge_numbers=general_settings.show_edge_numbers,
        edge_number_font_size=general_settings.edge_number_font_size,
        edge_number_offset=units.blender_distance_to_cm(line_settings.edge_number_offset),
        show_build_step_numbers=general_settings.show_step_numbers,
        apply_main_texture=texture_settings.apply_textures,
        prints_on_model_inside=general_settings.print_on_inside,
        two_sided_with_texture=texture_settings.print_two_sided,
        color_of_cut_edges=line_settings.lines_color,
        color_of_convex_fold_edges=line_settings.convex_fold_edges_color,
        color_of_concave_fold_edges = line_settings.concave_fold_edges_color,
        color_of_edge_numbers=general_settings.edge_number_color,
        color_of_build_steps=general_settings.steps_color,
        build_step_font_size=general_settings.build_steps_font_size,
        triangle_bleed=units.blender_distance_to_cm(texture_settings.triangle_bleed),
        color_glue_flaps=texture_settings.apply_glue_flap_color,
        color_of_glue_flaps=texture_settings.glue_flap_color
    )

    exporter = exporters.MatplotlibBasedExporter(output_format=output_format, export_settings=export_settings)
    return exporter

def compute_max_fit_scaling_factor(ao : bpy.types.Object, settings):
    all_component_bb_dims_cm = [units.blender_distance_to_cm(bb_dim) for bb_dim in printprepper.compute_all_connected_components_bb_dimensions(ao)]
    max_fit_scaling = np.inf
    if len(all_component_bb_dims_cm) > 0:
        page_margin_in_cm = units.blender_distance_to_cm(settings.page_margin)
        if settings.paper_size != "Custom":
            curr_page_size = exporters.paper_sizes[settings.paper_size]
        else:
            curr_page_size = (units.blender_distance_to_cm(settings.custom_page_width), units.blender_distance_to_cm(settings.custom_page_height))
        effective_page_dim_asc = sorted([curr_page_size[0] - 2 * page_margin_in_cm, curr_page_size[1] - 2 * page_margin_in_cm])
        for component_bb_dim in all_component_bb_dims_cm:
            bb_asc = sorted(component_bb_dim)
            max_fit_scaling = min(effective_page_dim_asc[0] / bb_asc[0], effective_page_dim_asc[1] / bb_asc[1], max_fit_scaling)
    return max_fit_scaling

#################################
#      Page Layout Editing      #
#################################

def read_custom_page_layout(obj : Object):
    mesh : Mesh = obj.data
    general_mesh_props = mesh.polyzamboni_general_mesh_props
    # collect all component print data
    component_print_data = create_print_data_for_all_components(obj, general_mesh_props.model_scale)

    # read and set correct page transforms
    page_transforms_per_component = io.read_page_transforms(mesh)
    current_component_print_data : ComponentPrintData
    for current_component_print_data in component_print_data:
        current_component_print_data.page_transform = page_transforms_per_component[current_component_print_data.og_component_id]

    # create page layout
    page_numbers_per_components = io.read_page_numbers(mesh)
    num_pages = max(page_numbers_per_components.values()) + 1 if len(page_numbers_per_components) > 0 else 0
    custom_components_on_pages = [[] for _ in range(num_pages)]
    for current_component_print_data in component_print_data:
        custom_components_on_pages[page_numbers_per_components[current_component_print_data.og_component_id]].append(current_component_print_data)
    return custom_components_on_pages

def compute_and_save_page_layout(obj : Object, scaling_factor, paper_size, page_margin, component_margin, separate_materials):
    if isinstance(paper_size, str):
        page_size = paper_sizes[paper_size]
    else:
        page_size = paper_size
    component_print_data = create_print_data_for_all_components(obj, scaling_factor)
    print_data_on_pages = fit_components_on_pages(component_print_data, page_size, page_margin, component_margin, separate_materials)
    # store computed layout
    mesh = obj.data
    general_props = mesh.polyzamboni_general_mesh_props
    general_props.model_scale = scaling_factor
    if isinstance(paper_size, str):
        general_props.paper_size = paper_size
    else:
        general_props.paper_size = "Custom"
        general_props.custom_page_width = page_size[0] / (100 * bpy.context.scene.unit_settings.scale_length)
        general_props.custom_page_height = page_size[1] / (100 * bpy.context.scene.unit_settings.scale_length)
    page_numbers_per_component = {}
    page_transforms_per_component = {}
    for page_index, component_print_data_on_page in enumerate(print_data_on_pages):
        current_print_data : ComponentPrintData
        for current_print_data in component_print_data_on_page:
            current_c_id = current_print_data.og_component_id
            page_numbers_per_component[current_c_id] = page_index
            page_transforms_per_component[current_c_id] = current_print_data.page_transform
    io.write_page_numbers(mesh, page_numbers_per_component)
    io.write_page_transforms(mesh, page_transforms_per_component)

class PageEditorState(Enum):
    SELECT_PIECES = 0
    MOVE_PIECE = 1
    ROTATE_PIECE = 2
    EDIT_BUILD_STEP_NUMBER = 3

def find_page_under_mouse_position(pos_x, pos_y, pages, paper_size, margin_between_pages = 1.0, pages_per_row = 2):
    row_grid_size = paper_size[0] + margin_between_pages
    row_index = math.floor(pos_x / row_grid_size)
    if pos_x < 0:
        row_index -= 1
    if row_index < 0 or row_index >= pages_per_row or pos_x - row_index * row_grid_size > paper_size[0]:
        return None
    transformed_pos_y = -(pos_y - paper_size[1])
    col_grid_size = paper_size[1] + margin_between_pages
    col_index = math.floor(transformed_pos_y / col_grid_size)
    if transformed_pos_y < 0:
        col_index -= 1
    if col_index < 0 or transformed_pos_y - col_index * col_grid_size > paper_size[1]:
        return None
    # compute page index
    hovered_page_index = pages_per_row * col_index + row_index
    if hovered_page_index <= pages:
        return hovered_page_index
    else:
        return None

def compute_page_anchor(page_index, pages_per_row, paper_size, margin_between_pages):
    row_index = page_index % pages_per_row
    col_index = page_index // pages_per_row
    return np.array([row_index * (paper_size[0] + margin_between_pages), -col_index * (paper_size[1] + margin_between_pages)], dtype=np.float64)

def find_papermodel_piece_under_mouse_position(pos_x, pos_y, print_data_on_pages, current_page_index, paper_size, pages_per_row = 2, margin_between_pages = 1, search_subset = None):
    if current_page_index is None or current_page_index >= len(print_data_on_pages):
        return None

    page_anchor = compute_page_anchor(current_page_index, pages_per_row, paper_size, margin_between_pages)
    current_print_data : ComponentPrintData
    for current_print_data in print_data_on_pages[current_page_index].values():
        if search_subset is not None and current_print_data.og_component_id not in search_subset:
            continue
        colored_triangle : ColoredTriangleData
        for colored_triangle in current_print_data.colored_triangles:
            if geometry.point_in_2d_triangle(np.array([pos_x, pos_y], dtype=np.float64) - page_anchor, *[current_print_data.page_transform * coord for coord in colored_triangle.coords]):
                return current_print_data.og_component_id
        glue_flap : GlueFlapData
        for glue_flap in current_print_data.glue_flaps:
            for glue_flap_triangle_coords in glue_flap.tris:
                if geometry.point_in_2d_triangle(np.array([pos_x, pos_y], dtype=np.float64) - page_anchor, *[current_print_data.page_transform * coord for coord in glue_flap_triangle_coords]):
                    return current_print_data.og_component_id
    return None

def find_page_under_papermodel_piece(component_print_data : ComponentPrintData, component_page, bonus_transform : geometry.AffineTransform2D, pages, paper_size, margin_between_pages = 1.0, pages_per_row = 2):
    component_cog = component_print_data.get_cog()
    image_space_cog = (bonus_transform @ component_print_data.page_transform) * component_cog + compute_page_anchor(component_page, pages_per_row, paper_size, margin_between_pages)
    return find_page_under_mouse_position(image_space_cog[0], image_space_cog[1], pages, paper_size, margin_between_pages, pages_per_row)

#################################
#         Build sections        #
#################################

def collect_all_selected_connected_components(bm : BMesh, face_ids_to_component_ids):
    res = set()
    for face in bm.faces:
        if not face.select:
            continue
        res.add(face_ids_to_component_ids[face.index])
    return res

def remove_component_set_from_all_existing_unlocked_build_sections(props, component_set : set):
    build_sections = props.build_sections
    for build_section in build_sections:
        if build_section.locked:
            continue
        components_in_section = set([component.id for component in build_section.connected_components])
        components_in_section.difference_update(component_set)
        io.overwrite_build_section(build_section, components_in_section)

def subtract_all_locked_build_sections_from_component_set(props, component_set : set, skip_index = -1):
    build_sections = props.build_sections
    for section_index, build_section in enumerate(build_sections):
        if section_index == skip_index:
            continue
        if not build_section.locked:
            continue
        components_in_section = set([component.id for component in build_section.connected_components])
        component_set.difference_update(components_in_section)

def create_build_section_from_selected_faces(mesh : Mesh, bm : BMesh, props):
    _, face_ids_to_component_ids = io.read_connected_components(mesh)
    components_in_new_section = collect_all_selected_connected_components(bm, face_ids_to_component_ids)
    remove_component_set_from_all_existing_unlocked_build_sections(props, components_in_new_section)
    subtract_all_locked_build_sections_from_component_set(props, components_in_new_section)
    io.write_new_build_section(mesh, utils.get_default_section_name_from_section_index(len(props.build_sections)), components_in_new_section)

def get_active_build_section_set(mesh : Mesh, props):
    active_section_index = props.active_build_section
    if active_section_index == -1 or active_section_index >= len(props.build_sections):
        return None
    return set([component.id for component in props.build_sections[active_section_index].connected_components])

def change_active_build_section_from_selected_faces(mesh : Mesh, bm : BMesh, props):
    active_section_index = props.active_build_section
    if active_section_index == -1 or active_section_index >= len(props.build_sections):
        return
    _, face_ids_to_component_ids = io.read_connected_components(mesh)
    selected_components = collect_all_selected_connected_components(bm, face_ids_to_component_ids)
    remove_component_set_from_all_existing_unlocked_build_sections(props, selected_components)
    subtract_all_locked_build_sections_from_component_set(props, selected_components, skip_index=active_section_index)
    io.overwrite_build_section(props.build_sections[active_section_index], selected_components)

def add_components_of_selected_faces_to_active_build_section(mesh : Mesh, bm : BMesh, props):
    active_section_index = props.active_build_section
    if active_section_index == -1 or active_section_index >= len(props.build_sections):
        return
    _, face_ids_to_component_ids = io.read_connected_components(mesh)
    section_to_components_dict, _ = io.read_build_sections(mesh)
    selected_components = collect_all_selected_connected_components(bm, face_ids_to_component_ids)
    selected_components.update(section_to_components_dict[active_section_index])
    remove_component_set_from_all_existing_unlocked_build_sections(props, selected_components)
    subtract_all_locked_build_sections_from_component_set(props, selected_components, skip_index=active_section_index)
    io.overwrite_build_section(props.build_sections[active_section_index], selected_components)

def remove_components_of_selected_faces_from_active_build_section(mesh : Mesh, bm : BMesh, props):
    active_section_index = props.active_build_section
    if active_section_index == -1 or active_section_index >= len(props.build_sections):
        return
    _, face_ids_to_component_ids = io.read_connected_components(mesh)
    section_to_components_dict, _ = io.read_build_sections(mesh)
    selected_components = collect_all_selected_connected_components(bm, face_ids_to_component_ids)
    remaining_components = section_to_components_dict[active_section_index].difference(selected_components)
    io.overwrite_build_section(props.build_sections[active_section_index], remaining_components)

def remove_active_build_section(mesh : Mesh):
    active_section_index = mesh.polyzamboni_general_mesh_props.active_build_section
    if active_section_index == -1:
        return
    if active_section_index >= len(mesh.polyzamboni_general_mesh_props.build_sections):
        return
    mesh.polyzamboni_general_mesh_props.build_sections.remove(active_section_index)
    mesh.polyzamboni_general_mesh_props.active_build_section = active_section_index - 1

def get_face_ids_in_active_build_section(mesh, props):
    active_section_index = props.active_build_section
    if active_section_index == -1:
        return
    if active_section_index >= len(props.build_sections):
        return
    components_as_sets, _ = io.read_connected_components(mesh)
    res = set()
    active_section = props.build_sections[active_section_index]
    for component in active_section.connected_components:
        res.update(components_as_sets[component.id])
    return res