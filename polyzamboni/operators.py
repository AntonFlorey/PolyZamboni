import bpy
import bmesh
import numpy as np
import os
import threading
from bpy.props import StringProperty, PointerProperty
from bpy_extras.io_utils import ExportHelper
from .properties import GeneralExportSettings, LineExportSettings, TextureExportSettings, ZamboniGeneralMeshProps, PageLayoutCreationSettings
from .drawing import *
from . import exporters
from . import printprepper
from .geometry import compute_planarity_score
from .autozamboni import greedy_auto_cuts
from . import printprepper
from . import operators_backend
from . import utils
from .zambonipolice import check_if_build_step_numbers_exist_and_make_sense, all_components_have_unfoldings, check_if_page_numbers_and_transforms_exist_for_all_components

def _active_object_is_mesh(context : bpy.types.Context):
    active_object = context.active_object
    is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())
    return is_mesh

def _active_object_is_mesh_with_paper_model(context : bpy.types.Context):
    if not _active_object_is_mesh(context):
        return False
    active_mesh = context.active_object.data
    mesh_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
    return mesh_props.has_attached_paper_model

class InitializeCuttingOperator(bpy.types.Operator):
    """Start the unfolding process for this mesh"""
    bl_label = "Unfold this mesh"
    bl_idname  = "polyzamboni.cut_initialization_op"

    weird_mode_table = {
        "PAINT_VERTEX" : "VERTEX_PAINT",
        "EDIT_MESH" : "EDIT",
        "PAINT_WEIGHT" : "WEIGHT_PAINT",
        "PAINT_TEXTURE" : "TEXTURE_PAINT"
    }   

    def invoke(self, context, event):
        returnto=False
        if(context.mode != 'OBJECT'):
            returnto=context.mode
            bpy.ops.object.mode_set(mode="OBJECT")
        ao = bpy.context.active_object
        active_mesh = ao.data
        mesh_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
        bm : bmesh.types.BMesh = bmesh.new()
        bm.from_mesh(ao.data)
        if returnto:
            bpy.ops.object.mode_set(mode=self.weird_mode_table[returnto] if returnto in self.weird_mode_table else returnto)
        self.selected_mesh_is_manifold = np.all([edge.is_manifold or edge.is_boundary for edge in bm.edges] + [v.is_manifold for v in bm.verts])
        self.normals_are_okay = np.all([edge.is_contiguous or edge.is_boundary for edge in bm.edges])
        self.double_connected_face_pair_present = False
        self.non_triangulatable_faces_present = False
        self.max_planarity_score = max([compute_planarity_score([np.array(v.co) for v in face.verts]) for face in bm.faces])

        if not self.selected_mesh_is_manifold:
            wm = context.window_manager
            bm.free()
            return wm.invoke_props_dialog(self, title="Something went wrong D:", confirm_text="Okay")
        
        if not self.normals_are_okay:
            wm = context.window_manager
            bm.free()
            return wm.invoke_props_dialog(self, title="Something went wrong D:", confirm_text="Okay")

        self.double_connected_face_pair_present = len(operators_backend.get_indices_of_multi_touching_faces(bm)) > 0
        mesh_props.multi_touching_faces_present = self.double_connected_face_pair_present
        if self.double_connected_face_pair_present:
            wm = context.window_manager
            bm.free()
            return wm.invoke_props_dialog(self, title="Something went wrong D:", confirm_text="Okay")

        # check all face triangulations
        self.non_triangulatable_faces_present = len(operators_backend.get_indices_of_not_triangulatable_faces(bm)) > 0
        mesh_props.faces_which_cant_be_triangulated_are_present = self.non_triangulatable_faces_present
        if self.non_triangulatable_faces_present:
            wm = context.window_manager
            bm.free()
            return wm.invoke_props_dialog(self, title="Something went wrong D:", confirm_text="Okay")

        if self.max_planarity_score > 0.1:
            wm = context.window_manager
            bm.free()
            return wm.invoke_props_dialog(self, title="Warning!", confirm_text="Okay")

        bm.free()
        return self.execute(context)
    
    def draw(self, context):
        layout = self.layout
        if not self.selected_mesh_is_manifold:
            layout.row().label(text="The selected mesh is not manifold!", icon="ERROR")
        if not self.normals_are_okay:
            layout.row().label(text="Bad normals! Try \"Recalculate Outside\" (Shift-N)", icon="ERROR")
        if self.double_connected_face_pair_present:
            layout.row().label(text="Some faces touch at more than one edge!", icon="ERROR")
        if self.non_triangulatable_faces_present:
            layout.row().label(text="Some faces cant be triangulated by PolyZamboni!", icon="ERROR")
        if self.max_planarity_score > 0.1:
            layout.row().label(text="Some faces are highly non-planar! (err: {:.2f})".format(self.max_planarity_score))
            layout.row().label(text="This might crash the addon later...")

    def execute(self, context):
        if not self.selected_mesh_is_manifold or self.double_connected_face_pair_present or not self.normals_are_okay or self.non_triangulatable_faces_present:
            return { 'FINISHED' }
        # get the currently selected object
        returnto=False
        if(context.mode != 'OBJECT'):
            returnto=context.mode
            bpy.ops.object.mode_set(mode="OBJECT")
        ao = bpy.context.active_object
        active_mesh = ao.data
        if returnto:
            bpy.ops.object.mode_set(mode=self.weird_mode_table[returnto] if returnto in self.weird_mode_table else returnto)
        operators_backend.initialize_paper_model(active_mesh)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }

    @classmethod
    def poll(cls, context : bpy.types.Context):
        return _active_object_is_mesh(context)

class SelectMultiTouchingFacesOperator(bpy.types.Operator):
    """Select all face pairs that touch at more than one edge"""
    bl_label = "Select ambiguous faces"
    bl_description = "Select all faces that share more than one edge with another face"
    bl_idname  = "polyzamboni.multi_touching_face_selection_op"

    def execute(self, context):
        bpy.ops.object.mode_set(mode="EDIT")
        ao = bpy.context.active_object
        active_mesh = ao.data
        mesh_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
        bm : bmesh.types.BMesh = bmesh.from_edit_mesh(active_mesh)

        face_indices_to_select = operators_backend.get_indices_of_multi_touching_faces(bm)

        if len(face_indices_to_select) == 0:
            mesh_props.multi_touching_faces_present = False

        # deselect all faces
        for face in bm.faces:
            face.select_set(False)
        bm.faces.ensure_lookup_table()
        for face_index in face_indices_to_select:
            bm.faces[face_index].select_set(True)

        bmesh.update_edit_mesh(ao.data)
        return {'FINISHED'}

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh(context)

class SelectNonTriangulatableFacesOperator(bpy.types.Operator):
    """Select all faces that can not be triangulated by PolyZamboni"""
    bl_label = "Select non triangulatable faces"
    bl_description = "Select all faces that can not be triangulated by PolyZamboni"
    bl_idname  = "polyzamboni.no_tri_face_selection_op"

    def execute(self, context):
        bpy.ops.object.mode_set(mode="EDIT")
        ao = bpy.context.active_object
        active_mesh = ao.data
        mesh_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
        bm : bmesh.types.BMesh = bmesh.from_edit_mesh(active_mesh)
        
        face_indices_to_select = operators_backend.get_indices_of_not_triangulatable_faces(bm)

        if len(face_indices_to_select) == 0:
            mesh_props.faces_which_cant_be_triangulated_are_present = False

        # deselect all faces
        for face in bm.faces:
            face.select_set(False)
        bm.faces.ensure_lookup_table()
        for face_index in face_indices_to_select:
            bm.faces[face_index].select_set(True)

        bmesh.update_edit_mesh(ao.data)
        return {'FINISHED'}

    @classmethod
    def poll(cls, context : bpy.types.Context):
        return _active_object_is_mesh(context)

class RemoveAllPolyzamboniDataOperator(bpy.types.Operator):
    """Remove all attached Polyzamboni Data"""
    bl_label = "Remove Paper Model"
    bl_idname = "polyzamboni.remove_all_op"

    def execute(self, context):
        operators_backend.delete_paper_model(context.active_object.data)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event, message="Sure? This will delete all cuts.", confirm_text="Delete")
    
    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh

class SyncMeshOperator(bpy.types.Operator):
    """Transfer mesh changes to the cutgraph. Topology changes will likely break everything!"""
    bl_label = "Sync Mesh Changes"
    bl_idname = "polyzamboni.mesh_sync_op"

    weird_mode_table = {
        "PAINT_VERTEX" : "VERTEX_PAINT",
        "EDIT_MESH" : "EDIT",
        "PAINT_WEIGHT" : "WEIGHT_PAINT",
        "PAINT_TEXTURE" : "TEXTURE_PAINT"
    }

    def execute(self, context):
        returnto=False
        if(context.mode != 'OBJECT'):
            returnto=context.mode
            bpy.ops.object.mode_set(mode="OBJECT")

        operators_backend.sync_paper_model_with_mesh_geometry(context.active_object.data)

        if returnto:
            bpy.ops.object.mode_set(mode=self.weird_mode_table[returnto] if returnto in self.weird_mode_table else returnto)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class SeparateAllMaterialsOperator(bpy.types.Operator):
    """ Adds cuts to all edges between faces with a different material. """
    bl_label = "Separate Materials"
    bl_idname = "polyzamboni.material_separation_op"

    def execute(self, context):
        operators_backend.add_cuts_between_different_materials(bpy.context.active_object.data)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class RemoveAllAutoCutsOperator(bpy.types.Operator):
    """ Removes all auto cuts from the selected paper model. """
    bl_label = "Remove Auto Cuts"
    bl_idname = "polyzamboni.auto_cuts_removal_op"
    
    def execute(self, context):
        operators_backend.remove_auto_cuts(bpy.context.active_object.data)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class RecomputeFlapsOperator(bpy.types.Operator):
    """ Applies the current flap settings and recomputes all glue flaps """
    bl_label = "Recompute Flaps"
    bl_idname = "polyzamboni.flaps_recompute_op"

    def execute(self, context : bpy.types.Context):
        operators_backend.recompute_all_glue_flaps(context.active_object.data)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class FlipGlueFlapsOperator(bpy.types.Operator):
    """ Flip all glue flaps attached to the selected edges """
    bl_label = "Flip Glue Flaps"
    bl_idname = "polyzamboni.flip_flap_op"

    def execute(self, context):
        ao = context.active_object
        ao_mesh = ao.data
        ao_bmesh = bmesh.from_edit_mesh(ao_mesh)
        selected_edges = [e.index for e in ao_bmesh.edges if e.select] 
        operators_backend.flip_glue_flaps(ao_mesh, selected_edges)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context) and context.mode == 'EDIT_MESH'

class ComputeBuildStepsOperator(bpy.types.Operator):
    """ Starting at the selected face, compute a polyzamboni build order of the current mesh """
    bl_label = "Compute Build Order"
    bl_idname = "polyzamboni.build_order_op"

    def execute(self, context):
        ao = context.active_object
        ao_mesh = ao.data
        ao_bmesh = bmesh.from_edit_mesh(ao_mesh)
        selected_faces = [f.index for f in ao_bmesh.faces if f.select]
        operators_backend.compute_build_step_numbers(ao_mesh, selected_faces)
        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context) and context.mode == 'EDIT_MESH'

class ZamboniGlueFlapDesignOperator(bpy.types.Operator):
    """ Control the placement of glue flaps """
    bl_label = "PolyZamboni Glue Flap Design Tool"
    bl_idname = "polyzamboni.glue_flap_editing_operator"

    design_actions: bpy.props.EnumProperty(
        name="actions",
        description="Select an action",
        items=[
            ("FLIP_FLAPS", "Flip Glue Flaps", "Flip all glue flaps attached to the selected edges", "AREA_SWAP", 0),
            ("RECOMPUTE_FLAPS", "Recompute all Glue Flaps", "Automatically place glue flaps at all cut edges", "FILE_SCRIPT", 1)
        ]
    )

    def execute(self, context):

        if self.design_actions == "FLIP_FLAPS":
            bpy.ops.polyzamboni.flip_flap_op()
            return {"FINISHED"}
        elif self.design_actions == "RECOMPUTE_FLAPS":
            bpy.ops.polyzamboni.flaps_recompute_op()
        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context) and context.mode == 'EDIT_MESH'

class ZamboniCutDesignOperator(bpy.types.Operator):
    """ Add or remove cuts """
    bl_label = "PolyZamboni Cut Design Tool"
    bl_idname  = "polyzamboni.cut_editing_operator"
    bl_options = {'UNDO'}

    design_actions: bpy.props.EnumProperty(
        name="actions",
        description="Select an action",
        items=[
            ("ADD_CUT", "Add Cuts", "Cut the selected edges", "UNLINKED", 0),
            ("GLUE_EDGE", "Glue Edges", "Prevent selected edges from being cut", "LOCKED", 1),
            ("RESET_EDGE", "Clear Edges", "Remove any constraints", "BRUSH_DATA", 2),
            ("REGION_CUTOUT", "Define Region", "Mark the selected faces as one region", "OUTLINER_DATA_SURFACE", 3),
            #("FLIP_FLAPS", "Flip Glue Flaps", "Flip all glue flaps attached to the selected edges", "AREA_SWAP", 4)
        ],
        default="ADD_CUT"
    )

    def execute(self, context):
        ao = context.active_object
        ao_mesh = ao.data
        ao_bmesh = bmesh.from_edit_mesh(ao_mesh)
        selected_edges = [e.index for e in ao_bmesh.edges if e.select]
        ao_bmesh.free()

        if self.design_actions == "ADD_CUT":
            operators_backend.cut_edges(ao_mesh, selected_edges)
        elif self.design_actions == "GLUE_EDGE":
            operators_backend.glue_edges(ao_mesh, selected_edges)
        elif self.design_actions == "RESET_EDGE":
            operators_backend.clear_edges(ao_mesh, selected_edges)
        elif self.design_actions == "REGION_CUTOUT":
            selected_faces = [f.index for f in ao_bmesh.faces if f.select]
            operators_backend.add_cutout_region(ao_mesh, selected_faces)
        update_all_polyzamboni_drawings(None, context)

        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context) and context.mode == 'EDIT_MESH'

class AutoCutsOperator(bpy.types.Operator):
    """ Automatically generate cuts """
    bl_label = "Auto Unfold"
    bl_idname  = "polyzamboni.auto_cuts_op"

    cutting_algorithm : bpy.props.EnumProperty(
        name="Algorithm",
        description="The algorithm used to generate auto-cuts",
        items=[
            ("GREEDY", "Greedy", "Start with all edges cut and then remove as many as possible", "", 0)
        ],
        default="GREEDY"
    )
    quality_level : bpy.props.EnumProperty(
        name="Quality",
        description="Determine what kind of print overlaps are allowed",
        items=[
             ("NO_OVERLAPS_ALLOWED", "No overlaps", "Allow no overlaps (all cut pieces have to be green)", "", 0),
             ("GLUE_FLAP_OVERLAPS_ALLOWED", "Allow overlapping glue flaps", "Allow glue flap overlaps (all cut pieces have to be yellow or green)", "", 1),
             ("ALL_OVERLAPS_ALLOWED", "Allow all overlaps", "Allow all overlaps (all cut pieces have to be not red)", "", 2)
        ],
        default="NO_OVERLAPS_ALLOWED"
    )
    loop_alignment : bpy.props.EnumProperty(
        name="Loop Alignment",
        description="Determine what edges should have a high cut priority, depending on their alignment with one coordinate axis",
        items=[
            ("X", "X axis", "Loops around the x-axis", "", 0),
            ("Y", "Y axis", "Loops around the y-axis", "", 1),
            ("Z", "Z axis", "Loops around the z-axis", "", 2),
        ],
        default="Z"
    )
    max_pieces_per_component : bpy.props.IntProperty(
        name="Max pieces per component",
        description="The maximum amount of mesh faces per component. High values can lead to very high runtimes!",
        default=10,
        min=1
    )

    _timer = None
    _running = False

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self, title="Auto cut options", confirm_text="Lets go!")

    def draw(self, context):
        layout = self.layout
        write_custom_split_property_row(layout.row(), "Quality", self.properties, "quality_level", 0.5)
        write_custom_split_property_row(layout.row(), "Loops", self.properties, "loop_alignment", 0.5)
        write_custom_split_property_row(layout.row(), "Pieces per component", self.properties, "max_pieces_per_component", 0.5)

    def execute(self, context):
        ao = context.active_object
        ao_mesh = ao.data
        # progress bar setup
        self._running = True
        wm = context.window_manager
        wm.polyzamboni_auto_cuts_progress = 0.0
        wm.polyzamboni_auto_cuts_running = True
        self._timer = wm.event_timer_add(time_step=0.1, window=context.window)

        if self.cutting_algorithm == "GREEDY":
            def compute_and_report_progress():
                for progress in greedy_auto_cuts(ao_mesh, self.quality_level, self.loop_alignment, self.max_pieces_per_component):
                    wm.polyzamboni_auto_cuts_progress = progress
                self._running = False
                wm.polyzamboni_auto_cuts_running = False
            threading.Thread(target=compute_and_report_progress).start()

        wm.modal_handler_add(self)
        return { "RUNNING_MODAL" }

    def modal(self, context, event):
        if event.type == 'TIMER':
            for region in context.area.regions:
                if region.active_panel_category == 'PolyZamboni':
                    region.tag_redraw()
            if not self._running:
                update_all_polyzamboni_drawings(None, context)
                return {'FINISHED'}
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        self._running = False
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()

    @classmethod
    def poll(cls, context : bpy.types.Context):
        return _active_object_is_mesh_with_paper_model(context) and context.mode == 'EDIT_MESH'

class ZamboniCutEditingPieMenu(bpy.types.Menu):
    """This is a custom pie menu for all Zamboni cut design operators"""
    bl_label = "Polyzamboni Cut Tools"
    bl_idname = "POLYZAMBONI_MT_CUT_EDITING_PIE_MENU"

    def draw(self, context): 
        layout = self.layout 
        pie = layout.menu_pie() 
        pie.operator_enum("polyzamboni.cut_editing_operator", "design_actions")

class ZamboniGLueFlapEditingPieMenu(bpy.types.Menu):
    """This is a custom pie menu for all Zamboni glue flap design operators"""
    bl_label = "Polyzamboni Glue Flap Tools"
    bl_idname = "POLYZAMBONI_MT_GLUE_FLAP_EDITING_PIE_MENU"

    def draw(self, context): 
        layout = self.layout 
        pie = layout.menu_pie() 
        pie.operator_enum("polyzamboni.glue_flap_editing_operator", "design_actions")

def ext_list_to_filter_str(ext_list):
    res = ""
    for ext in ext_list[:-1]:
        res += "*." + ext + ","
    res += "*." + ext_list[-1]

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

def export_draw_func(operator):
    layout = operator.layout
    
    general_settings_props : GeneralExportSettings = operator.properties.general_settings
    line_settings_props : LineExportSettings = operator.properties.line_settings
    texture_settings_props : TextureExportSettings = operator.properties.texture_settings

    # General settings
    layout.label(text="General print settings", icon="TOOL_SETTINGS")
    general_settings = layout.box()
    # paper size
    write_custom_split_property_row(general_settings, "Paper size", general_settings_props, "paper_size", 0.6)

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
    curr_page_size = exporters.paper_sizes[general_settings_props.paper_size]

    bu_to_cm_scale_factor =  100 * general_settings_props.target_model_height * operator.curr_len_scale / operator.mesh_height
    page_margin_in_cm = 100 * operator.curr_len_scale * general_settings_props.page_margin
    if operator.max_comp_with * bu_to_cm_scale_factor > curr_page_size[0] - 2 * page_margin_in_cm or operator.max_comp_height * bu_to_cm_scale_factor > curr_page_size[1] - 2 * page_margin_in_cm:
        general_settings.row().label(icon="ERROR", text="A piece does not fit on one page!")
    # margin
    write_custom_split_property_row(general_settings, "Page margin", general_settings_props, "page_margin", 0.6)
    # island spacing
    write_custom_split_property_row(general_settings, "Island spacing", general_settings_props, "space_between_components", 0.6)
    # side of prints
    write_custom_split_property_row(general_settings, "Prints inside", general_settings_props, "print_on_inside", 0.6)
    # one mat per page
    write_custom_split_property_row(general_settings, "One material per page", general_settings_props, "one_material_per_page", 0.6)
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
        step_num_warning_row.label(icon="ERROR", text="Invalid build step numbers!")

    # Line settings
    layout.label(text="Detailed line settings", icon="LINE_DATA")
    line_settings = layout.box()
    # line width
    write_custom_split_property_row(line_settings, "Line width (pt)", line_settings_props, "line_width", 0.6)
    # lines color
    write_custom_split_property_row(line_settings, "Line color", line_settings_props, "lines_color", 0.6)
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
    texture_settings = layout.box()
    texture_row = texture_settings.row().column_flow(columns=2, align=True)
    show_textures_col, double_sided_col = (texture_row.column(align=True), texture_row.column(align=True))
    show_textures_col.prop(texture_settings_props, "apply_textures", toggle=1)
    double_sided_col.prop(texture_settings_props, "print_two_sided", toggle=1)
    double_sided_col.active = texture_settings_props.apply_textures

def create_exporter_for_operator(operator, output_format="pdf"):
    general_settings = operator.properties.general_settings
    line_settings = operator.properties.line_settings
    texture_settings = operator.properties.texture_settings
    exporter = exporters.MatplotlibBasedExporter(output_format=output_format, 
                                                 paper_size=general_settings.paper_size, 
                                                 line_width=line_settings.line_width,
                                                 cut_edge_ls=line_settings.cut_edge_ls,
                                                 convex_fold_edge_ls=line_settings.convex_fold_edge_ls,
                                                 concave_fold_edge_ls=line_settings.concave_fold_edge_ls,
                                                 glue_flap_ls=line_settings.glue_flap_ls,
                                                 fold_hide_threshold_angle=line_settings.hide_fold_edge_angle_th,
                                                 show_edge_numbers=general_settings.show_edge_numbers,
                                                 edge_number_font_size=general_settings.edge_number_font_size,
                                                 edge_number_offset=100 * operator.curr_len_scale * line_settings.edge_number_offset,
                                                 show_build_step_numbers=general_settings.show_step_numbers,
                                                 apply_main_texture=texture_settings.apply_textures,
                                                 print_on_inside=general_settings.print_on_inside,
                                                 two_sided_w_texture=texture_settings.print_two_sided,
                                                 color_of_lines=line_settings.lines_color,
                                                 color_of_edge_numbers=general_settings.edge_number_color,
                                                 color_of_build_steps=general_settings.steps_color,
                                                 build_step_font_size=general_settings.build_steps_font_size)
    return exporter

class PolyZamboniExportPDFOperator(bpy.types.Operator, ExportHelper):
    """Export Unfolding of active object as pdf"""
    bl_label = "Export PDF"
    bl_idname = "polyzamboni.export_operator_pdf"

    # Export Helper settings
    filename_ext = ".pdf"

    filter_glob: StringProperty(
        default="*.pdf",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # Polyzamboni export settings
    general_settings : PointerProperty(type=GeneralExportSettings)
    line_settings : PointerProperty(type=LineExportSettings)
    texture_settings : PointerProperty(type=TextureExportSettings)

    def invoke(self, context, event):
        # do stuff
        ao = context.active_object
        active_mesh = ao.data
        self.build_steps_valid = check_if_build_step_numbers_exist_and_make_sense(active_mesh)
        self.max_comp_with, self.max_comp_height = printprepper.compute_max_print_component_dimensions(ao)
        self.mesh_height = utils.compute_mesh_height(active_mesh)
        if self.mesh_height == 0:
            print("POLYZAMBONI WARNING: Mesh has zero height!")
            self.mesh_height = 1 # to prevent crashes
        self.curr_len_unit = context.scene.unit_settings.length_unit
        self.curr_unit_system = context.scene.unit_settings.system
        self.curr_len_scale = context.scene.unit_settings.scale_length
        
        #compute maximum target mesh height
        page_margin_in_cm = 100 * self.curr_len_scale * self.general_settings.page_margin
        max_w_in_cm = 100 * self.curr_len_scale * self.max_comp_with
        max_h_in_cm = 100 * self.curr_len_scale * self.max_comp_height
        curr_page_size = exporters.paper_sizes[self.general_settings.paper_size]
        max_scaling = min((curr_page_size[0] - 2 * page_margin_in_cm) / max_w_in_cm, (curr_page_size[1] - 2 * page_margin_in_cm) / max_h_in_cm)
        self.general_settings.sizing_scale = 0.9 * max_scaling
        self.general_settings.target_model_height = 0.9 * max_scaling * self.mesh_height

        return super().invoke(context, event)

    def draw(self, context):
        export_draw_func(self)

    def execute(self, context):
        # first, check if the selected model can be unfolded
        ao = context.active_object
        ao_mesh = ao.data
        if not all_components_have_unfoldings(ao_mesh):
            print("POLYZAMBONI WARNING: You exported a mesh that is not fully foldable yet!")

        # prepare everything
        component_print_info = printprepper.create_print_data_for_all_components(ao, printprepper.compute_scaling_factor_for_target_model_height(ao_mesh, 100 * self.curr_len_scale * self.general_settings.target_model_height))
        page_arrangement = printprepper.fit_components_on_pages(component_print_info,
                                                                exporters.paper_sizes[self.general_settings.paper_size], 
                                                                100 * self.curr_len_scale * self.general_settings.page_margin, 
                                                                100 * self.curr_len_scale * self.general_settings.space_between_components, 
                                                                self.general_settings.one_material_per_page)

        # initialize exporter
        pdf_exporter = create_exporter_for_operator(self, "pdf")
        filename, extension = os.path.splitext(self.filepath)
        
        # export file
        pdf_exporter.export(page_arrangement, filename)

        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class PolyZamboniExportSVGOperator(bpy.types.Operator, ExportHelper):
    """Export Unfolding of active object as svg"""
    bl_label = "Export SVG"
    bl_idname = "polyzamboni.export_operator_svg"

    # Export Helper settings
    filename_ext = ".svg"

    filter_glob: StringProperty(
        default="*.svg",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # Polyzamboni export settings
    general_settings : PointerProperty(type=GeneralExportSettings)
    line_settings : PointerProperty(type=LineExportSettings)
    texture_settings : PointerProperty(type=TextureExportSettings)

    def invoke(self, context, event):
        # do stuff
        ao = context.active_object
        active_mesh = ao.data
        self.build_steps_valid = check_if_build_step_numbers_exist_and_make_sense(active_mesh)
        self.max_comp_with, self.max_comp_height = printprepper.compute_max_print_component_dimensions(ao)
        self.mesh_height = utils.compute_mesh_height(active_mesh)
        if self.mesh_height == 0:
            print("POLYZAMBONI WARNING: Mesh has zero height!")
            self.mesh_height = 1 # to prevent crashes
        self.curr_len_unit = context.scene.unit_settings.length_unit
        self.curr_unit_system = context.scene.unit_settings.system
        self.curr_len_scale = context.scene.unit_settings.scale_length

        #compute maximum target mesh height
        page_margin_in_cm = 100 * self.curr_len_scale * self.general_settings.page_margin
        max_w_in_cm = 100 * self.curr_len_scale * self.max_comp_with
        max_h_in_cm = 100 * self.curr_len_scale * self.max_comp_height
        curr_page_size = exporters.paper_sizes[self.general_settings.paper_size]
        max_scaling = min((curr_page_size[0] - 2 * page_margin_in_cm) / max_w_in_cm, (curr_page_size[1] - 2 * page_margin_in_cm) / max_h_in_cm)
        self.general_settings.sizing_scale = 0.9 * max_scaling
        self.general_settings.target_model_height = 0.9 * max_scaling * self.mesh_height

        return super().invoke(context, event)

    def draw(self, context):
        export_draw_func(self)

    def execute(self, context):
        # first, check if the selected model can be unfolded
        ao = context.active_object
        ao_mesh = ao.data
        if not all_components_have_unfoldings(ao_mesh):
            print("POLYZAMBONI WARNING: You exported a mesh that is not fully foldable yet!")

        # prepare everything
        component_print_info = printprepper.create_print_data_for_all_components(ao, printprepper.compute_scaling_factor_for_target_model_height(ao_mesh, 100 * self.curr_len_scale * self.general_settings.target_model_height))
        page_arrangement = printprepper.fit_components_on_pages(component_print_info,
                                                                exporters.paper_sizes[self.general_settings.paper_size], 
                                                                100 * self.curr_len_scale * self.general_settings.page_margin, 
                                                                100 * self.curr_len_scale * self.general_settings.space_between_components, 
                                                                self.general_settings.one_material_per_page)

        # initialize exporter
        svg_exporter = create_exporter_for_operator(self, "svg")
        filename, extension = os.path.splitext(self.filepath)
        
        # export file
        svg_exporter.export(page_arrangement, filename)

        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class PolyZamboniPageLayoutOperator(bpy.types.Operator):
    """ Creates the final print layout of the papermodel instructions """
    bl_label = "Create print preview"
    bl_idname  = "polyzamboni.page_layout_op"

    page_layout_options : PointerProperty(type=PageLayoutCreationSettings)

    def invoke(self, context, event):
        # do stuff
        ao = context.active_object
        active_mesh = ao.data
        self.max_comp_with, self.max_comp_height = printprepper.compute_max_print_component_dimensions(ao)
        self.mesh_height = utils.compute_mesh_height(active_mesh)
        if self.mesh_height == 0:
            print("POLYZAMBONI WARNING: Mesh has zero height!")
            self.mesh_height = 1 # to prevent crashes
        self.curr_len_unit = context.scene.unit_settings.length_unit
        self.curr_unit_system = context.scene.unit_settings.system
        self.curr_len_scale = context.scene.unit_settings.scale_length
        
        #compute maximum target mesh height
        page_margin_in_cm = 100 * self.curr_len_scale * self.page_layout_options.page_margin
        max_w_in_cm = 100 * self.curr_len_scale * self.max_comp_with
        max_h_in_cm = 100 * self.curr_len_scale * self.max_comp_height
        curr_page_size = exporters.paper_sizes[self.page_layout_options.paper_size]
        max_scaling = min((curr_page_size[0] - 2 * page_margin_in_cm) / max_w_in_cm, (curr_page_size[1] - 2 * page_margin_in_cm) / max_h_in_cm)
        self.page_layout_options.sizing_scale = 0.9 * max_scaling
        self.page_layout_options.target_model_height = 0.9 * max_scaling * self.mesh_height

        wm = context.window_manager
        return wm.invoke_props_dialog(self, title="Page Layout Options")

    def draw(self, context):
        layout : bpy.types.UILayout = self.layout
        # paper size
        write_custom_split_property_row(layout, "Paper size", self.page_layout_options, "paper_size", 0.6)
        # model scale
        # scaling mode
        scaling_mode_row = layout.row().column_flow(columns=2, align=True)
        scaling_mode_row.column(align=True).prop_enum(self.page_layout_options, "scaling_mode", "HEIGHT")
        scaling_mode_row.column(align=True).prop_enum(self.page_layout_options, "scaling_mode", "SCALE")

        if self.page_layout_options.scaling_mode == "HEIGHT":
            # target model height
            write_custom_split_property_row(layout, "Target height", self.page_layout_options, "target_model_height", 0.6)
            curr_scale_factor = self.page_layout_options.target_model_height / self.mesh_height
            # set correct scaling
            self.page_layout_options.sizing_scale = curr_scale_factor 
        elif self.page_layout_options.scaling_mode == "SCALE":
            write_custom_split_property_row(layout, "Model scale", self.page_layout_options, "sizing_scale", 0.6)
            curr_scale_factor = self.page_layout_options.sizing_scale
            # set target model height
            self.page_layout_options.target_model_height = curr_scale_factor * self.mesh_height

        bu_to_cm_scale_factor =  100 * self.page_layout_options.target_model_height * self.curr_len_scale / self.mesh_height
        page_margin_in_cm = 100 * self.curr_len_scale * self.page_layout_options.page_margin
        curr_page_size = exporters.paper_sizes[self.page_layout_options.paper_size]
        if self.max_comp_with * bu_to_cm_scale_factor > curr_page_size[0] - 2 * page_margin_in_cm or self.max_comp_height * bu_to_cm_scale_factor > curr_page_size[1] - 2 * page_margin_in_cm:
            layout.row().label(icon="ERROR", text="A piece does not fit on one page!")

        # one mat per page
        write_custom_split_property_row(layout, "One material per page", self.page_layout_options, "one_material_per_page", 0.6)
        # margin
        write_custom_split_property_row(layout, "Page margin", self.page_layout_options, "page_margin", 0.6)
        # island spacing
        write_custom_split_property_row(layout, "Island spacing", self.page_layout_options, "space_between_components", 0.6)

    def execute(self, context):
        ao = context.active_object

        my_options : PageLayoutCreationSettings = self.page_layout_options
        scaling_factor = printprepper.compute_scaling_factor_for_target_model_height(ao.data, 100 * self.curr_len_scale * self.page_layout_options.target_model_height)

        operators_backend.compute_and_save_page_layout(ao, scaling_factor, 
                                                       my_options.paper_size, 
                                                       100 * self.curr_len_scale * my_options.page_margin, 
                                                       100 * self.curr_len_scale * my_options.space_between_components, 
                                                       my_options.one_material_per_page)

        # trigger a redraw
        update_all_page_layout_drawings(None, context)

        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

class PolyZamboniPageLayoutEditingOperator(bpy.types.Operator):
    """ Select, move and rotate your papermodel pieces """
    bl_label = "Edit Page Layout"
    bl_idname  = "polyzamboni.page_layout_editing_op"

    def invoke(self, context, event):
        ao = context.active_object
        mesh = ao.data
        if not check_if_page_numbers_and_transforms_exist_for_all_components(mesh):
            print("cancelled page editing operator")
            return { 'CANCELLED' }
        self.general_mesh_props = mesh.polyzamboni_general_mesh_props

        self.editing_state = operators_backend.PageEditorState.SELECT_PIECES
        self.active_page = None
        self.selected_component_id = None
        self.paper_size = paper_sizes[self.general_mesh_props.paper_size]
        self.fold_angle_th = context.scene.polyzamboni_drawing_settings.hide_fold_edge_angle_th

        # collect all component print data
        component_print_data = create_print_data_for_all_components(ao, self.general_mesh_props.model_scale)

        # read and set correct page transforms
        page_transforms_per_component = io.read_page_transforms(mesh)
        current_component_print_data : ComponentPrintData
        for current_component_print_data in component_print_data:
            current_component_print_data.page_transform = page_transforms_per_component[current_component_print_data.og_component_id]

        # create page layout
        page_numbers_per_components = io.read_page_numbers(mesh)
        self.num_pages = max(page_numbers_per_components.values()) + 1 if len(page_numbers_per_components) > 0 else 0
        self.components_on_pages = [{} for _ in range(self.num_pages)]
        for current_component_print_data in component_print_data:
            self.components_on_pages[page_numbers_per_components[current_component_print_data.og_component_id]][current_component_print_data.og_component_id] = current_component_print_data
        
        print("Entered page editing mode :)")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def draw_current_page_layout(self, context):
        components_per_page = [list(page_contents.values()) for page_contents in self.components_on_pages]
        num_display_pages = self.num_pages
        if self.active_page is not None and self.active_page == self.num_pages:
            num_display_pages += 1
        show_pages(num_display_pages, components_per_page, self.paper_size, self.active_page, self.selected_component_id, fold_angle_th=self.fold_angle_th)
        redraw_image_editor(context)

    def get_mouse_image_coords(self, context : bpy.types.Context, event :bpy.types.Event):
        mouse_x = event.mouse_region_x
        mouse_y = event.mouse_region_y
        return context.region.view2d.region_to_view(mouse_x, mouse_y)
    
    def select_component(self, context, component_id):
        self.general_mesh_props.selected_component_id = component_id if component_id is not None else -1
        if component_id != self.select_component:
            self.selected_component_id = component_id
            self.draw_current_page_layout(context)

    def exit_modal_mode(self, context):
        print("Exiting page editing mode :)")
        self.select_component(context, None)
        update_all_page_layout_drawings(None, context)
        return {"FINISHED"} 

    def modal(self, context, event : bpy.types.Event):
        if context.region is None or context.region.view2d is None:
            print("Left the workspace!")
            return self.exit_modal_mode(context)
        if self.editing_state == operators_backend.PageEditorState.SELECT_PIECES:
            if event.type in {'ESC', 'ENTER'}:
                return self.exit_modal_mode(context)
            mouse_x, mouse_y = event.mouse_x, event.mouse_y
            image_x, image_y = self.get_mouse_image_coords(context, event)
            region_x = context.region.x
            region_y = context.region.y
            if mouse_x < region_x or mouse_x > region_x + context.region.width or mouse_y < region_y or mouse_y > region_y + context.region.height:
                return {'PASS_THROUGH'}
            if event.type == "MOUSEMOVE":
                page_hovered_over = operators_backend.find_page_under_mouse_position(image_x, image_y, self.num_pages, self.paper_size)
                if page_hovered_over != self.active_page:
                    self.active_page = page_hovered_over
                    self.draw_current_page_layout(context)
            if event.type == "LEFTMOUSE" and event.value == "PRESS":
                page_hovered_over = operators_backend.find_page_under_mouse_position(image_x, image_y, self.num_pages, self.paper_size)
                selected_component_id = operators_backend.find_papermodel_piece_under_mouse_position(image_x, image_y, self.components_on_pages, page_hovered_over, self.paper_size)
                self.select_component(context, selected_component_id)
            if event.type == "RIGHTMOUSE" and event.value == "PRESS":
                self.select_component(context, None)
        return {'PASS_THROUGH'}

    @classmethod
    def poll(cls, context):
        return _active_object_is_mesh_with_paper_model(context)

polyzamboni_keymaps = []

def menu_func_polyzamboni_export_pdf(self, context):
    self.layout.operator(PolyZamboniExportPDFOperator.bl_idname, text="Polyzamboni Export PDF")

def menu_func_polyzamboni_export_svg(self, context):
    self.layout.operator(PolyZamboniExportSVGOperator.bl_idname, text="Polyzamboni Export SVG")

def register():
    bpy.utils.register_class(InitializeCuttingOperator)
    bpy.utils.register_class(ZamboniCutDesignOperator)
    bpy.utils.register_class(ZamboniCutEditingPieMenu)
    bpy.utils.register_class(SyncMeshOperator)
    bpy.utils.register_class(RecomputeFlapsOperator)
    bpy.utils.register_class(SeparateAllMaterialsOperator)
    bpy.utils.register_class(RemoveAllAutoCutsOperator)
    bpy.utils.register_class(FlipGlueFlapsOperator)
    bpy.utils.register_class(ZamboniGlueFlapDesignOperator)
    bpy.utils.register_class(ZamboniGLueFlapEditingPieMenu)
    bpy.utils.register_class(PolyZamboniExportPDFOperator)
    bpy.utils.register_class(PolyZamboniExportSVGOperator)
    bpy.utils.register_class(RemoveAllPolyzamboniDataOperator)
    bpy.utils.register_class(ComputeBuildStepsOperator)
    bpy.utils.register_class(AutoCutsOperator)
    bpy.utils.register_class(SelectMultiTouchingFacesOperator)
    bpy.utils.register_class(SelectNonTriangulatableFacesOperator)
    bpy.utils.register_class(PolyZamboniPageLayoutOperator)
    bpy.utils.register_class(PolyZamboniPageLayoutEditingOperator)

    bpy.types.TOPBAR_MT_file_export.append(menu_func_polyzamboni_export_pdf)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_polyzamboni_export_svg)

    windowmanager = bpy.context.window_manager
    if windowmanager.keyconfigs.addon:
        keymap = windowmanager.keyconfigs.addon.keymaps.new(name="3D View", space_type="VIEW_3D")
        # keymap for the cut editing pie menu
        keymap_item = keymap.keymap_items.new("wm.call_menu_pie", "C", "PRESS", alt=True)
        keymap_item.properties.name = "POLYZAMBONI_MT_CUT_EDITING_PIE_MENU"
        polyzamboni_keymaps.append((keymap, keymap_item))
        # keymap for the glue flap editing pie menu
        keymap_item = keymap.keymap_items.new("wm.call_menu_pie", "X", "PRESS", alt=True)
        keymap_item.properties.name = "POLYZAMBONI_MT_GLUE_FLAP_EDITING_PIE_MENU"
        polyzamboni_keymaps.append((keymap, keymap_item))

def unregister():
    bpy.utils.unregister_class(InitializeCuttingOperator)
    bpy.utils.unregister_class(ZamboniCutDesignOperator)
    bpy.utils.unregister_class(ZamboniCutEditingPieMenu)
    bpy.utils.unregister_class(SyncMeshOperator)
    bpy.utils.unregister_class(RecomputeFlapsOperator)
    bpy.utils.unregister_class(SeparateAllMaterialsOperator)
    bpy.utils.unregister_class(RemoveAllAutoCutsOperator)
    bpy.utils.unregister_class(FlipGlueFlapsOperator)
    bpy.utils.unregister_class(ZamboniGlueFlapDesignOperator)
    bpy.utils.unregister_class(ZamboniGLueFlapEditingPieMenu)
    bpy.utils.unregister_class(PolyZamboniExportPDFOperator)
    bpy.utils.unregister_class(PolyZamboniExportSVGOperator)
    bpy.utils.unregister_class(RemoveAllPolyzamboniDataOperator)
    bpy.utils.unregister_class(ComputeBuildStepsOperator)
    bpy.utils.unregister_class(AutoCutsOperator)
    bpy.utils.unregister_class(SelectMultiTouchingFacesOperator)
    bpy.utils.unregister_class(SelectNonTriangulatableFacesOperator)
    bpy.utils.unregister_class(PolyZamboniPageLayoutOperator)
    bpy.utils.unregister_class(PolyZamboniPageLayoutEditingOperator)

    bpy.types.TOPBAR_MT_file_export.remove(menu_func_polyzamboni_export_pdf)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_polyzamboni_export_svg)

    for keymap, keymap_item in polyzamboni_keymaps:
        keymap.keymap_items.remove(keymap_item)
    polyzamboni_keymaps.clear()
