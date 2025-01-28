import bpy
import bmesh
import numpy as np
import os
from bpy.types import Context
from bpy.props import StringProperty, EnumProperty, FloatProperty, IntProperty, BoolProperty, FloatVectorProperty
from bpy_extras.io_utils import ExportHelper
from .drawing import *
from . import globals
from . import objective_functions
from . import cutgraph
import torch
from .constants import CUTGRAPH_ID_PROPERTY_NAME, CUT_CONSTRAINTS_PROP_NAME, LOCKED_EDGES_PROP_NAME
from . import exporters
from .printprepper import fit_components_on_pages

class FlatteningOperator(bpy.types.Operator):
    bl_label = "Flatten Polygons"
    bl_idname  = "wm.flattening_op"
    _handle = None

    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        # get the currently selected object
        ao = bpy.context.active_object
        flattening_settings = context.scene.polyzamboni_flattening_settings
        print("Attempting to create objective")
        objective_func = objective_functions.ObjectiveFunction(ao, flattening_settings.shape_preservation_weight, 
                                                                   flattening_settings.angle_weight, 
                                                                   flattening_settings.det_weight)
        print("Attempting to create optimizer")
        # print("parameters", list(objective_func.parameters()))
        optimizer = torch.optim.Adam(objective_func.parameters(), lr=flattening_settings.learning_rate)

        print("Starting the best optimizer in the world.")
    
        max_steps = flattening_settings.optimization_iterations
        for step in range(max_steps):
            print("step", step)

            optimizer.zero_grad()

            # Make predictions for this batch
            loss = objective_func()
            print("loss", loss.item())

            # Compute gradient
            loss.backward()
            print("grad-norm", np.linalg.norm(list(objective_func.parameters())[0].grad))

            optimizer.step()
            # print("new variable", list(objective_func.parameters())[0])

            g = list(objective_func.parameters())[0].grad

            if np.linalg.norm(g) <= 1e-8:
                break

            if loss <= 1e-12:
                break
        
        objective_func.apply_back_to_mesh(ao)
            
        return wm.invoke_props_dialog(self)

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        return active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

    def draw(self, context: Context):
        pass

    def cancel(self, context: Context):
        pass

class PrintPlanarityOperator(bpy.types.Operator):
    bl_label = "Check planarity"
    bl_idname = "wm.planarity_printer"

    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager

        ao = bpy.context.active_object
        
        flattening_settings = context.scene.polyzamboni_flattening_settings

        objective_func = objective_functions.ObjectiveFunction(ao, 0, 
                                                               flattening_settings.angle_weight, 
                                                               flattening_settings.det_weight)
        value = objective_func()

        print("AVG Angle-based flatness measure:", value.item())

        return wm.invoke_props_dialog(self, title="AVG Angle-based flatness measure: " + str(value.item()))    

class InitializeCuttingOperator(bpy.types.Operator):
    """Start the unfolding process for this mesh"""
    bl_label = "Unfold this mesh"
    bl_idname  = "polyzamboni.cut_initialization_op"

    def invoke(self, context, event):     
        return self.execute(context)
        
    def execute(self, context):
        # get the currently selected object
        ao = bpy.context.active_object
        ao_pz_settings = ao.polyzamboni_object_prop
        new_cutgraph = cutgraph.CutGraph(ao, np.deg2rad(ao_pz_settings.glue_flap_angle), ao_pz_settings.glue_flap_height, ao_pz_settings.prefer_alternating_flaps)
        globals.add_cutgraph(ao, new_cutgraph)
        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME not in active_object

class ResetAllCutsOperator(bpy.types.Operator):
    """Restart the unfolding process for this mesh"""
    bl_label = "Reset Unfolding"
    bl_idname  = "polyzamboni.cut_reset_op"
    
    def execute(self, context):
        # get the currently selected object
        ao = bpy.context.active_object
        ao[CUT_CONSTRAINTS_PROP_NAME] = []
        ao[LOCKED_EDGES_PROP_NAME] = []
        ao_pz_settings = ao.polyzamboni_object_prop
        new_cutgraph = cutgraph.CutGraph(ao, np.deg2rad(ao_pz_settings.glue_flap_angle), ao_pz_settings.glue_flap_height, ao_pz_settings.prefer_alternating_flaps)
        globals.reset_cutgraph(ao, new_cutgraph)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event, message="Sure? This will delete all cuts.", confirm_text="Reset")

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME in active_object

class SyncMeshOperator(bpy.types.Operator):
    """Transfer mesh changes to the cutgraph to repair it"""
    bl_label = "Sync Mesh Changes"
    bl_idname = "polyzamboni.mesh_sync_op"

    def execute(self, context):
        ao = bpy.context.active_object
        curr_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[ao[CUTGRAPH_ID_PROPERTY_NAME]]
        curr_cutgraph.construct_dual_graph_from_mesh(ao)
        curr_cutgraph.unfold_all_connected_components()
        curr_cutgraph.update_all_flap_geometry()
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME in active_object

class SeparateAllMaterialsOperator(bpy.types.Operator):
    """ Adds cuts to all edges between faces with a different material. """
    bl_label = "Separate Materials"
    bl_idname = "polyzamboni.material_separation_op"

    def execute(self, context):
        ao = bpy.context.active_object
        active_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[ao[CUTGRAPH_ID_PROPERTY_NAME]]
        selected_edges = active_cutgraph.add_cuts_between_different_materials()
        active_cutgraph.compute_all_connected_components()
        active_cutgraph.update_unfoldings_along_edges(selected_edges)
        active_cutgraph.greedy_update_flaps_around_changed_components(selected_edges)
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME in active_object

class RecomputeFlapsOperator(bpy.types.Operator):
    """ Applies the current flap settings and recomputes all glue flaps """
    bl_label = "Recompute Flaps"
    bl_idname = "polyzamboni.flaps_recompute_op"

    def execute(self, context):
        ao = bpy.context.active_object
        ao_zamboni_settings = ao.polyzamboni_object_prop
        curr_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[ao[CUTGRAPH_ID_PROPERTY_NAME]]
        curr_cutgraph.flap_height = ao_zamboni_settings.glue_flap_height
        curr_cutgraph.flap_angle =  np.deg2rad(ao_zamboni_settings.glue_flap_angle)
        curr_cutgraph.prefer_zigzag = ao_zamboni_settings.prefer_alternating_flaps
        if ao_zamboni_settings.lock_glue_flaps:
            curr_cutgraph.update_all_flap_geometry() # recompute flap geometry
        else:
            curr_cutgraph.greedy_place_all_flaps() # replace flaps
        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME in active_object

class FlipGlueFlapsOperator(bpy.types.Operator):
    """ Flip all glue flaps attached to the selected edges """
    bl_label = "Flip Glue Flaps"
    bl_idname = "polyzamboni.flip_flap_op"

    def execute(self, context):
        ao = context.active_object
        ao_mesh = ao.data
        ao_bmesh = bmesh.from_edit_mesh(ao_mesh)
        active_cutgraph_index = ao[CUTGRAPH_ID_PROPERTY_NAME]
        if active_cutgraph_index < len(globals.PZ_CUTGRAPHS):
            active_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[active_cutgraph_index]
        else:
            print("huch?!")
            print("cutgraph", active_cutgraph_index, "not in", globals.PZ_CUTGRAPHS)

        selected_edges = [e.index for e in ao_bmesh.edges if e.select] 

        for edge_index in selected_edges:
            active_cutgraph.swap_glue_flap(edge_index)

        update_all_polyzamboni_drawings(None, context)
        return { 'FINISHED' }
    
    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' 
        return is_mesh and context.mode == 'EDIT_MESH' and CUTGRAPH_ID_PROPERTY_NAME in active_object

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
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' 
        return is_mesh and context.mode == 'EDIT_MESH' and CUTGRAPH_ID_PROPERTY_NAME in active_object

class ZamboniCutDesignOperator(bpy.types.Operator):
    """ Add or remove cuts """
    bl_label = "PolyZamboni Cut Design Tool"
    bl_idname  = "polyzamboni.cut_editing_operator"

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
        active_cutgraph_index = ao[CUTGRAPH_ID_PROPERTY_NAME]
        if active_cutgraph_index < len(globals.PZ_CUTGRAPHS):
            active_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[active_cutgraph_index]
        else:
            print("huch?!")
            print("cutgraph", active_cutgraph_index, "not in", globals.PZ_CUTGRAPHS)

        selected_edges = [e.index for e in ao_bmesh.edges if e.select] 

        if self.design_actions == "ADD_CUT":
            for e_index in selected_edges:
                active_cutgraph.add_cut_constraint(e_index)
        elif self.design_actions == "GLUE_EDGE":
            for e_index in selected_edges:
                active_cutgraph.add_lock_constraint(e_index)
        elif self.design_actions == "RESET_EDGE":
            for e_index in selected_edges:  
                active_cutgraph.clear_edge_constraint(e_index)
        elif self.design_actions == "REGION_CUTOUT":
            selected_faces = [f.index for f in ao_bmesh.faces if f.select]
            active_cutgraph.add_cutout_region(selected_faces)
        # elif self.design_actions == "FLIP_FLAPS":
        #     for e_index in selected_edges:
        #         active_cutgraph.swap_glue_flap(e_index)
        #     update_all_polyzamboni_drawings(None, context)
        #     return {"FINISHED"}

        active_cutgraph.compute_all_connected_components()
        active_cutgraph.update_unfoldings_along_edges(selected_edges)
        active_cutgraph.greedy_update_flaps_around_changed_components(selected_edges)
        update_all_polyzamboni_drawings(None, context)
        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' 
        return is_mesh and context.mode == 'EDIT_MESH' and CUTGRAPH_ID_PROPERTY_NAME in active_object

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

class PolyZamboniExportPDFOperator(bpy.types.Operator, ExportHelper):
    """Export Unfolding of active object"""
    bl_label = "Export Polyzamboni Unfolding as PDF"
    bl_idname = "polyzamboni.export_operator_pdf"

    # Export Helper settings
    filename_ext = ".pdf"

    filter_glob: StringProperty(
        default="*.pdf",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    # Polyzamboni export settings
    paper_size: EnumProperty(
        name="Page Size",
        items=[
            ("A0", "A0", "", "", 0),
            ("A1", "A1", "", "", 1),
            ("A2", "A2", "", "", 2),
            ("A3", "A3", "", "", 3),
            ("A4", "A4", "", "", 4),
            ("A5", "A5", "", "", 5),
            ("A6", "A6", "", "", 6),
            ("A7", "A7", "", "", 7),
            ("A8", "A8", "", "", 8)
        ],
        default="A4"
    )
    page_margin: FloatProperty(
        name="Page margin (cm)",
        default=0.5,
        min=0,
    )
    line_width: FloatProperty(
        name="Line width",
        default=1,
        min=0.1,
    )
    space_between_components: FloatProperty(
        name="Space between pieces",
        default=0.25,
        min=0,
    )
    one_material_per_page: BoolProperty(
        name="One material per page",
        default=True,
    )
    target_model_height: FloatProperty(
        name="Target model height (cm)",
        default=5,
        min=1
    )
    show_step_numbers : BoolProperty(
        name="Show build steps",
        default=True
    )
    show_edge_numbers : BoolProperty(
        name="Show edge numbers",
        default=True
    )
    edge_number_font_size : FloatProperty(
        name="Edge number font size",
        default=8,
        min = 1
    )
    build_steps_font_size : FloatProperty(
        name="Build steps font size",
        default=16,
        min=1
    )
    edge_number_offset : FloatProperty(
        name="Edge number offset (cm)",
        default=0.1,
        min=0
    )
    edge_number_color : FloatVectorProperty(
        name="Edge number color",
        subtype="COLOR",
        default=[0.0,0.0,0.0],
        min=0,
        max=1
    )
    lines_color : FloatVectorProperty(
        name="Line color",
        subtype="COLOR",
        default=[0.0,0.0,0.0],
        min=0,
        max=1
    )
    steps_color : FloatVectorProperty(
        name="Build steps color",
        subtype="COLOR",
        default=[0.0,0.0,0.0],
        min=0,
        max=1
    )
    cut_edge_ls: EnumProperty(
        name="Cut edge linestyle",
        items=[
            ("-", "-", "", "", 0),
            ("..", "..", "", "", 1),
            ("-.", "-.", "", "", 2),
            ("--.", "--.", "", "", 3),
            ("-..", "-..", "", "", 4),
        ],
        default="-"
    )
    convex_fold_edge_ls: EnumProperty(
        name="Convex fold edges linestyle",
        items=[
            ("-", "-", "", "", 0),
            ("..", "..", "", "", 1),
            ("-.", "-.", "", "", 2),
            ("--.", "--.", "", "", 3),
            ("-..", "-..", "", "", 4),
        ],
        default=".."
    )
    concave_fold_edge_ls: EnumProperty(
        name="Concave fold edges linestyle",
        items=[
            ("-", "-", "", "", 0),
            ("..", "..", "", "", 1),
            ("-.", "-.", "", "", 2),
            ("--.", "--.", "", "", 3),
            ("-..", "-..", "", "", 4),
        ],
        default="--."
    )
    glue_flap_ls: EnumProperty(
        name="Glue flap edges linestyle",
        items=[
            ("-", "-", "", "", 0),
            ("..", "..", "", "", 1),
            ("-.", "-.", "", "", 2),
            ("--.", "--.", "", "", 3),
            ("-..", "-..", "", "", 4),
        ],
        default="-"
    )
    apply_textures: BoolProperty(
        name="Apply textures",
        default=True
    )
    print_on_inside: BoolProperty(
        name="Prints inside of mesh",
        default=True
    )
    print_two_sided: BoolProperty(
        name="Two sided texture mode",
        default=False
    )
    hide_fold_edge_angle_th: FloatProperty(
        name="Fold edge hide angle (in deg)",
        default=1,
        min=0,
        max=180
    )



    def execute(self, context):
        print("executed polyzamboni export operator")

        # first, check if the selected model can be unfolded
        ao = context.active_object
        active_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[ao[CUTGRAPH_ID_PROPERTY_NAME]]
        if not active_cutgraph.all_components_have_unfoldings():
            print("POLYZAMBONI WARNING: You exported a mesh that is not fully foldable yet!")

        # prepare everything
        component_print_info = active_cutgraph.create_print_data_for_all_components(ao, active_cutgraph.compute_scaling_factor_for_target_model_height(self.target_model_height))
        page_arrangement = fit_components_on_pages(component_print_info, 
                                                   exporters.paper_sizes[self.paper_size], 
                                                   self.page_margin, 
                                                   self.space_between_components, 
                                                   self.one_material_per_page)

        # initialize exporter
        pdf_exporter = exporters.MatplotlibBasedExporter("pdf", 
                                                         paper_size=self.paper_size, 
                                                         line_width=self.line_width,
                                                         cut_edge_ls=self.cut_edge_ls,
                                                         convex_fold_edge_ls=self.convex_fold_edge_ls,
                                                         concave_fold_edge_ls=self.concave_fold_edge_ls,
                                                         glue_flap_ls=self.glue_flap_ls,
                                                         fold_hide_threshold_angle=self.hide_fold_edge_angle_th,
                                                         show_edge_numbers=self.show_edge_numbers,
                                                         edge_number_font_size=self.edge_number_font_size,
                                                         edge_number_offset=self.edge_number_offset,
                                                         show_build_step_numbers=self.show_step_numbers,
                                                         apply_main_texture=self.apply_textures,
                                                         print_on_inside=self.print_on_inside,
                                                         two_sided_w_texture=self.print_two_sided,
                                                         color_of_lines=self.lines_color,
                                                         color_of_edge_numbers=self.edge_number_color,
                                                         color_of_build_steps=self.steps_color,
                                                         build_step_font_size=self.build_steps_font_size)

        filename, extension = os.path.splitext(self.filepath)
        
        # export file
        pdf_exporter.export(page_arrangement, filename)

        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME in active_object


polyzamboni_keymaps = []

def menu_func_polyzamboni_export_pdf(self, context):
    self.layout.operator(PolyZamboniExportPDFOperator.bl_idname, text="Polyzamboni Export PDF")

def register():
    bpy.utils.register_class(FlatteningOperator)
    bpy.utils.register_class(PrintPlanarityOperator)
    bpy.utils.register_class(InitializeCuttingOperator)
    bpy.utils.register_class(ZamboniCutDesignOperator)
    bpy.utils.register_class(ZamboniCutEditingPieMenu)
    bpy.utils.register_class(ResetAllCutsOperator)
    bpy.utils.register_class(SyncMeshOperator)
    bpy.utils.register_class(RecomputeFlapsOperator)
    bpy.utils.register_class(SeparateAllMaterialsOperator)
    bpy.utils.register_class(FlipGlueFlapsOperator)
    bpy.utils.register_class(ZamboniGlueFlapDesignOperator)
    bpy.utils.register_class(ZamboniGLueFlapEditingPieMenu)
    bpy.utils.register_class(PolyZamboniExportPDFOperator)

    bpy.types.TOPBAR_MT_file_export.append(menu_func_polyzamboni_export_pdf)

    windowmanager = bpy.context.window_manager
    if windowmanager.keyconfigs.addon:
        keymap = windowmanager.keyconfigs.addon.keymaps.new(name="3D View", space_type="VIEW_3D")
        # keymap for the cut editing pie menu
        keymap_item = keymap.keymap_items.new("wm.call_menu_pie", "C", "PRESS", alt=True)
        keymap_item.properties.name = "POLYZAMBONI_MT_CUT_EDITING_PIE_MENU"

        # keymap for the glue flap editing pie menu
        keymap_item = keymap.keymap_items.new("wm.call_menu_pie", "X", "PRESS", alt=True)
        keymap_item.properties.name = "POLYZAMBONI_MT_GLUE_FLAP_EDITING_PIE_MENU"

        polyzamboni_keymaps.append((keymap, keymap_item))

def unregister():
    bpy.utils.unregister_class(FlatteningOperator)
    bpy.utils.unregister_class(PrintPlanarityOperator)
    bpy.utils.unregister_class(InitializeCuttingOperator)
    bpy.utils.unregister_class(ZamboniCutDesignOperator)
    bpy.utils.unregister_class(ZamboniCutEditingPieMenu)
    bpy.utils.unregister_class(ResetAllCutsOperator)
    bpy.utils.unregister_class(SyncMeshOperator)
    bpy.utils.unregister_class(RecomputeFlapsOperator)
    bpy.utils.unregister_class(SeparateAllMaterialsOperator)
    bpy.utils.unregister_class(FlipGlueFlapsOperator)
    bpy.utils.unregister_class(ZamboniGlueFlapDesignOperator)
    bpy.utils.unregister_class(ZamboniGLueFlapEditingPieMenu)
    bpy.utils.unregister_class(PolyZamboniExportPDFOperator)

    #if menu_func_polyzamboni_export in bpy.types.TOPBAR_MT_file_export:
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_polyzamboni_export_pdf)

    for keymap, keymap_item in polyzamboni_keymaps:
        keymap.keymap_items.remove(keymap_item)
    polyzamboni_keymaps.clear()
