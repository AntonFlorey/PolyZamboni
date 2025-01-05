import bpy
import bmesh
import numpy as np
from bpy.types import Context
from .drawing import *
from . import globals
from . import objective_functions
from . import cutgraph
import torch
from .constants import CUTGRAPH_ID_PROPERTY_NAME

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
    bl_idname  = "wm.cut_initialization_op"

    def invoke(self, context, event):     
        return self.execute(context)
 
    def execute(self, context):
        # get the currently selected object
        ao = bpy.context.active_object
        new_cutgraph = cutgraph.CutGraph(ao)
        globals.add_cutgraph(ao, new_cutgraph)
        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and CUTGRAPH_ID_PROPERTY_NAME not in active_object

    def draw(self, context: Context):
        pass

    def cancel(self, context: Context):
        pass

class ZamboniDesignOperator(bpy.types.Operator):
    """Add or remove cuts"""
    bl_label = "Zamboni Design Tool"
    bl_idname  = "wm.zamboni_design_op"

    design_actions: bpy.props.EnumProperty(
        name="actions",
        description="Select an action",
        items=[
            ("ADD_CUT", "Add Cuts", "Cut the selected edges", "UNLINKED", 0),
            ("GLUE_EDGE", "Glue Edges", "Prevent selected edges from being cut", "LOCKED", 1),
            ("RESET_EDGE", "Clear Edges", "Remove any constraints", "BRUSH_DATA", 2),
            ("REGION_CUTOUT", "Define Region", "Mark the selected faces as one region", "OUTLINER_DATA_SURFACE", 3)
        ],
        default="ADD_CUT"
    )

    def execute(self, context):

        ao_mesh = context.active_object.data
        ao_bmesh = bmesh.from_edit_mesh(ao_mesh)

        selected_verts = [v.index for v in ao_bmesh.verts if v.select] 
        selected_edges = [e.index for e in ao_bmesh.edges if e.select] 
        selected_faces = [f.index for f in ao_bmesh.faces if f.select]

        print("selected vertices are:", selected_verts)
        print("selected edges are:", selected_edges)
        print("selected faces are:", selected_faces)

        if self.design_actions == "ADD_CUT":
            print("Adding a cut")
        elif self.design_actions == "GLUE_EDGE":
            print("Glueing an edge")
        elif self.design_actions == "RESET_EDGE":
            print("Resetting edge")
        elif self.design_actions == "REGION_CUTOUT":
            print("Cutting out a region")

        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' 
        return is_mesh and context.mode == 'EDIT_MESH'

class ZamboniDesignPieMenu(bpy.types.Menu):
    """This is a custom pie menu for all Zamboni design operators"""
    bl_label = "Polyzamboni Tools"
    bl_idname = "VIEW3D_MT_PZ_DESIGN_MENU_PIE"

    def draw(self, context): 
        layout = self.layout 
        pie = layout.menu_pie() 
        pie.operator_enum("wm.zamboni_design_op", "design_actions")

polyzamboni_keymaps = []

def register():
    bpy.utils.register_class(FlatteningOperator)
    bpy.utils.register_class(PrintPlanarityOperator)
    bpy.utils.register_class(InitializeCuttingOperator)
    bpy.utils.register_class(ZamboniDesignOperator)
    bpy.utils.register_class(ZamboniDesignPieMenu)

    windowmanager = bpy.context.window_manager
    if windowmanager.keyconfigs.addon:
        keymap = windowmanager.keyconfigs.addon.keymaps.new(name="3D View", space_type="VIEW_3D")
        keymap_item = keymap.keymap_items.new("wm.call_menu_pie", "C", "PRESS", alt=True)
        keymap_item.properties.name = "VIEW3D_MT_PZ_DESIGN_MENU_PIE"

        polyzamboni_keymaps.append((keymap, keymap_item))

def unregister():
    bpy.utils.unregister_class(FlatteningOperator)
    bpy.utils.unregister_class(PrintPlanarityOperator)
    bpy.utils.unregister_class(InitializeCuttingOperator)
    bpy.utils.unregister_class(ZamboniDesignOperator)
    bpy.utils.unregister_class(ZamboniDesignPieMenu)

    for keymap, keymap_item in polyzamboni_keymaps:
        keymap.keymap_items.remove(keymap_item)
    polyzamboni_keymaps.clear()

