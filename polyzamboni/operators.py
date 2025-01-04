import bpy
import bmesh
import numpy as np
from bpy.types import Context
from .drawing import *
from . import pz_globals
from . import objective_functions
from . import cutgraph
import torch

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
        pz_globals.add_cutgraph(ao, new_cutgraph)
        return { 'FINISHED' }

    @classmethod
    def poll(cls, context):
        active_object = context.active_object
        is_mesh = active_object is not None and active_object.type == 'MESH' and (context.mode == 'EDIT_MESH' or active_object.select_get())

        return is_mesh and "cut_graph_id" not in active_object

    def draw(self, context: Context):
        pass

    def cancel(self, context: Context):
        pass

# bpy.types.SpaceView3D.
 
def register():
    bpy.utils.register_class(FlatteningOperator)
    bpy.utils.register_class(PrintPlanarityOperator)
    bpy.utils.register_class(InitializeCuttingOperator)

def unregister():
    bpy.utils.unregister_class(FlatteningOperator)
    bpy.utils.unregister_class(PrintPlanarityOperator)
    bpy.utils.unregister_class(InitializeCuttingOperator)
