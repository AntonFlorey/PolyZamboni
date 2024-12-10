import bpy
import bmesh
import numpy as np
from bpy.types import Context
from .drawing import *
from . import pz_globals
from . import objective_functions
import torch

class TestOperator(bpy.types.Operator):
    bl_label = "Flatten Polygons"
    bl_idname  = "wm.test_op"
    _handle = None

    my_float : bpy.props.FloatProperty(name="TestFloat")
    
    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        # get the currently selected object
        ao = bpy.context.active_object
        polyzamboni_settings = context.scene.polyzamboni_settings
        print("Attempting to create objective")
        objective_func = objective_functions.ObjectiveFunction(ao, polyzamboni_settings.shape_preservation_weight, 
                                                                   polyzamboni_settings.angle_weight, 
                                                                   polyzamboni_settings.det_weight)
        print("Attempting to create optimizer")
        # print("parameters", list(objective_func.parameters()))
        optimizer = torch.optim.Adam(objective_func.parameters(), lr=polyzamboni_settings.learning_rate)

        print("Starting the best optimizer in the world.")
    
        max_steps = polyzamboni_settings.optimization_iterations
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

    def draw(self, context: Context):
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.prop(self, "my_float")
        row = col.row()
        row.operator("wm.sub_op")

    def cancel(self, context: Context):
        pass

class SubOperator(bpy.types.Operator):
    bl_label = "Sub-Operator"
    bl_idname  = "wm.sub_op"

    def execute(self, context):
        print("I am a nested operator")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def draw(self, context: Context):
        layout = self.layout
        col = layout.column()
        row = col.row()
        row.label(text="I am a nested  operator")

class PrintPlanarityOperator(bpy.types.Operator):
    bl_label = "PlanarityInfo-Operator"
    bl_idname = "wm.planarity_printer"

    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager

        ao = bpy.context.active_object
        
        polyzamboni_settings = context.scene.polyzamboni_settings

        objective_func = objective_functions.ObjectiveFunction(ao, 0, 
                                                               polyzamboni_settings.angle_weight, 
                                                               polyzamboni_settings.det_weight)
        value = objective_func()

        print("AVG Angle-based flatness measure:", value)

        return wm.invoke_props_dialog(self, title="AVG Angle-based flatness measure: " + str(value.item()))    
    

def register():
    bpy.utils.register_class(TestOperator)
    bpy.utils.register_class(SubOperator)
    bpy.utils.register_class(PrintPlanarityOperator)

def unregister():
    bpy.utils.unregister_class(TestOperator)
    bpy.utils.unregister_class(SubOperator)
    bpy.utils.unregister_class(PrintPlanarityOperator)
