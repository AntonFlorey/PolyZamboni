import bpy
import bmesh
from bpy.types import Context
from .drawing import *
from . import pz_globals
from . import objective_functions

class TestOperator(bpy.types.Operator):
    bl_label = "Test Operator"
    bl_idname  = "wm.test_op"
    _handle = None

    my_float : bpy.props.FloatProperty(name="TestFloat")
    
    def execute(self, context):
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        # get the currently selected object
        ao = bpy.context.active_object
        objective_func = objective_functions.ObjectiveFunction(ao)
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

def register():
    bpy.utils.register_class(TestOperator)
    bpy.utils.register_class(SubOperator)

def unregister():
    bpy.utils.unregister_class(TestOperator)
    bpy.utils.unregister_class(SubOperator)
