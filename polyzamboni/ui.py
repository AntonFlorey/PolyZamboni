import bpy
from bpy.types import Panel

class TestPanel(bpy.types.Panel):
    bl_label = "Test Panel"
    bl_idname  = "POLYZAMBONI_PT_TestPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        row = layout.row()

        row.label(text="dummy text", icon="FUND")
        layout.operator("wm.test_op")

class SubPanel(bpy.types.Panel):
    bl_label = "I am a child"
    bl_idname  = "POLYZAMBONI_PT_PanelB"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_TestPanel"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        scene = context.scene
        polyzamboni_settings = scene.polyzamboni_settings
        row = layout.row()
        row.prop(polyzamboni_settings, "optimization_iterations")
        

def register():
    bpy.utils.register_class(TestPanel)
    bpy.utils.register_class(SubPanel)

def unregister():
    bpy.utils.unregister_class(TestPanel)
    bpy.utils.unregister_class(SubPanel)
    