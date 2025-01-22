import bpy
from bpy.types import Panel
from .constants import CUTGRAPH_ID_PROPERTY_NAME

class MainPanel(bpy.types.Panel):
    bl_label = "Poly Zamboni"
    bl_idname  = "POLYZAMBONI_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        row = layout.row()

        row.label(text="thanks for using me!", icon="FUND")
        row = layout.row()
        col1 = row.column()
        col2 = row.column()
        col1.operator("wm.flattening_op")
        col2.label(icon="AUTO")

        if CUTGRAPH_ID_PROPERTY_NAME not in context.active_object:
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.cut_initialization_op")
            col2.label(icon="SHADERFX")
        else:
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.cut_reset_op")
            col2.label(icon="TRASH")
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.mesh_sync_op")
            col2.label(icon="FILE_REFRESH")

            ao = context.active_object
            zamboni_object_settings = ao.polyzamboni_object_prop
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.material_separation_op")
            col2.label(icon="MATERIAL")
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.prop(zamboni_object_settings, "apply_auto_cuts_to_previev", toggle=1)
            col2.label(icon="LIGHT_DATA")
            pass

class GlueFlapSettingsPanel(bpy.types.Panel):
    bl_label = "Glue Flap Settings"
    bl_idname  = "POLYZAMBONI_PT_GlueFlapSettingsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"

    def draw(self, context : bpy.types.Context):
        layout = self.layout
        if CUTGRAPH_ID_PROPERTY_NAME not in context.active_object:
            layout.label(text="No Cutgraph selected", icon="GHOST_DISABLED")
        else:
            ao = context.active_object
            zamboni_object_settings = ao.polyzamboni_object_prop
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.flaps_recompute_op")
            flaps_locked = zamboni_object_settings.lock_glue_flaps
            col2.prop(zamboni_object_settings, "lock_glue_flaps", icon="LOCKED" if flaps_locked else "UNLOCKED", icon_only=True)
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.prop(zamboni_object_settings, "glue_flap_height", icon="DRIVER_DISTANCE")
            col2.label(icon="DRIVER_DISTANCE")
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.prop(zamboni_object_settings, "glue_flap_angle", icon="DRIVER_ROTATIONAL_DIFFERENCE")
            col2.label(icon="DRIVER_ROTATIONAL_DIFFERENCE")
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.prop(zamboni_object_settings, "prefer_alternating_flaps", toggle=1)
            col2.label(icon="RIGID_BODY")
            pass

class FlatteningSettingsPanel(bpy.types.Panel):
    bl_label = "Flattening Settings"
    bl_idname  = "POLYZAMBONI_PT_PanelB"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        scene = context.scene
        flattening_settings = scene.polyzamboni_flattening_settings
        row = layout.row()
        row.prop(flattening_settings, "optimization_iterations")
        row = layout.row()
        row.prop(flattening_settings, "shape_preservation_weight")
        row = layout.row()
        row.prop(flattening_settings, "angle_weight")
        row = layout.row()
        row.prop(flattening_settings, "det_weight")
        row = layout.row()
        row.prop(flattening_settings, "learning_rate")
        
class PrintSettingsPanel(bpy.types.Panel):
    bl_label = "Print Settings"
    bl_idname = "POLYZAMBONI_PT_PrintSettingsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"

    def draw(self, context : bpy.types.Context):
        layout = self.layout
        scene = context.scene
        print_settings = scene.polyzamboni_print_settings
        row = layout.row()
        row.prop(print_settings, "save_format")
        row = layout.row()
        row.prop(print_settings, "target_model_height")
        row = layout.row()
        row.prop(print_settings, "show_step_numbers")
        row = layout.row()
        row.prop(print_settings, "show_edge_numbers")

class DrawSettingsPanel(bpy.types.Panel):
    bl_label = "Draw Settings"
    bl_idname = "POLYZAMBONI_PT_DrawSettingsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"

    def draw(self, context : bpy.types.Context):
        layout = self.layout
        scene = context.scene
        drawing_settings = scene.polyzamboni_drawing_settings
        row = layout.row()
        row.prop(drawing_settings, "drawing_enabled")
        row = layout.row()
        row.prop(drawing_settings, "show_auto_completed_cuts")
        row = layout.row()
        row.prop(drawing_settings, "show_glue_flaps")
        row = layout.row()
        row.prop(drawing_settings, "color_faces_by_quality")
        row = layout.row()
        row.prop(drawing_settings, "dotted_line_length")
        row = layout.row()
        row.prop(drawing_settings, "normal_offset")

class DebugPanel(bpy.types.Panel):
    bl_label = "Debug Info"
    bl_idname = "POLYZAMBONI_PT_PanelDebug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        scene = context.scene
        layout.operator("wm.planarity_printer")

def register():
    bpy.utils.register_class(MainPanel)
    bpy.utils.register_class(GlueFlapSettingsPanel)
    bpy.utils.register_class(FlatteningSettingsPanel)
    bpy.utils.register_class(PrintSettingsPanel)
    bpy.utils.register_class(DrawSettingsPanel)
    bpy.utils.register_class(DebugPanel)

def unregister():
    bpy.utils.unregister_class(MainPanel)
    bpy.utils.unregister_class(GlueFlapSettingsPanel)
    bpy.utils.unregister_class(FlatteningSettingsPanel)
    bpy.utils.unregister_class(PrintSettingsPanel)
    bpy.utils.unregister_class(DrawSettingsPanel)
    bpy.utils.unregister_class(DebugPanel)
