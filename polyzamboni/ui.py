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

        if CUTGRAPH_ID_PROPERTY_NAME not in context.active_object:
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.cut_initialization_op")
            col2.label(icon="SHADERFX")
        else:
            row = layout.row()
            row.label(text="Press Alt+C in Edit-Mode :)")
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.operator("polyzamboni.remove_all_op")
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
            col1.operator("polyzamboni.build_order_op")
            col2.label(icon="MOD_BUILD")
            row = layout.row()
            col1 = row.column()
            col2 = row.column()
            col1.prop(zamboni_object_settings, "apply_auto_cuts_to_previev", toggle=1)
            col2.label(icon="LIGHT_DATA")

            # layout.row().label(text="Export")
            row = layout.row()
            col1 = row.column(align=True).column_flow(columns=2, align=True)
            col11 = col1.column(align=True)
            col12 = col1.column(align=True)
            col3 = row.column()
            col11.operator("polyzamboni.export_operator_pdf")
            col12.operator("polyzamboni.export_operator_svg")
            col3.label(icon="FILE_IMAGE")
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


def register():
    bpy.utils.register_class(MainPanel)
    bpy.utils.register_class(GlueFlapSettingsPanel)
    bpy.utils.register_class(DrawSettingsPanel)

def unregister():
    bpy.utils.unregister_class(MainPanel)
    bpy.utils.unregister_class(GlueFlapSettingsPanel)
    bpy.utils.unregister_class(DrawSettingsPanel)
