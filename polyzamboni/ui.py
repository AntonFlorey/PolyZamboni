import bpy
from .properties import ZamboniGeneralMeshProps

class MainPanel(bpy.types.Panel):
    bl_label = "Poly Zamboni"
    bl_idname  = "POLYZAMBONI_PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        ao = context.active_object
        row = layout.row()

        row.label(text="thanks for using me!", icon="FUND")

        if ao is not None and ao.type == 'MESH':
            active_mesh = ao.data
            zamboni_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props

            if not zamboni_props.has_attached_paper_model:
                row = layout.row()
                col1 = row.column()
                col2 = row.column()
                col1.operator("polyzamboni.cut_initialization_op")
                col2.label(icon="SHADERFX")
                if zamboni_props.multi_touching_faces_present:
                    row = layout.row()
                    col1 = row.column()
                    col2 = row.column()
                    col1.operator("polyzamboni.multi_touching_face_selection_op")
                    col2.label(icon="ERROR")
                if zamboni_props.faces_which_cant_be_triangulated_are_present:
                    row = layout.row()
                    col1 = row.column()
                    col2 = row.column()
                    col1.operator("polyzamboni.no_tri_face_selection_op")
                    col2.label(icon="ERROR")
            else:
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

                row = layout.row()
                col1 = row.column()
                col2 = row.column()
                col1.operator("polyzamboni.build_order_op")
                col2.label(icon="MOD_BUILD")

                # TODO AUTO CUTS REMOVAL
                # row = layout.row()
                # col1 = row.column()
                # col2 = row.column()
                # col1.prop(zamboni_props, "use_auto_cuts")
                # col2.label(icon="LIGHT_DATA")

                # cutgraph editing
                in_edit_mode = context.mode == 'EDIT_MESH'
                editing_box = layout.box()
                if in_edit_mode:
                    editing_box.label(text="PolyZamboni Editing Tools")
                    # editing_box.row().label(text="Leave (TAB, Alt+Y) or cancel (ESC)")
                    row = editing_box.row()
                    row.label(text="Press Alt+C to edit cuts")
                    row = editing_box.row()
                    row.label(text="Press Alt+X to edit glue flaps")
                    row = editing_box.row()
                    col1 = row.column()
                    col2 = row.column()
                    col1.operator("polyzamboni.material_separation_op")
                    col2.label(icon="MATERIAL")
                    row = editing_box.row()
                    col1 = row.column()
                    col2 = row.column()
                    if context.window_manager.polyzamboni_auto_cuts_running:
                        col1.progress(text="running...", factor=context.window_manager.polyzamboni_auto_cuts_progress)
                        col2.label(icon="SETTINGS")
                    else:
                        col1.operator("polyzamboni.auto_cuts_op")
                        col2.label(icon="FILE_SCRIPT")
                    row = editing_box.row()
                    col1 = row.column()
                    col2 = row.column()
                    col1.operator("polyzamboni.flaps_recompute_op")
                    col2.prop(zamboni_props, "prefer_alternating_flaps", icon="RIGID_BODY", icon_only=True)
                # else:
                #     editing_box.row().operator("polyzamboni.cutgraph_editing_modal_operator")

                # layout.row().label(text="Export")
                row = layout.row()
                col1 = row.column(align=True).column_flow(columns=2, align=True)
                col11 = col1.column(align=True)
                col12 = col1.column(align=True)
                col3 = row.column()
                col11.operator("polyzamboni.export_operator_pdf")
                col12.operator("polyzamboni.export_operator_svg")
                col3.label(icon="FILE_IMAGE")

class GlueFlapSettingsPanel(bpy.types.Panel):
    bl_label = "Glue Flap Settings"
    bl_idname  = "POLYZAMBONI_PT_GlueFlapSettingsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context : bpy.types.Context):
        layout = self.layout
        ao = context.active_object
        if ao is not None and ao.type == 'MESH':
            active_mesh = ao.data
            zamboni_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
            
            if zamboni_props.has_attached_paper_model:
                row = layout.row()
                col1 = row.column()
                col2 = row.column()
                col1.prop(zamboni_props, "glue_flap_height", icon="DRIVER_DISTANCE")
                col2.label(icon="DRIVER_DISTANCE")
                row = layout.row()
                col1 = row.column()
                col2 = row.column()
                col1.prop(zamboni_props, "glue_flap_angle", icon="DRIVER_ROTATIONAL_DIFFERENCE")
                col2.label(icon="DRIVER_ROTATIONAL_DIFFERENCE")
            else: 
                layout.label(text="No Cutgraph selected", icon="GHOST_DISABLED")

class DrawSettingsPanel(bpy.types.Panel):
    bl_label = "Render Settings"
    bl_idname = "POLYZAMBONI_PT_DrawSettingsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context : bpy.types.Context):
        layout = self.layout
        scene = context.scene
        drawing_settings = scene.polyzamboni_drawing_settings
        row = layout.row()
        row.prop(drawing_settings, "drawing_enabled")
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
