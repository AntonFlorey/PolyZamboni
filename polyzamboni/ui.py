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

                # cutgraph editing
                in_edit_mode = context.mode == 'EDIT_MESH'
                editing_box = layout.box()
                if in_edit_mode:
                    if context.window_manager.polyzamboni_in_page_edit_mode:
                        editing_box.label(text="PolyZamboni Editing Tools")
                        row = editing_box.row()
                        row.label(text="Please leave the page layout editor first.")
                    else:
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
                        col1.operator("polyzamboni.auto_cuts_removal_op")
                        col2.label(icon="BRUSH_DATA")
                        row = editing_box.row()
                        col1 = row.column()
                        col2 = row.column()
                        col1.operator("polyzamboni.flaps_recompute_op")
                        col2.prop(zamboni_props, "prefer_alternating_flaps", icon="RIGID_BODY", icon_only=True)
                        row = editing_box.row()
                        col1 = row.column()
                        col2 = row.column()
                        col1.operator("polyzamboni.build_order_op")
                        col2.label(icon="MOD_BUILD")
                else:
                    editing_box.label(text="Modify Paper Model in Edit Mode")
                    # editing_box.row().operator("polyzamboni.cutgraph_editing_modal_operator")

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

class PageLayoutPanel(bpy.types.Panel):
    bl_label = "Poly Zamboni"
    bl_idname  = "POLYZAMBONI_PT_PageLayoutPanel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"

    def draw(self, context : bpy.types.Context):
        layout = self.layout
        scene = context.scene
        drawing_settings = scene.polyzamboni_drawing_settings
        row = layout.row()
        row.label(text="Page Layout Preview", icon="FUND")
        row = layout.row()
        row.operator("polyzamboni.page_layout_op")
        row.active = not context.window_manager.polyzamboni_in_page_edit_mode
        editing_box = layout.box()
        if not context.window_manager.polyzamboni_in_page_edit_mode:
            editing_box.row().operator("polyzamboni.page_layout_editing_op")
        else:
            editing_box.row().operator("polyzamboni.exit_page_layout_editing_op")
            editing_box.row().label(text="Select piece with LMB", icon="MOUSE_LMB")
            editing_box.row().label(text="Move selected piece with G", icon="VIEW_PAN")
            editing_box.row().label(text="Rotate selected piece with R", icon="PREFERENCES")
            editing_box.row().label(text="Change step number with F", icon="MOD_BUILD")
        row = layout.row()
        col1 = row.column(align=True).column_flow(columns=2, align=True)
        col11 = col1.column(align=True)
        col12 = col1.column(align=True)
        col3 = row.column()
        col11.operator("polyzamboni.export_operator_pdf")
        col12.operator("polyzamboni.export_operator_svg")
        col3.label(icon="FILE_IMAGE")

class PageLayoutDrawSettingsPanel(bpy.types.Panel):
    bl_label = "Render Settings"
    bl_idname = "POLYZAMBONI_PT_PageLayoutDrawSettingsPanel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_PageLayoutPanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context : bpy.types.Context):
        scene = context.scene
        drawing_settings = scene.polyzamboni_drawing_settings
        layout = self.layout
        row = layout.row()
        row.prop(drawing_settings, "show_page_layout")
        row = layout.row()
        col1 = row.column()
        col2 = row.column()
        col1.prop(drawing_settings, "show_component_colors")
        col2.label(icon="COLOR")
        row = layout.row()
        col1 = row.column()
        col2 = row.column()
        col1.prop(drawing_settings, "show_build_step_numbers")
        col2.label(icon="MOD_BUILD")
        row = layout.row()
        col1 = row.column()
        col2 = row.column()
        col1.prop(drawing_settings, "hide_fold_edge_angle_th")
        col2.label(icon="CON_ROTLIMIT")

def register():
    bpy.utils.register_class(MainPanel)
    bpy.utils.register_class(GlueFlapSettingsPanel)
    bpy.utils.register_class(DrawSettingsPanel)
    bpy.utils.register_class(PageLayoutPanel)
    bpy.utils.register_class(PageLayoutDrawSettingsPanel)

def unregister():
    bpy.utils.unregister_class(MainPanel)
    bpy.utils.unregister_class(GlueFlapSettingsPanel)
    bpy.utils.unregister_class(DrawSettingsPanel)
    bpy.utils.unregister_class(PageLayoutPanel)
    bpy.utils.unregister_class(PageLayoutDrawSettingsPanel)
