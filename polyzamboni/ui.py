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
                if zamboni_props.mesh_is_non_manifold:
                    row = layout.row()
                    col1 = row.column()
                    col2 = row.column()
                    col1.operator("polyzamboni.non_manifold_selection_op")
                    col2.label(icon="ERROR")
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
    bl_label = "Glue Flap Editing"
    bl_idname  = "POLYZAMBONI_PT_GlueFlapEditingPanel"
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
                col1.operator("polyzamboni.flaps_recompute_op")
                col2.prop(zamboni_props, "prefer_alternating_flaps", icon="RIGID_BODY", icon_only=True)
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
                row = layout.row()
                row.prop(zamboni_props, "smart_trim_glue_flaps")
            else: 
                layout.label(text="No Cutgraph selected", icon="GHOST_DISABLED")

class POLYZAMBONI_UL_build_sections_list(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        build_section = item
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row()
            row.column().prop(build_section, "name", text="", emboss=False)
            row.column().label(text="(" + str(len(build_section.connected_components)) + " islands)")
            row.column().prop(build_section, "locked", text="", emboss=False, icon="LOCKED" if build_section.locked else "UNLOCKED")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=build_section.name, icon_value=icon)

class POLYZAMBONI_UL_build_sections_list_2D_view(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        build_section = item
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row()
            row.column().prop(build_section, "name", text="", emboss=False)
            row.column().label(text="(" + str(len(build_section.connected_components)) + " islands)")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=build_section.name, icon_value=icon)

class BuildSectionsDetailMenu(bpy.types.Menu):
    bl_label = "Build section specials"
    bl_idname  = "POLYZAMBONI_MT_BuildSectionsSpecials"
    bl_category = "PolyZamboni"

    def draw(self, context):
        layout = self.layout
        ao = context.active_object
        if ao is not None and ao.type == 'MESH':
            active_mesh = ao.data
            zamboni_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
            if zamboni_props.has_attached_paper_model:
                layout.row().operator("polyzamboni.select_build_section_faces_op", icon="SELECT_SET")
                layout.row().operator("polyzamboni.section_overwrite_op", icon="CURRENT_FILE")
                layout.row().operator("polyzamboni.add_to_section_op", icon="ADD")
                layout.row().operator("polyzamboni.remove_from_section_op", icon="REMOVE")
                layout.row().operator("polyzamboni.lock_all_sections_op", icon="VIEW_LOCKED")
                layout.row().operator("polyzamboni.unlock_all_sections_op", icon="VIEW_UNLOCKED")
            else: 
                layout.label(text="No Papermodel selected", icon="GHOST_DISABLED")


class BuildSectionsPanel(bpy.types.Panel):
    bl_label = "Build Sections"
    bl_idname  = "POLYZAMBONI_PT_BuildSectionsPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_MainPanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        ao = context.active_object
        if ao is not None and ao.type == 'MESH':
            active_mesh = ao.data
            zamboni_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
            drawing_settings = context.scene.polyzamboni_drawing_settings

            if zamboni_props.has_attached_paper_model:
                layout.row().prop(drawing_settings, "highlight_active_section")
                layout.row().prop(drawing_settings, "highlight_factor")

                row = layout.row()
                list_col = row.column()
                list_col.template_list("POLYZAMBONI_UL_build_sections_list", "build_sections_view_3d", zamboni_props, "build_sections", zamboni_props, "active_build_section", rows=6)

                actions_col = row.column(align=True)
                actions_col.row(align=True).operator("polyzamboni.section_creation_op", icon="ADD", text="")
                actions_col.row(align=True).operator("polyzamboni.section_removal_op", icon="REMOVE", text="")
                actions_col.separator()
                actions_col.row().menu("POLYZAMBONI_MT_BuildSectionsSpecials", icon="DOWNARROW_HLT", text="")
                actions_col.separator()
                actions_col.row(align=True).operator("polyzamboni.section_move_active_up", icon="TRIA_UP", text="")
                actions_col.row(align=True).operator("polyzamboni.section_move_active_down", icon="TRIA_DOWN", text="")
                actions_col.separator()
                actions_col.row().operator("polyzamboni.section_clear_selection", icon="OBJECT_HIDDEN", text="")
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
        layout.row().prop(drawing_settings, "drawing_enabled")
        layout.row().prop(drawing_settings, "draw_edges")
        layout.row().prop(drawing_settings, "show_glue_flaps")
        layout.row().prop(drawing_settings, "color_faces_by_quality")
        layout.row().prop(drawing_settings, "island_transparency")
        layout.row().prop(drawing_settings, "dotted_line_length")
        layout.row().prop(drawing_settings, "normal_offset")
        
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

class PageLayoutBuildSectionsPanel(bpy.types.Panel):
    bl_label = "Build Sections"
    bl_idname = "POLYZAMBONI_PT_PageLayoutBuildSectionsPanel"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "PolyZamboni"
    bl_parent_id = "POLYZAMBONI_PT_PageLayoutPanel"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        ao = context.active_object
        if ao is not None and ao.type == 'MESH':
            active_mesh = ao.data
            zamboni_props : ZamboniGeneralMeshProps = active_mesh.polyzamboni_general_mesh_props
            drawing_settings = context.scene.polyzamboni_drawing_settings

            if zamboni_props.has_attached_paper_model:
                layout.row().prop(drawing_settings, "highlight_active_section")
                layout.row().prop(drawing_settings, "highlight_factor")

                row = layout.row()
                list_col = row.column()
                list_col.template_list("POLYZAMBONI_UL_build_sections_list_2D_view", "build_sections_view_2", zamboni_props, "build_sections", zamboni_props, "active_build_section")

                actions_col = row.column(align=True)
                actions_col.row(align=True).operator("polyzamboni.section_move_active_up", icon="TRIA_UP", text="")
                actions_col.row(align=True).operator("polyzamboni.section_move_active_down", icon="TRIA_DOWN", text="")
                actions_col.separator()
                actions_col.row().operator("polyzamboni.section_clear_selection", icon="OBJECT_HIDDEN", text="")
            else: 
                layout.label(text="No Cutgraph selected", icon="GHOST_DISABLED")

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
    bpy.utils.register_class(POLYZAMBONI_UL_build_sections_list)
    bpy.utils.register_class(POLYZAMBONI_UL_build_sections_list_2D_view)
    bpy.utils.register_class(BuildSectionsDetailMenu)
    bpy.utils.register_class(MainPanel)
    bpy.utils.register_class(GlueFlapSettingsPanel)
    bpy.utils.register_class(BuildSectionsPanel)
    bpy.utils.register_class(DrawSettingsPanel)
    bpy.utils.register_class(PageLayoutPanel)
    bpy.utils.register_class(PageLayoutBuildSectionsPanel)
    bpy.utils.register_class(PageLayoutDrawSettingsPanel)

def unregister():
    bpy.utils.unregister_class(POLYZAMBONI_UL_build_sections_list)
    bpy.utils.unregister_class(POLYZAMBONI_UL_build_sections_list_2D_view)
    bpy.utils.unregister_class(BuildSectionsDetailMenu)
    bpy.utils.unregister_class(MainPanel)
    bpy.utils.unregister_class(GlueFlapSettingsPanel)
    bpy.utils.unregister_class(BuildSectionsPanel)
    bpy.utils.unregister_class(DrawSettingsPanel)
    bpy.utils.unregister_class(PageLayoutPanel)
    bpy.utils.unregister_class(PageLayoutBuildSectionsPanel)
    bpy.utils.unregister_class(PageLayoutDrawSettingsPanel)
