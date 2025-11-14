import bpy
from bpy.types import Scene
import numpy as np
from .drawing import update_all_polyzamboni_drawings, update_all_page_layout_drawings
from .callbacks import update_glueflap_geometry_callback, update_all_drawings_callback

# For more information about Blender Properties, visit:
# <https://blender.org/api/blender_python_api_2_78a_release/bpy.types.Property.html>
from bpy.props import BoolProperty
from bpy.props import EnumProperty
from bpy.props import FloatProperty
from bpy.props import IntProperty
from bpy.props import FloatVectorProperty
from bpy.props import CollectionProperty
from bpy.props import StringProperty

class DrawSettings(bpy.types.PropertyGroup):
    drawing_enabled: BoolProperty(
        name="Show Zamboni interface",
        default=True,
        update=update_all_polyzamboni_drawings
    )
    draw_edges: BoolProperty(
        name="Draw edges",
        default=True,
        update=update_all_polyzamboni_drawings
    )
    color_faces_by_quality: BoolProperty(
        name="Show region quality",
        default=True,
        update=update_all_polyzamboni_drawings
    )
    dotted_line_length: FloatProperty(
        name="Dotted line length",
        default=0.1,
        min=0,
        max=10,
        update=update_all_polyzamboni_drawings
    )
    island_transparency: FloatProperty(
        name="Region transparency",
        default=0.5,
        min=0,
        max=1,
        update=update_all_polyzamboni_drawings
    )
    normal_offset: FloatProperty(
        name="Normal offset",
        default=0.01,
        min=0.001,
        max=1.0,
        update=update_all_polyzamboni_drawings
    )
    show_glue_flaps: BoolProperty(
        name="Show glue flaps",
        default=True,
        update=update_all_polyzamboni_drawings
    )
    show_page_layout: BoolProperty(
        name="Show Page Layout",
        default=True,
        update=update_all_page_layout_drawings
    )
    hide_fold_edge_angle_th: FloatProperty(
        name="Min fold edge angle",
        description="Folds below this angle wont be drawn",
        default=np.deg2rad(1),
        min=0,
        max=np.pi,
        subtype="ANGLE",
        update=update_all_page_layout_drawings
    )
    show_component_colors: BoolProperty(
        name="Display colors",
        default=True,
        update=update_all_page_layout_drawings
    )
    show_build_step_numbers: BoolProperty(
        name="Build step numbers",
        default=True,
        update=update_all_page_layout_drawings
    )
    highlight_active_section : BoolProperty(
        name="Highlight selected section",
        default=True,
        update=update_all_polyzamboni_drawings
    )
    highlight_factor : FloatProperty(
        name="Highlight factor",
        description="A value of 1 makes all non-selected sections disappear",
        default=0.5,
        min=0,
        max=1,
        update=update_all_polyzamboni_drawings
    )

class ConnectedComponentProperty(bpy.types.PropertyGroup):
    id : IntProperty(
        name="Component id"
    )

class BuildSectionProperty(bpy.types.PropertyGroup):
    name : StringProperty(
        name="Section Name",
        default="A"
    )
    connected_components : CollectionProperty(
        type = ConnectedComponentProperty,
        name="Connected components"
    )
    locked : BoolProperty(
        name="Lock section",
        description="When set to true, this section can only be changed manually.",
        default=True
    )

class ZamboniGeneralMeshProps(bpy.types.PropertyGroup):
    has_attached_paper_model : BoolProperty(
        name="Has attached paper model",
        description="Is true if the mesh has a paper model attached to it",
        default=False
    )
    glue_flap_height : FloatProperty(
        name="Glue flap height",
        description="Controls how far the glue flaps extend",
        default=0.15,
        min=0.01,
        update=update_glueflap_geometry_callback
    )
    glue_flap_angle : FloatProperty(
        name="Glue flap angle",
        description="Determines the shape of all glue flaps",
        default=np.pi / 4,
        min=np.deg2rad(10),
        max=np.deg2rad(90),
        subtype="ANGLE",
        update=update_glueflap_geometry_callback
    )
    prefer_alternating_flaps : BoolProperty(
        name="ZigZag Flaps",
        description="If glue flaps should alternate along a patch boundary",
        default=True
    )
    smart_trim_glue_flaps : BoolProperty(
        name="Smart flap trimming",
        description="Glue flaps get automatically trimmed to fit on the piece it is glued onto",
        default=True,
        update=update_glueflap_geometry_callback
    )
    mesh_is_non_manifold : BoolProperty(
        name="Mesh is non manifold",
        default=False
    )
    multi_touching_faces_present : BoolProperty(
        name="Multi touching faces present",
        default=False
    )
    faces_which_cant_be_triangulated_are_present : BoolProperty(
        name="Not tri-able faces present",
        default=False
    )
    paper_size: EnumProperty(
        name="Page Size",
        items=[
            ("A0", "A0", "", "", 0),
            ("A1", "A1", "", "", 1),
            ("A2", "A2", "", "", 2),
            ("A3", "A3", "", "", 3),
            ("A4", "A4", "", "", 4),
            ("A5", "A5", "", "", 5),
            ("A6", "A6", "", "", 6),
            ("A7", "A7", "", "", 7),
            ("A8", "A8", "", "", 8),
            ("Letter", "Letter", ", ", 9),
            ("Legal", "Legal", "", "", 10),
            ("Tabloid", "Tabloid", "", "", 11),
            ("Custom", "Custom", "", "", 12)
        ],
        default="A4"
    )
    custom_page_width : FloatProperty(
        name="Width",
        subtype="DISTANCE",
        min=0.001,
        default=0.21
    )
    custom_page_height : FloatProperty(
        name="Height",
        subtype="DISTANCE",
        min=0.001,
        default=0.297
    )
    model_scale : FloatProperty(
        name="Model scale",
        description="Controls the size of the printed patterns",
        default=1,
        min=0
    )
    selected_component_id : IntProperty(
        name="Selected Component ID",
        default=-1,
        update=update_all_polyzamboni_drawings
    )
    build_sections : CollectionProperty(
        type=BuildSectionProperty,
        name="Build sections"
    )
    active_build_section : IntProperty(
        name="Active build section",
        default=-1,
        update=update_all_drawings_callback
    )

linestyles = [
        ("-", "-", "", "", 0),
        ("..", "..", "", "", 1),
        ("-.", "-.", "", "", 2),
        ("--.", "--.", "", "", 3),
        ("-..", "-..", "", "", 4)
    ]

class GeneralExportSettings(bpy.types.PropertyGroup):
    use_custom_layout: BoolProperty(
        name="Use custom layout",
        default=False
    )
    paper_size: EnumProperty(
        name="Page Size",
        items=[
            ("A0", "A0", "", "", 0),
            ("A1", "A1", "", "", 1),
            ("A2", "A2", "", "", 2),
            ("A3", "A3", "", "", 3),
            ("A4", "A4", "", "", 4),
            ("A5", "A5", "", "", 5),
            ("A6", "A6", "", "", 6),
            ("A7", "A7", "", "", 7),
            ("A8", "A8", "", "", 8),
            ("Letter", "Letter", ", ", 9),
            ("Legal", "Legal", "", "", 10),
            ("Tabloid", "Tabloid", "", "", 11),
            ("Custom", "Custom", "", "", 12)
        ],
        default="A4"
    )
    custom_page_width : FloatProperty(
        name="Width",
        subtype="DISTANCE",
        min=0.001,
        default=0.21
    )
    custom_page_height : FloatProperty(
        name="Height",
        subtype="DISTANCE",
        min=0.001,
        default=0.297
    )
    page_margin: FloatProperty(
        name="Page margin",
        default=0.005,
        min=0,
        subtype="DISTANCE"
    )
    space_between_components: FloatProperty(
        name="Space between pieces",
        default=0.0025,
        min=0,
        subtype="DISTANCE"
    )
    one_material_per_page: BoolProperty(
        name="One material per page",
        default=True,
    )
    target_model_height: FloatProperty(
        name="Target model height",
        default=0.1,
        min=0.0001,
        subtype="DISTANCE"
    )
    show_step_numbers : BoolProperty(
        name="Show build steps",
        default=True
    )
    show_edge_numbers : BoolProperty(
        name="Show edge numbers",
        default=True
    )
    edge_number_font_size : IntProperty(
        name="Edge number font size",
        default=6,
        min = 1
    )
    build_steps_font_size : IntProperty(
        name="Build steps font size",
        default=12,
        min=1
    )
    edge_number_color : FloatVectorProperty(
        name="Edge number color",
        subtype="COLOR",
        default=[0.0,0.0,0.0],
        min=0,
        max=1
    )
    steps_color : FloatVectorProperty(
        name="Build steps color",
        subtype="COLOR",
        default=[0.0,0.0,0.0],
        min=0,
        max=1
    )
    print_on_inside : BoolProperty(
        name="Prints inside of mesh",
        description="After glueing the pieces together, prints will be on the meshes inside if set to True",
        default=True
    )
    scaling_mode : EnumProperty(
        name="Scaling mode",
        items=[
            ("HEIGHT", "Target height", "Scales all pieces to achieve the desired model height", "DRIVER_DISTANCE", 0),
            ("SCALE", "Set scale", "Directly define the scaling factor from blender units to the unit active in the scene (m per default)", "FULLSCREEN_ENTER", 1),
        ],
        default="HEIGHT"
    )
    sizing_scale: FloatProperty(
        name="Custom scaling factor",
        default=1,
        min=0
    )

class LineExportSettings(bpy.types.PropertyGroup):
    line_width: FloatProperty(
        name="Line width",
        default=0.5,
        min=0.1,
    )
    hide_fold_edge_angle_th: FloatProperty(
        name="Min fold angle to print a fold edge.",
        default=np.deg2rad(1),
        min=0,
        max=np.pi,
        subtype="ANGLE"
    )
    cut_edge_ls: EnumProperty(
        name="Cut edge linestyle",
        items=linestyles,
        default="-"
    )
    convex_fold_edge_ls: EnumProperty(
        name="Convex fold edges linestyle",
        items=linestyles,
        default=".."
    )
    concave_fold_edge_ls: EnumProperty(
        name="Concave fold edges linestyle",
        items=linestyles,
        default="--."
    )
    glue_flap_ls: EnumProperty(
        name="Glue flap edges linestyle",
        items=linestyles,
        default="-"
    )
    lines_color : FloatVectorProperty(
        name="Line color",
        subtype="COLOR",
        default=[0.0,0.0,0.0],
        min=0,
        max=1
    )
    edge_number_offset : FloatProperty(
        name="Edge number offset",
        default=0.0005,
        min=0,
        subtype="DISTANCE"
    )

class TextureExportSettings(bpy.types.PropertyGroup):
    apply_textures : BoolProperty(
        name="Apply textures",
        description="Applies some texture to all faces. If no texture can be found in a materials node tree, the diffuse color is used",
        default=True
    )
    print_two_sided : BoolProperty(
        name="Two sided texture mode",
        description="When selected, build instructions and textures are printed on separate pages for two-sided printing",
        default=False
    )
    triangle_bleed : FloatProperty(
        name="Triangle bleed",
        description="Each textured triangle drawn gets thickened by this amount. Increase this value to close visible gaps in your prints. A high value will lead to visible seams due to misalignment",
        subtype="DISTANCE",
        min=0,
        default=0.0001
    )
    apply_glue_flap_color : BoolProperty(
        name="Color glue flaps",
        description="Enable this to color glue flap faces. This is helpful for knowing what has to be glued",
        default=True
    )
    glue_flap_color : FloatVectorProperty(
        name="Glue flap color",
        subtype="COLOR",
        default=[0.8,0.8,0.8],
        min=0,
        max=1
    )

class PageLayoutCreationSettings(bpy.types.PropertyGroup):
    paper_size: EnumProperty(
        name="Page Size",
        items=[
            ("A0", "A0", "", "", 0),
            ("A1", "A1", "", "", 1),
            ("A2", "A2", "", "", 2),
            ("A3", "A3", "", "", 3),
            ("A4", "A4", "", "", 4),
            ("A5", "A5", "", "", 5),
            ("A6", "A6", "", "", 6),
            ("A7", "A7", "", "", 7),
            ("A8", "A8", "", "", 8),
            ("Letter", "Letter", ", ", 9),
            ("Legal", "Legal", "", "", 10),
            ("Tabloid", "Tabloid", "", "", 11),
            ("Custom", "Custom", "", "", 12)
        ],
        default="A4"
    )
    custom_page_width : FloatProperty(
        name="Width",
        subtype="DISTANCE",
        min=0.001,
        default=0.21
    )
    custom_page_height : FloatProperty(
        name="Height",
        subtype="DISTANCE",
        min=0.001,
        default=0.297
    )
    page_margin: FloatProperty(
        name="Page margin",
        default=0.005,
        min=0,
        subtype="DISTANCE"
    )
    space_between_components: FloatProperty(
        name="Space between pieces",
        default=0.0025,
        min=0,
        subtype="DISTANCE"
    )
    one_material_per_page: BoolProperty(
        name="One material per page",
        default=True,
    )
    scaling_mode: EnumProperty(
        name="Scaling mode",
        items=[
            ("HEIGHT", "Target height", "Scales all pieces to achieve the desired model height", "DRIVER_DISTANCE", 0),
            ("SCALE", "Set scale", "Directly define the scaling factor from blender units to the unit active in the scene (m per default)", "FULLSCREEN_ENTER", 1),
        ],
        default="HEIGHT"
    )
    target_model_height: FloatProperty(
        name="Target model height",
        default=0.1,
        min=0.0001,
        subtype="DISTANCE"
    )
    sizing_scale: FloatProperty(
        name="Custom scaling factor",
        default=1,
        min=0
    )

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(ConnectedComponentProperty)
    bpy.utils.register_class(BuildSectionProperty)
    bpy.utils.register_class(DrawSettings)
    bpy.utils.register_class(ZamboniGeneralMeshProps)
    bpy.utils.register_class(GeneralExportSettings)
    bpy.utils.register_class(LineExportSettings)
    bpy.utils.register_class(TextureExportSettings)
    bpy.utils.register_class(PageLayoutCreationSettings)
    Scene.polyzamboni_drawing_settings = bpy.props.PointerProperty(type=DrawSettings)
    bpy.types.Mesh.polyzamboni_general_mesh_props = bpy.props.PointerProperty(type=ZamboniGeneralMeshProps)
    bpy.types.WindowManager.polyzamboni_auto_cuts_progress = FloatProperty(name="Auto Cuts Progress", min=0, max=1, default=0.0)
    bpy.types.WindowManager.polyzamboni_auto_cuts_running = BoolProperty(name="Auto Cuts computation running", default=False)
    bpy.types.WindowManager.polyzamboni_in_page_edit_mode = BoolProperty(name="Editing page layout", default=False)
    
def unregister():
    bpy.utils.unregister_class(ZamboniGeneralMeshProps)
    bpy.utils.unregister_class(BuildSectionProperty)
    bpy.utils.unregister_class(ConnectedComponentProperty)
    bpy.utils.unregister_class(DrawSettings)
    bpy.utils.unregister_class(GeneralExportSettings)
    bpy.utils.unregister_class(LineExportSettings)
    bpy.utils.unregister_class(TextureExportSettings)
    bpy.utils.unregister_class(PageLayoutCreationSettings)
    del Scene.polyzamboni_drawing_settings
    del bpy.types.Mesh.polyzamboni_general_mesh_props
    del bpy.types.WindowManager.polyzamboni_auto_cuts_progress
    del bpy.types.WindowManager.polyzamboni_auto_cuts_running
    del bpy.types.WindowManager.polyzamboni_in_page_edit_mode