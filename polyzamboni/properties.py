import bpy
from bpy.types import Scene
import numpy as np
from .drawing import update_all_polyzamboni_drawings
from .callbacks import update_glueflap_geometry_callback

# For more information about Blender Properties, visit:
# <https://blender.org/api/blender_python_api_2_78a_release/bpy.types.Property.html>
from bpy.props import BoolProperty
from bpy.props import EnumProperty
from bpy.props import FloatProperty
from bpy.props import IntProperty
from bpy.props import FloatVectorProperty

class DrawSettings(bpy.types.PropertyGroup):
    drawing_enabled: BoolProperty(
        name="Show Zamboni interface",
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
    normal_offset: FloatProperty(
        name="Normal offset",
        default=0.01,
        min=0.01,
        max=0.5,
        update=update_all_polyzamboni_drawings
    )
    show_glue_flaps: BoolProperty(
        name="Show glue flaps",
        default=True,
        update=update_all_polyzamboni_drawings
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
        max=np.deg2rad(170),
        subtype="ANGLE",
        update=update_glueflap_geometry_callback
    )
    prefer_alternating_flaps : BoolProperty(
        name="ZigZag Flaps",
        description="If glue flaps should alternate along a patch boundary",
        default=True
    )
    lock_glue_flaps : BoolProperty(
        name="Lock Flap Positions",
        description="Determines if glue flaps are allowed to be relocated when applying new settings",
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

linestyles = [
        ("-", "-", "", "", 0),
        ("..", "..", "", "", 1),
        ("-.", "-.", "", "", 2),
        ("--.", "--.", "", "", 3),
        ("-..", "-..", "", "", 4)
    ]

class GeneralExportSettings(bpy.types.PropertyGroup):
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
            ("Tabloid", "Tabloid", "", "", 11)
        ],
        default="A4"
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
    print_on_inside: BoolProperty(
        name="Prints inside of mesh",
        description="After glueing the pieces together, prints will be on the meshes inside if set to True",
        default=True
    )
    scaling_mode: EnumProperty(
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
    apply_textures: BoolProperty(
        name="Apply textures",
        description="Applies some texture to all faces. If no texture can be found in a materials node tree, the diffuse color is used",
        default=True
    )
    print_two_sided: BoolProperty(
        name="Two sided texture mode",
        description="When selected, build instructions and textures are printed on separate pages for two-sided printing",
        default=False
    )

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(DrawSettings)
    bpy.utils.register_class(ZamboniGeneralMeshProps)
    bpy.utils.register_class(GeneralExportSettings)
    bpy.utils.register_class(LineExportSettings)
    bpy.utils.register_class(TextureExportSettings)
    Scene.polyzamboni_drawing_settings = bpy.props.PointerProperty(type=DrawSettings)
    bpy.types.Mesh.polyzamboni_general_mesh_props = bpy.props.PointerProperty(type=ZamboniGeneralMeshProps)
    bpy.types.WindowManager.polyzamboni_auto_cuts_progress = FloatProperty(name="Auto Cuts Progress", min=0, max=1, default=0.0)
    bpy.types.WindowManager.polyzamboni_auto_cuts_running = BoolProperty(name="Auto Cuts computation running", default=False)

def unregister():
    bpy.utils.unregister_class(DrawSettings)
    bpy.utils.unregister_class(ZamboniGeneralMeshProps)
    bpy.utils.unregister_class(GeneralExportSettings)
    bpy.utils.unregister_class(LineExportSettings)
    bpy.utils.unregister_class(TextureExportSettings)
    del Scene.polyzamboni_drawing_settings
    del bpy.types.Mesh.polyzamboni_general_mesh_props
    del bpy.types.WindowManager.polyzamboni_auto_cuts_progress
    del bpy.types.WindowManager.polyzamboni_auto_cuts_running