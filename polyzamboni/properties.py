import bpy
from bpy.types import Scene
from .drawing import update_all_polyzamboni_drawings

# For more information about Blender Properties, visit:
# <https://blender.org/api/blender_python_api_2_78a_release/bpy.types.Property.html>
from bpy.props import BoolProperty
# from bpy.props import CollectionProperty
from bpy.props import EnumProperty
from bpy.props import FloatProperty
from bpy.props import IntProperty
# from bpy.props import PointerProperty
from bpy.props import StringProperty
# from bpy.props import PropertyGroup

#
# Add additional functions or classes here
#

class FlatteningSettings(bpy.types.PropertyGroup):
    optimization_iterations: IntProperty(
        name="Optimization Rounds",
        default=250,
        min=0,
        max=1000
    )
    shape_preservation_weight: FloatProperty(
        name="Shape Preservation Weight",
        default=1,
        min=0,
        max=1000000
    )
    angle_weight: FloatProperty(
        name="Angle Weight",
        default=0,
        min=0,
        max=1
    )
    det_weight: FloatProperty(
        name="Determinant Weight",
        default=1,
        min=0,
        max=1
    )
    learning_rate: FloatProperty(
        name="Optimizer Step Size",
        default=0.001,
        min=1e-8,
        max=10
    )

class DrawSettings(bpy.types.PropertyGroup):
    drawing_enabled: BoolProperty(
        name="Show Zamboni interface",
        default=True,
        update=update_all_polyzamboni_drawings
    )
    show_auto_completed_cuts: BoolProperty(
        name="Show auto-cuts",
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
        default=0.1,
        min=0.01,
        max=0.5,
        update=update_all_polyzamboni_drawings
    )
    show_glue_flaps: BoolProperty(
        name="Show glue flaps",
        default=True,
        update=update_all_polyzamboni_drawings
    )

class PrintSettings(bpy.types.PropertyGroup):
    target_model_height: FloatProperty(
        name="Target model height (cm)",
        default=15,
        min=1
    )
    # Enum Item: (identifier, name, description, icon, number)
    save_format: EnumProperty(
        name="Output format",
        items=[
            ("pdf", "pdf", "", "", 0),
            ("svg", "svg", "", "", 1)
        ],
        default="pdf"
    )
    show_step_numbers : BoolProperty(
        name="Show build steps",
        default=True
    )
    show_edge_numbers : BoolProperty(
        name="Show edge numbers",
        default=True
    )

class ZamboniSettingsPerObject(bpy.types.PropertyGroup):
    glue_flap_height : FloatProperty(
        name="Glue flap height",
        description="Controls how far the glue flaps extend.",
        default=1.0,
        min=0.01
    )
    glue_flap_angle : FloatProperty(
        name="Glue flap angle",
        description="Determines the shape of all glue flaps.",
        default=45,
        min=10,
        max=170
    )
    prefer_alternating_flaps : BoolProperty(
        name="ZigZag Flaps",
        description="If glue flaps should alternate along a patch boundary.",
        default=True
    )
    apply_auto_cuts_to_previev : BoolProperty(
        name="Auto cuts preview",
        description="If set to True, all automatically generated cuts will be condidered when showing a preview of the mesh unfolding.",
        default=False
    )

print("registering polyzamboni properties")

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(FlatteningSettings)
    bpy.utils.register_class(DrawSettings)
    bpy.utils.register_class(PrintSettings)
    bpy.utils.register_class(ZamboniSettingsPerObject)
    Scene.polyzamboni_flattening_settings = bpy.props.PointerProperty(type=FlatteningSettings)
    Scene.polyzamboni_drawing_settings = bpy.props.PointerProperty(type=DrawSettings)
    Scene.polyzamboni_print_settings = bpy.props.PointerProperty(type=PrintSettings)
    bpy.types.Object.polyzamboni_object_prop = bpy.props.PointerProperty(type=ZamboniSettingsPerObject)

def unregister():
    bpy.utils.unregister_class(FlatteningSettings)
    bpy.utils.unregister_class(DrawSettings)
    bpy.utils.unregister_class(PrintSettings)
    bpy.utils.unregister_class(ZamboniSettingsPerObject)
    del Scene.polyzamboni_flattening_settings
    del Scene.polyzamboni_drawing_settings
    del Scene.polyzamboni_print_settings
    del bpy.types.Object.polyzamboni_object_prop