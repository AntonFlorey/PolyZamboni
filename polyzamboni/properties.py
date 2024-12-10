import bpy
from bpy.types import Scene


# For more information about Blender Properties, visit:
# <https://blender.org/api/blender_python_api_2_78a_release/bpy.types.Property.html>
from bpy.props import BoolProperty
# from bpy.props import CollectionProperty
# from bpy.props import EnumProperty
from bpy.props import FloatProperty
from bpy.props import IntProperty
# from bpy.props import PointerProperty
# from bpy.props import StringProperty
# from bpy.props import PropertyGroup

#
# Add additional functions or classes here
#

class MySettings(bpy.types.PropertyGroup):
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

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(MySettings)
    Scene.polyzamboni_settings = bpy.props.PointerProperty(type=MySettings)

def unregister():
    bpy.utils.unregister_class(MySettings)
    del Scene.polyzamboni_settings