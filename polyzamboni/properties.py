import bpy
from bpy.types import Scene


# For more information about Blender Properties, visit:
# <https://blender.org/api/blender_python_api_2_78a_release/bpy.types.Property.html>
from bpy.props import BoolProperty
# from bpy.props import CollectionProperty
# from bpy.props import EnumProperty
# from bpy.props import FloatProperty
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
        default=100,
        min=0,
        max=1000
    )

# This is where you assign any variables you need in your script. Note that they
# won't always be assigned to the Scene object but it's a good place to start.
def register():
    bpy.utils.register_class(MySettings)
    Scene.polyzamboni_settings = bpy.props.PointerProperty(type=MySettings)

def unregister():
    bpy.utils.unregister_class(MySettings)
    del Scene.polyzamboni_settings