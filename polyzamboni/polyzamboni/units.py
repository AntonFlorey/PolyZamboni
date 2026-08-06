import bpy

def blender_distance_to_cm(distance, context : bpy.types.Context = None):
    if context is None:
        context = bpy.context
    return 100 * distance * context.scene.unit_settings.scale_length

def cm_to_blender_distance(length, context : bpy.types.Context = None):
    if context is None:
        context = bpy.context
    return length / (100 * context.scene.unit_settings.length_unit)