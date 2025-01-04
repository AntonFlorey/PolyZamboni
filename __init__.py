bl_info = {
    "name": "PolyZamboni",
    "author": "Anton Florey",
    "version": (1,0),
    "blender": (3,5,1),
    "location": "View3D",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh"
}

if "bpy" in locals():
    import importlib
    importlib.reload(locals()["ui"])
    importlib.reload(locals()["operators"])
    importlib.reload(locals()["properties"])
else:
    import bpy
    from .polyzamboni import pz_globals
    from .polyzamboni import drawing
    from .polyzamboni import ui
    from .polyzamboni import operators
    from .polyzamboni import properties
    from .polyzamboni import callbacks

def register():
    # Init globals
    pz_globals.init()
    # Properties first!
    properties.register()
    # Other stuff later
    operators.register()
    ui.register()
    bpy.app.handlers.load_post.append(callbacks.on_file_load)
    bpy.app.handlers.depsgraph_update_post.append(callbacks.on_object_select)

def unregister():
    operators.unregister()
    ui.unregister()
    properties.unregister()
    if callbacks.on_file_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(callbacks.on_file_load)
    if callbacks.on_object_select in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(callbacks.on_object_select)

if __name__ == "__main__":
    register()
    