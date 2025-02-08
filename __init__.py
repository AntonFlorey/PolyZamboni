bl_info = {
    "name": "PolyZamboni",
    "author": "Anton Florey",
    "version": (1,0),
    "blender": (4,1,0),
    "location": "View3D",
    "warning": "",
    "wiki_url": "",
    "category": "Import-Export"
}

if "bpy" in locals():
    import importlib
    importlib.reload(locals()["ui"])
    importlib.reload(locals()["operators"])
    importlib.reload(locals()["properties"])
    importlib.reload(locals()["drawing"])
    importlib.reload(locals()["globals"])
    importlib.reload(locals()["callbacks"])
else:
    import bpy
    from .polyzamboni import properties
    from .polyzamboni import globals
    from .polyzamboni import drawing
    from .polyzamboni import operators
    from .polyzamboni import ui
    from .polyzamboni import callbacks

def register():
    # Init globals
    globals.init()
    # Properties first!
    properties.register()
    # Other stuff later
    operators.register()
    ui.register()
    bpy.app.handlers.load_post.append(callbacks.on_file_load)
    bpy.app.handlers.depsgraph_update_post.append(callbacks.on_object_select)
    bpy.app.handlers.save_pre.append(callbacks.save_cutgraph_data)
    # bpy.app.handlers.undo_post.append(callbacks.refresh_drawings)

def unregister():
    print("PolyZamboni is cleaning up after itself...")
    globals.remove_all_existing_cutgraph_ids()
    drawing.hide_all_drawings()
    operators.unregister()
    ui.unregister()
    properties.unregister()
    if callbacks.on_file_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(callbacks.on_file_load)
    if callbacks.on_object_select in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(callbacks.on_object_select)
    if callbacks.save_cutgraph_data in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(callbacks.save_cutgraph_data)
    # if callbacks.refresh_drawings in bpy.app.handlers.undo_post:
    #     bpy.app.handlers.undo_post.remove(callbacks.refresh_drawings)
    print("Done.")

if __name__ == "__main__":
    register()
    