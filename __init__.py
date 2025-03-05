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
    importlib.reload(locals()["callbacks"])
else:
    import bpy
    from .polyzamboni import properties
    from .polyzamboni import drawing
    from .polyzamboni import operators
    from .polyzamboni import ui
    from .polyzamboni import callbacks

def register():
    # Properties first!
    properties.register()
    # Other stuff later
    operators.register()
    ui.register()
    bpy.app.handlers.save_pre.append(callbacks.compactify_before_saving)
    bpy.app.handlers.load_post.append(callbacks.post_load_handler)
    bpy.app.handlers.undo_post.append(callbacks.redraw_callback)

def unregister():
    drawing.hide_all_drawings()
    operators.unregister()
    ui.unregister()
    properties.unregister()
    if callbacks.compactify_before_saving in bpy.app.handlers.save_pre:
        bpy.app.handlers.save_pre.remove(callbacks.compactify_before_saving)
    if callbacks.post_load_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(callbacks.post_load_handler)
    if callbacks.redraw_callback in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(callbacks.redraw_callback)

if __name__ == "__main__":
    register()
    