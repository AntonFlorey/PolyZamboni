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

def register():
    # Init globals
    pz_globals.init()
    # Properties first!
    properties.register()
    # Other stuff later
    operators.register()
    ui.register()
    
def unregister():
    operators.unregister()
    ui.unregister()
    properties.unregister()

if __name__ == "__main__":
    register()