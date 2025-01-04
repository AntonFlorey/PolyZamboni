import bpy
from . import pz_globals
from .drawing import update_all_polyzamboni_drawings
from . import cutgraph
from bpy.app.handlers import persistent

@persistent
def on_file_load(dummy):
    pz_globals.reset_file_dependent_globals()
    pz_globals.remove_all_existing_cutgraph_ids()

@persistent
def on_object_select(scene):
    print("something may happen now")
    if pz_globals.PZ_LOCK_SELECT_CALLBACK:
        print("nah it doesnt")
        return
    active_object = bpy.context.active_object
    is_mesh = active_object is not None and active_object.type == 'MESH'
    if not is_mesh:
        return
    if "manual_cuts" not in active_object:
        return
    pz_globals.PZ_LOCK_SELECT_CALLBACK = True
    print("entering select callback lock")
    if "cut_graph_id" not in active_object:
        print("initializing cutgraph from existing cuts")
        loaded_cutgraph = cutgraph.CutGraph(active_object)
        pz_globals.add_cutgraph(active_object, loaded_cutgraph)
        print("We now have", len(pz_globals.PZ_CUTGRAPHS), "cutgraphs")
    
    if pz_globals.PZ_CURRENT_CUTGRAPH_ID is None or pz_globals.PZ_CURRENT_CUTGRAPH_ID != active_object["cut_graph_id"]:
        print("set a different cut graph as active:", active_object["cut_graph_id"])
        pz_globals.PZ_CURRENT_CUTGRAPH_ID = active_object["cut_graph_id"]
        update_all_polyzamboni_drawings(None, bpy.context)

    pz_globals.PZ_LOCK_SELECT_CALLBACK = False