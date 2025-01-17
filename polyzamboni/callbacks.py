import bpy
from . import globals
from .drawing import update_all_polyzamboni_drawings, hide_all_drawings
from . import cutgraph
from bpy.app.handlers import persistent
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME, CUTGRAPH_ID_PROPERTY_NAME

@persistent
def on_file_load(dummy):
    globals.reset_file_dependent_globals()
    globals.remove_all_existing_cutgraph_ids()

@persistent
def on_object_select(scene):
    # print("something may happen now")
    if globals.PZ_LOCK_SELECT_CALLBACK:
        # print("nah it doesnt")
        return
    any_objects_selected = len(bpy.context.selected_objects) != 0
    active_object = bpy.context.active_object
    is_mesh = active_object is not None and active_object.type == 'MESH'
    in_edit_mode = is_mesh and active_object.mode == "EDIT"

    if not any_objects_selected and not in_edit_mode:
        globals.PZ_CURRENT_CUTGRAPH_ID = None
        hide_all_drawings()
        return
    if not is_mesh:
        globals.PZ_CURRENT_CUTGRAPH_ID = None
        hide_all_drawings()
        return
    if CUT_CONSTRAINTS_PROP_NAME not in active_object and LOCKED_EDGES_PROP_NAME not in active_object:
        globals.PZ_CURRENT_CUTGRAPH_ID = None
        hide_all_drawings()
        #print("no preexisting cutgraph detected")
        return
    globals.PZ_LOCK_SELECT_CALLBACK = True
    # print("entering select callback lock")
    if CUTGRAPH_ID_PROPERTY_NAME not in active_object:
        # print("initializing cutgraph from existing cuts")
        loaded_cutgraph = cutgraph.CutGraph(active_object)
        globals.add_cutgraph(active_object, loaded_cutgraph)
        # print("We now have", len(pz_globals.PZ_CUTGRAPHS), "cutgraphs")
    
    if globals.PZ_CURRENT_CUTGRAPH_ID is None or globals.PZ_CURRENT_CUTGRAPH_ID != active_object[CUTGRAPH_ID_PROPERTY_NAME]:
        # print("set a different cut graph as active:", active_object["cut_graph_id"])
        globals.PZ_CURRENT_CUTGRAPH_ID = active_object[CUTGRAPH_ID_PROPERTY_NAME]
        update_all_polyzamboni_drawings(None, bpy.context)
    # if in_edit_mode:
    #     update_all_polyzamboni_drawings(None, bpy.context)
    globals.PZ_LOCK_SELECT_CALLBACK = False

@persistent
def save_all_edge_constraints(dummy):
    for obj in bpy.data.objects:
        if CUTGRAPH_ID_PROPERTY_NAME in obj:
            obj[CUT_CONSTRAINTS_PROP_NAME] = globals.PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]].get_manual_cuts_list()
            obj[LOCKED_EDGES_PROP_NAME] = globals.PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]].get_locked_edges_list()