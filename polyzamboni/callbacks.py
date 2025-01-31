import bpy
from . import globals
from .drawing import update_all_polyzamboni_drawings, hide_all_drawings
from . import cutgraph
from bpy.app.handlers import persistent
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME, CUTGRAPH_ID_PROPERTY_NAME, BUILD_ORDER_PROPERTY_NAME, GLUE_FLAP_PROPERTY_NAME
import numpy as np

@persistent
def on_file_load(dummy):
    print("PolyZamboni is cleaning up after herself...")
    globals.reset_file_dependent_globals()
    globals.remove_all_existing_cutgraph_ids()
    print("Done.")

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
        loaded_cutgraph = cutgraph.CutGraph(active_object, 
                                            np.deg2rad(active_object.polyzamboni_object_prop.glue_flap_angle),
                                            active_object.polyzamboni_object_prop.glue_flap_height,
                                            active_object.polyzamboni_object_prop.prefer_alternating_flaps)
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

@persistent
def save_build_order(dummy):
    for obj in bpy.data.objects:
        if CUTGRAPH_ID_PROPERTY_NAME in obj:
            sparse_build_order_dict = globals.PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]].create_sparse_build_steps_dict()
            obj[BUILD_ORDER_PROPERTY_NAME] = sparse_build_order_dict


@persistent
def save_glue_flaps(dummy):
    for obj in bpy.data.objects:
        if CUTGRAPH_ID_PROPERTY_NAME in obj:
            glue_flaps_dict = globals.PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]].create_glue_flaps_dict()
            obj[GLUE_FLAP_PROPERTY_NAME] = glue_flaps_dict