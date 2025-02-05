import bpy
from . import globals
from .drawing import update_all_polyzamboni_drawings, hide_all_drawings
from . import cutgraph
from bpy.app.handlers import persistent
from .constants import LOCKED_EDGES_PROP_NAME, CUT_CONSTRAINTS_PROP_NAME, CUTGRAPH_ID_PROPERTY_NAME, BUILD_ORDER_PROPERTY_NAME, GLUE_FLAP_PROPERTY_NAME, AUTO_CUT_EDGES_PROP_NAME
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
                                            active_object.polyzamboni_object_prop.glue_flap_angle,
                                            active_object.polyzamboni_object_prop.glue_flap_height,
                                            active_object.polyzamboni_object_prop.prefer_alternating_flaps,
                                            active_object.polyzamboni_object_prop.apply_auto_cuts_to_previev)
        globals.add_cutgraph(active_object, loaded_cutgraph)
        # print("We now have", len(pz_globals.PZ_CUTGRAPHS), "cutgraphs")
    
    if globals.PZ_CURRENT_CUTGRAPH_ID is None or globals.PZ_CURRENT_CUTGRAPH_ID != active_object[CUTGRAPH_ID_PROPERTY_NAME]:
        # print("set a different cut graph as active:", active_object["cut_graph_id"])
        globals.PZ_CURRENT_CUTGRAPH_ID = active_object[CUTGRAPH_ID_PROPERTY_NAME]
        update_all_polyzamboni_drawings(None, bpy.context)
    # if in_edit_mode:
    #     update_all_polyzamboni_drawings(None, bpy.context)
    globals.PZ_LOCK_SELECT_CALLBACK = False

def get_active_cutgraph_and_pz_settings(context):
    ao = context.active_object
    if CUTGRAPH_ID_PROPERTY_NAME not in ao:
        return None, None
    return globals.PZ_CUTGRAPHS[ao[CUTGRAPH_ID_PROPERTY_NAME]], ao.polyzamboni_object_prop

def update_flap_angle_callback(self, context : bpy.types.Context):
    ao_cutgraph, ao_pz_prop = get_active_cutgraph_and_pz_settings(context)
    if ao_cutgraph is None:
        return
    ao_cutgraph.flap_angle = ao_pz_prop.glue_flap_angle
    ao_cutgraph.update_all_flap_geometry()
    update_all_polyzamboni_drawings(self, context)

def update_flap_height_callback(self, context : bpy.types.Context):
    ao_cutgraph, ao_pz_prop = get_active_cutgraph_and_pz_settings(context)
    if ao_cutgraph is None:
        return
    ao_cutgraph.flap_height = ao_pz_prop.glue_flap_height
    ao_cutgraph.update_all_flap_geometry()
    update_all_polyzamboni_drawings(self, context)

def update_auto_cuts_usage_callback(self, context : bpy.types.Context):
    ao_cutgraph : cutgraph.CutGraph
    ao_cutgraph, ao_pz_prop = get_active_cutgraph_and_pz_settings(context)
    if ao_cutgraph is None:
        return
    ao_cutgraph.use_auto_cuts = ao_pz_prop.apply_auto_cuts_to_previev
    ao_cutgraph.compute_all_connected_components()
    ao_cutgraph.unfold_all_connected_components()
    ao_cutgraph.greedy_place_all_flaps()
    update_all_polyzamboni_drawings(self, context)

@persistent
def save_cutgraph_data(dummy):
    for obj in bpy.data.objects:
        if CUTGRAPH_ID_PROPERTY_NAME in obj:
            curr_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]] 
            obj[CUT_CONSTRAINTS_PROP_NAME] = curr_cutgraph.get_manual_cuts_list()
            obj[LOCKED_EDGES_PROP_NAME] = curr_cutgraph.get_locked_edges_list()
            obj[AUTO_CUT_EDGES_PROP_NAME] = curr_cutgraph.get_auto_cuts_list()
            sparse_build_order_dict = curr_cutgraph.create_sparse_build_steps_dict()
            obj[BUILD_ORDER_PROPERTY_NAME] = sparse_build_order_dict
            glue_flaps_dict = curr_cutgraph.create_glue_flaps_dict()
            obj[GLUE_FLAP_PROPERTY_NAME] = glue_flaps_dict
