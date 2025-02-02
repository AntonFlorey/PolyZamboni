import bpy
from .constants import CUT_CONSTRAINTS_PROP_NAME, LOCKED_EDGES_PROP_NAME, CUTGRAPH_ID_PROPERTY_NAME, GLUE_FLAP_PROPERTY_NAME, BUILD_ORDER_PROPERTY_NAME

PZ_CUTGRAPHS = None
PZ_LOCK_SELECT_CALLBACK = False
PZ_CURRENT_CUTGRAPH_ID = None

def init():
    reset_file_dependent_globals()
    print("polyzamboni globals initialized")

def add_cutgraph(obj, cutgraph):
    global PZ_CUTGRAPHS    
    obj[CUTGRAPH_ID_PROPERTY_NAME] = len(PZ_CUTGRAPHS)
    obj[CUT_CONSTRAINTS_PROP_NAME] = cutgraph.get_manual_cuts_list()
    obj[LOCKED_EDGES_PROP_NAME] = cutgraph.get_locked_edges_list()
    PZ_CUTGRAPHS.append(cutgraph)

def reset_cutgraph(obj, cutgraph):
    if CUTGRAPH_ID_PROPERTY_NAME not in obj:
        print("WARNING! Tried to reset a cutgraph that does not exist")
        add_cutgraph(obj, cutgraph)
        return
    PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]] = cutgraph

def remove_cutgraph(obj):
    global PZ_CURRENT_CUTGRAPH_ID
    if CUTGRAPH_ID_PROPERTY_NAME not in obj:
        print("WARNING! Tried to remove a cutgraph that does not exist")
        return
    PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]] = None # remove cutgraph from list
    if obj[CUTGRAPH_ID_PROPERTY_NAME] == PZ_CURRENT_CUTGRAPH_ID:
        PZ_CURRENT_CUTGRAPH_ID = None
    del obj[CUTGRAPH_ID_PROPERTY_NAME]
    if CUT_CONSTRAINTS_PROP_NAME in obj:
        del obj[CUT_CONSTRAINTS_PROP_NAME]
    if LOCKED_EDGES_PROP_NAME in obj:
        del obj[LOCKED_EDGES_PROP_NAME]
    if GLUE_FLAP_PROPERTY_NAME in obj:
        del obj[GLUE_FLAP_PROPERTY_NAME]
    if BUILD_ORDER_PROPERTY_NAME in obj:
        del obj[BUILD_ORDER_PROPERTY_NAME]

def reset_file_dependent_globals():
    global PZ_CUTGRAPHS
    global PZ_CURRENT_CUTGRAPH_ID
    global PZ_LOCK_SELECT_CALLBACK
    PZ_CUTGRAPHS = []
    PZ_CURRENT_CUTGRAPH_ID = None
    PZ_LOCK_SELECT_CALLBACK = False

def remove_all_existing_cutgraph_ids():
    print("removing pre existing cut graph ids...")
    # remove all cut graph ids that might exist
    for obj in bpy.data.objects:
        if CUTGRAPH_ID_PROPERTY_NAME in obj:
            print("deleting cut graph id from object:", obj.name)
            del obj[CUTGRAPH_ID_PROPERTY_NAME]
    