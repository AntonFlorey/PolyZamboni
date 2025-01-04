import bpy

PZ_CUTGRAPHS = None
PZ_REMOVE_EXISTING_CUT_IDS = None
PZ_LOCK_SELECT_CALLBACK = False
PZ_CURRENT_CUTGRAPH_ID = None

def init():
    reset_file_dependent_globals()
    print("polyzamboni globals initialized")

def add_cutgraph(obj, cutgraph):
    global PZ_CUTGRAPHS    
    obj["cut_graph_id"] = len(PZ_CUTGRAPHS)
    obj["manual_cuts"] = cutgraph.get_manual_cuts_list()
    PZ_CUTGRAPHS.append(cutgraph)

def reset_file_dependent_globals():
    global PZ_CUTGRAPHS
    global PZ_REMOVE_EXISTING_CUT_IDS
    global PZ_CURRENT_CUTGRAPH_ID
    PZ_REMOVE_EXISTING_CUT_IDS = True
    PZ_CUTGRAPHS = []
    PZ_CURRENT_CUTGRAPH_ID = None

def remove_all_existing_cutgraph_ids():
    print("called the cut graph id removal function")
    global PZ_REMOVE_EXISTING_CUT_IDS
    if not PZ_REMOVE_EXISTING_CUT_IDS:
        return
    print("removing pre existing cut graph ids...")
    # remove all cut graph ids that might exist
    for obj in bpy.data.objects:
        if "cut_graph_id" in obj:
            print("deleting cut graph id from object:", obj.name)
            del obj["cut_graph_id"]
    PZ_REMOVE_EXISTING_CUT_IDS = False
    