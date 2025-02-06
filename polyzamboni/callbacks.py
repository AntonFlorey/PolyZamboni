import bpy
import bmesh
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

def try_to_load_cutgraph(obj : bpy.types.Object):
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh : bmesh.types.BMesh = bmesh.new()
    mesh.from_mesh(obj.data)

    selected_mesh_is_manifold = np.all([edge.is_manifold or edge.is_boundary for edge in mesh.edges] + [v.is_manifold for v in mesh.verts])
    normals_are_okay = np.all([edge.is_contiguous or edge.is_boundary for edge in mesh.edges])
    
    if not (selected_mesh_is_manifold and normals_are_okay):
        print("POLYZAMBONI WARNING: Cutgraph attached to an invalid mesh detected! (non manifold or bad normals)")
        globals.remove_cutgraph(obj)
        return

    double_connected_face_pair_present = False
    face_pair_set = set()
    for edge in mesh.edges:
        if edge.is_boundary:
            continue
        assert len(edge.link_faces) == 2
        if tuple(sorted([f.index for f in edge.link_faces])) in face_pair_set:
            double_connected_face_pair_present = True
            break
        face_pair_set.add(tuple(sorted([f.index for f in edge.link_faces])))
    
    if selected_mesh_is_manifold and normals_are_okay and not double_connected_face_pair_present:
        loaded_cutgraph = cutgraph.CutGraph(obj, 
                                            obj.polyzamboni_object_prop.glue_flap_angle,
                                            obj.polyzamboni_object_prop.glue_flap_height,
                                            obj.polyzamboni_object_prop.prefer_alternating_flaps,
                                            obj.polyzamboni_object_prop.apply_auto_cuts_to_previev)
        globals.add_cutgraph(obj, loaded_cutgraph)
        return
    
    print("POLYZAMBONI WARNING: Cutgraph attached to an invalid mesh detected! (Multi touching faces)")
    globals.remove_cutgraph(obj)

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
        try_to_load_cutgraph(active_object)
        # print("We now have", len(pz_globals.PZ_CUTGRAPHS), "cutgraphs")
    
    if CUTGRAPH_ID_PROPERTY_NAME in active_object and (globals.PZ_CURRENT_CUTGRAPH_ID is None or globals.PZ_CURRENT_CUTGRAPH_ID != active_object[CUTGRAPH_ID_PROPERTY_NAME]):
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

def save_data_of_object(obj):
    if CUTGRAPH_ID_PROPERTY_NAME in obj:
        curr_cutgraph : cutgraph.CutGraph = globals.PZ_CUTGRAPHS[obj[CUTGRAPH_ID_PROPERTY_NAME]] 
        obj[CUT_CONSTRAINTS_PROP_NAME] = curr_cutgraph.get_manual_cuts_list()
        obj[LOCKED_EDGES_PROP_NAME] = curr_cutgraph.get_locked_edges_list()
        obj[AUTO_CUT_EDGES_PROP_NAME] = curr_cutgraph.get_auto_cuts_list()
        sparse_build_order_dict = curr_cutgraph.create_sparse_build_steps_dict()
        obj[BUILD_ORDER_PROPERTY_NAME] = sparse_build_order_dict
        glue_flaps_dict = curr_cutgraph.create_glue_flaps_dict()
        obj[GLUE_FLAP_PROPERTY_NAME] = glue_flaps_dict

def load_cutgraph_from_data(obj):
    if CUTGRAPH_ID_PROPERTY_NAME in obj:
        obj_pz_settings = obj.polyzamboni_object_prop
        new_cutgraph = cutgraph.CutGraph(obj, 
                                         obj_pz_settings.glue_flap_angle, 
                                         obj_pz_settings.glue_flap_height, 
                                         obj_pz_settings.prefer_alternating_flaps,
                                         obj_pz_settings.apply_auto_cuts_to_previev)
        globals.reset_cutgraph(obj, new_cutgraph)

@persistent
def save_cutgraph_data(dummy):
    for obj in bpy.data.objects:
        save_data_of_object(obj)

@persistent
def reload_active_cutgraph_data(dummy):
    print("reloading from property data. object:", bpy.context.active_object.name)
    load_cutgraph_from_data(bpy.context.active_object)
    update_all_polyzamboni_drawings(None, bpy.context)

@persistent
def refresh_drawings(dummy):
    print(dummy)
    update_all_polyzamboni_drawings(None, bpy.context)