import bpy
from bpy.types import Mesh
from .drawing import update_all_polyzamboni_drawings, hide_all_drawings
from bpy.app.handlers import persistent

from .operators_backend import update_all_flap_geometry

# redraw polyzamboni feedback when object selection changes
def object_select_callback(*args):
    update_all_polyzamboni_drawings(None, bpy.context)

dummy_owner = object()

def subscribe_to_active_object():
    subscribe_to = bpy.types.LayerObjects, "active"
    bpy.msgbus.subscribe_rna(
        key = subscribe_to,
        owner = dummy_owner,
        args = tuple(),
        notify = object_select_callback,
        )

@persistent
def pre_load_handler(dummy):
    hide_all_drawings()

@persistent
def post_load_handler(dummy):
    subscribe_to_active_object()

def update_glueflap_geometry_callback(self, context : bpy.types.Context):
    ao = context.active_object
    if ao.type == 'MESH':
        active_mesh : Mesh = ao.data
        zamboni_props  = active_mesh.polyzamboni_general_mesh_props
        if zamboni_props.has_attached_paper_model:
            update_all_flap_geometry(active_mesh)
            update_all_polyzamboni_drawings(self, context)

@persistent
def redraw_callback(dummy):
    update_all_polyzamboni_drawings(None, bpy.context)