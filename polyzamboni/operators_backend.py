"""
This file contains several high level functions that meaningful combine multiple polyzamboni operations.
"""

from bpy.types import Mesh, Object
from bmesh.types import BMesh
from enum import Enum
import math

from . import geometry
from . import io
from .papermodel import PaperModel
from .printprepper import create_print_data_for_all_components, fit_components_on_pages, ComponentPrintData
from .exporters import paper_sizes

#################################
#    Init, delete and update    #
#################################

def get_indices_of_multi_touching_faces(bm : BMesh):
    face_pair_set = set()
    multi_touching_face_indices = set()
    for edge in bm.edges:
        if edge.is_boundary:
            continue
        assert len(edge.link_faces) == 2
        face_pair_key = tuple(sorted([f.index for f in edge.link_faces]))
        if face_pair_key in face_pair_set:
            for f in edge.link_faces:
                multi_touching_face_indices.add(f.index)
        face_pair_set.add(face_pair_key)
    return multi_touching_face_indices

def get_indices_of_not_triangulatable_faces(bm : BMesh):
    non_triangulatable_faces = set()
    for face in bm.faces:
        _, tri_ids = geometry.triangulate_3d_polygon([v.co for v in face.verts], face.normal)
        if len(tri_ids) != len(face.verts) - 2:
            non_triangulatable_faces.add(face.index)
    return non_triangulatable_faces

def initialize_paper_model(mesh : Mesh):
    new_papermodel = PaperModel.new_from_mesh(mesh)
    new_papermodel.close()

def delete_paper_model(mesh : Mesh):
    zamboni_props = mesh.polyzamboni_general_mesh_props
    zamboni_props.has_attached_paper_model = False
    io.remove_all_polyzamboni_data(mesh)

def sync_paper_model_with_mesh_geometry(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.apply_mesh_geometry_changes()

#################################
#   Cut, clear and glue edges   #
#################################

def cut_edges(mesh : Mesh, indices_of_edges_to_cut):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.cut_edges(indices_of_edges_to_cut)

def glue_edges(mesh : Mesh, indices_of_edges_to_lock):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.glue_edges(indices_of_edges_to_lock)

def clear_edges(mesh : Mesh, indices_of_edges_to_clear):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.clear_edges(indices_of_edges_to_clear)

def add_cutout_region(mesh : Mesh, face_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.cut_out_face_region(face_indices)

def add_cuts_between_different_materials(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.add_cuts_between_different_materials()

def remove_auto_cuts(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.remove_all_auto_cuts()

#################################
#           Glue Flaps          #
#################################

def update_all_flap_geometry(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.update_all_flap_geometry()

def flip_glue_flaps(mesh : Mesh, edge_indices):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.flip_glue_flaps_around_edges(edge_indices)

def recompute_all_glue_flaps(mesh : Mesh):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.compute_all_glueflaps_greedily()

#################################
#           Build Steps         #
#################################

def compute_build_step_numbers(mesh : Mesh, selected_start_face_ids):
    with PaperModel.from_existing(mesh) as papermodel:
        papermodel.compute_build_step_numbers(selected_start_face_ids)

#################################
#      Page Layout Editing      #
#################################

def compute_and_save_page_layout(obj : Object, scaling_factor, paper_size, page_margin, component_margin, separate_materials):
    page_size = paper_sizes[paper_size]
    component_print_data = create_print_data_for_all_components(obj, scaling_factor)
    print_data_on_pages = fit_components_on_pages(component_print_data, page_size, page_margin, component_margin, separate_materials)
    # store computed layout
    mesh = obj.data
    general_props = mesh.polyzamboni_general_mesh_props
    general_props.model_scale = scaling_factor
    general_props.paper_size = paper_size
    page_numbers_per_component = {}
    page_transforms_per_component = {}
    for page_index, component_print_data_on_page in enumerate(print_data_on_pages):
        current_print_data : ComponentPrintData
        for current_print_data in component_print_data_on_page:
            current_c_id = current_print_data.og_component_id
            page_numbers_per_component[current_c_id] = page_index
            page_transforms_per_component[current_c_id] = current_print_data.page_transform
    io.write_page_numbers(mesh, page_numbers_per_component)
    io.write_page_transforms(mesh, page_transforms_per_component)

class PageEditorState(Enum):
    SELECT_PIECES = 0
    MOVE_PIECE = 1
    ROTATE_PIECE = 2

def find_page_under_mouse_position(pos_x, pos_y, pages, paper_size, margin_between_pages = 1.0, pages_per_row = 2):
    row_grid_size = paper_size[0] + margin_between_pages
    row_index = math.floor(pos_x / row_grid_size)
    if pos_x < 0:
        row_index -= 1
    if row_index < 0 or row_index >= pages_per_row or pos_x - row_index * row_grid_size > paper_size[0]:
        return None
    transformed_pos_y = -(pos_y - paper_size[1])
    col_grid_size = paper_size[1] + margin_between_pages
    col_index = math.floor(transformed_pos_y / col_grid_size)
    if transformed_pos_y < 0:
        col_index -= 1
    if col_index < 0 or transformed_pos_y - col_index * col_grid_size > paper_size[1]:
        return None
    # compute page index
    hovered_page_index = pages_per_row * col_index + row_index
    if hovered_page_index <= pages:
        return hovered_page_index
    else:
        return None
