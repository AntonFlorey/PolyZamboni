"""
Contains functions that extract all data to print from a mesh with attached paper model. 
Objects to draw are grouped in instances of ComponentPrintData.
Use "fit_components_on_pages" to place all components (or islands) on pages of a given size.
"""

import os
import bpy
from bpy.types import Object, Mesh
import bmesh
import numpy as np

from .geometry import AffineTransform2D, signed_point_dist_to_line
from . import glueflaps
from . import unfolding
from . import utils
from . import io

class CutEdgeData():
    def __init__(self, coords, edge_index):
        self.coords = coords
        self.edge_index = edge_index

class FoldEdgeData():
    def __init__(self, coords, convex, fold_angle):
        self.coords = coords
        self.is_convex = convex
        self.fold_angle = fold_angle

class FoldEdgeAtGlueFlapData():
    def __init__(self, coords, convex, fold_angle, edge_index):
        self.coords = coords
        self.is_convex = convex
        self.fold_angle = fold_angle
        self.edge_index = edge_index

class GlueFlapEdgeData():
    def __init__(self, coords):
        self.coords = coords

class ColoredTriangleData():

    def __init__(self, coords, uvs, abs_texture_path, color=None):
        self.coords = coords
        self.uvs = uvs
        self.absolute_texture_path = abs_texture_path # for textured faces 
        self.color = color # for monocolored faces

class ComponentPrintData():
    """ Instances of this class store all data necessary to print a connected component. """ 
    
    def __init__(self):
        self.lower_left = np.zeros(2) # post page transform bounding box
        self.upper_right = np.zeros(2) # post page transform bounding box
        self.cut_edges = []
        self.fold_edges = []
        self.fold_edges_at_flaps = []
        self.glue_flap_edges = []
        self.colored_triangles = []
        self.dominating_mat_index = 0 # we need this if different materials should be printed on different pages
        self.build_step_number = 0
        self.build_step_number_position = np.zeros(2)
        self.page_transform : AffineTransform2D = AffineTransform2D()
        pass
    
    def __update_bb_after_adding_edge(self, edge_coords):
        self.lower_left = np.minimum(self.lower_left, edge_coords[0])
        self.lower_left = np.minimum(self.lower_left, edge_coords[1])
        self.upper_right = np.maximum(self.upper_right, edge_coords[0])
        self.upper_right = np.maximum(self.upper_right, edge_coords[1])
    
    def add_cut_edge(self, cut_edge : CutEdgeData):
        self.cut_edges.append(cut_edge)
        self.__update_bb_after_adding_edge(cut_edge.coords)

    def add_fold_edge(self, fold_edge : FoldEdgeData):
        self.fold_edges.append(fold_edge)
        self.__update_bb_after_adding_edge(fold_edge.coords)

    def add_fold_edges_at_flaps(self, fold_edge_at_flap : FoldEdgeAtGlueFlapData):
        self.fold_edges_at_flaps.append(fold_edge_at_flap)
        self.__update_bb_after_adding_edge(fold_edge_at_flap.coords)

    def add_glue_flap_edge(self, glue_flap_edge : GlueFlapEdgeData):
        self.glue_flap_edges.append(glue_flap_edge)
        self.__update_bb_after_adding_edge(glue_flap_edge.coords)

    def add_texured_triangle(self, triangle_w_colordata : ColoredTriangleData):
        self.colored_triangles.append(triangle_w_colordata)

    def __all_edge_data(self):
        for cut_edge_data in self.cut_edges:
            yield cut_edge_data
        for fold_edge_data in self.fold_edges:
            yield fold_edge_data
        for fold_edge_at_flap_data in self.fold_edges_at_flaps:
            yield fold_edge_at_flap_data
        for glue_flap_edge_data in self.glue_flap_edges:
            yield glue_flap_edge_data

    def __pca(self):
        # collect all coords along cut edges (not glue flaps for now)
        cut_edge_first_coords = [cut_edge_data.coords[0] for cut_edge_data in self.cut_edges]
        fold_edge_at_glue_flap_first_coords = [fold_edge_at_flap_data.coords[0] for fold_edge_at_flap_data in self.fold_edges_at_flaps]
        boundary_coords = np.array(cut_edge_first_coords + fold_edge_at_glue_flap_first_coords)
        mean_shifted_coords = boundary_coords - np.mean(boundary_coords, axis=0)
        corr_mat = mean_shifted_coords.T @ mean_shifted_coords
        eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
        return np.mean(boundary_coords, axis=0), eigenvectors

    def align_horizontally_via_pca(self):
        self.apply_page_transform_to_all_coords()
        boundary_cog, pca_basis = self.__pca()
        shift_cog_to_orig = AffineTransform2D(affine_part=-boundary_cog)
        long_axis_eigvec = pca_basis[:,1] # eigenvector of largest eigenvalue
        short_axis_eigvec = np.array([-long_axis_eigvec[1], long_axis_eigvec[0]])
        PCA_B = np.column_stack([long_axis_eigvec, short_axis_eigvec])
        rotate_x_axis_to_long_axis = AffineTransform2D(linear_part=np.linalg.inv(PCA_B))
        horizontal_alignment_transform = shift_cog_to_orig.inverse() @ rotate_x_axis_to_long_axis @ shift_cog_to_orig
        self.set_new_page_transform(horizontal_alignment_transform)

    def align_vertically_via_pca(self):
        self.apply_page_transform_to_all_coords()
        boundary_cog, pca_basis = self.__pca()
        shift_cog_to_orig = AffineTransform2D(affine_part=-boundary_cog)
        long_axis_eigvec = pca_basis[:,1] # eigenvector of largest eigenvalue
        short_axis_eigvec = np.array([-long_axis_eigvec[1], long_axis_eigvec[0]])
        rotate_y_axis_to_long_axis = AffineTransform2D(linear_part=np.linalg.inv(np.column_stack([-short_axis_eigvec, long_axis_eigvec])))
        vertical_alignment_transform = shift_cog_to_orig.inverse() @ rotate_y_axis_to_long_axis @ shift_cog_to_orig
        self.set_new_page_transform(vertical_alignment_transform)

    def __adjust_bounding_box_to_page_trasform(self):
        for edge_data in self.__all_edge_data():
            transformed_edge = tuple([self.page_transform * coord for coord in edge_data.coords])
            self.__update_bb_after_adding_edge(transformed_edge)

    def set_new_page_transform(self, new_page_transform):
        self.page_transform = new_page_transform
        self.__adjust_bounding_box_to_page_trasform

    def concat_page_transform(self, latest_page_trasform):
        self.set_new_page_transform(latest_page_trasform @ self.page_transform)

    def apply_page_transform_to_all_coords(self):
        self.lower_left = np.inf * np.ones(2)
        self.upper_right = -np.inf * np.ones(2)

        # edges
        for edge_data in self.__all_edge_data():
            edge_data.coords = tuple([self.page_transform * coord for coord in edge_data.coords])
            self.__update_bb_after_adding_edge(edge_data.coords)

        # triangles
        for triangle_data in self.colored_triangles:
            triangle_data.coords = tuple([self.page_transform * coord for coord in triangle_data.coords])

        # build step number
        self.build_step_number_position = self.page_transform * self.build_step_number_position

        # set page transform to identity
        self.page_transform = AffineTransform2D()

class PagePartitionNode():
    def __init__(self, ll, ur):
        self.lower_left_corner = ll
        self.upper_right_corner = ur
        self.width = ur[0] - ll[0]
        self.height = ur[1] - ll[1]
        self.area = self.width * self.height
        self.contained_component_id = None
        self.child_one = None
        self.child_two = None

def compute_max_print_component_dimensions(obj : Object):
    mesh : Mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # read all polyzamboni data needed
    edge_constraints = io.read_edge_constraints_dict(mesh)
    connected_components = io.read_connected_component_sets(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    local_coordinate_systems = io.read_local_coordinate_systems_per_face(mesh)
    affine_transforms_to_root = io.read_affine_transforms_to_roots(mesh)
    glue_flap_triangles = io.read_glue_flap_geometry_per_edge_per_component(mesh)

    max_height = -np.inf
    max_width = -np.inf
    for c_id, curr_connected_component_faces in connected_components.items():
        assert len(curr_connected_component_faces) > 0
        # skip all components that cant be unfolded
        if c_id in cyclic_components:
            continue
        curr_component_print_data = ComponentPrintData()

        def get_unfolded_vertex_coord(coord, face_index):
            return unfolding.get_globally_consistent_2d_coord_in_face(mesh, coord, face_index, c_id, local_coordinate_systems, affine_transforms_to_root)

        # collect cut edges and fold edges
        for face_index in curr_connected_component_faces:
            curr_face : bmesh.types.BMFace = bm.faces[face_index]

            # collect all edges
            for curr_edge in curr_face.edges:
                # compute edge coords in unfolding space
                vertex_coords_3d = [v.co for v in curr_edge.verts]    
                vertex_coords_unfolded = [get_unfolded_vertex_coord(co_3d, face_index) for co_3d in vertex_coords_3d]
                if curr_edge.is_boundary or utils.mesh_edge_is_cut(curr_edge.index, edge_constraints):
                    if glueflaps.check_if_edge_has_flap_geometry_attached_to_it(mesh, c_id, curr_edge, glue_flap_triangles):
                        # this is a fold edge of a glue flap
                        curr_component_print_data.add_fold_edges_at_flaps(FoldEdgeAtGlueFlapData(tuple(vertex_coords_unfolded), curr_edge.is_convex, curr_edge.calc_face_angle(), curr_edge.index))
                    else:
                        # this is a cut edge
                        curr_component_print_data.add_cut_edge(CutEdgeData(tuple(vertex_coords_unfolded), curr_edge.index))

        # collect edges for glue flaps
        for flap_triangles in glue_flap_triangles[c_id].values():
            assert len(flap_triangles) > 0
            curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((flap_triangles[0][0],flap_triangles[0][1])))
            curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((flap_triangles[0][1],flap_triangles[0][2])))
            if len(flap_triangles) == 2:
                curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((flap_triangles[1][1],flap_triangles[1][2])))
        
        # align vertically 
        curr_component_print_data.align_vertically_via_pca()
        # get width and height
        max_width = max(curr_component_print_data.upper_right[0] - curr_component_print_data.lower_left[0], max_width)
        max_height = max(curr_component_print_data.upper_right[1] - curr_component_print_data.lower_left[1], max_height)

    bm.free()
    return max_width, max_height

def compute_scaling_factor_for_target_model_height(mesh, target_height):
    mesh_height = utils.compute_mesh_height(mesh)
    if mesh_height <= 0:
        print("POLYZAMBONI WARNING: Mesh has zero height!")
        return 1
    return target_height / mesh_height

def create_print_data_for_all_components(obj : Object, scaling_factor):
    mesh : Mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # read all polyzamboni data needed
    edge_constraints = io.read_edge_constraints_dict(mesh)
    connected_components = io.read_connected_component_sets(mesh)
    cyclic_components = io.read_components_with_cycles_set(mesh)
    local_coordinate_systems = io.read_local_coordinate_systems_per_face(mesh)
    affine_transforms_to_root = io.read_affine_transforms_to_roots(mesh)
    face_triangulations = io.read_triangulation_indices_per_face(mesh)
    unfolded_face_triangles = io.read_facewise_triangles_per_component(mesh)
    glue_flap_triangles = io.read_glue_flap_geometry_per_edge_per_component(mesh)
    build_step_numbers = io.read_build_step_numbers(mesh)

    all_print_data = [] 
    for c_id, curr_connected_component_faces in connected_components.items():
        assert len(curr_connected_component_faces) > 0
        # skip all components that cant be unfolded
        if c_id in cyclic_components:
            continue
        curr_component_print_data = ComponentPrintData()

        def get_unfolded_vertex_coord(coord, face_index):
            return unfolding.get_globally_consistent_2d_coord_in_face(mesh, coord, face_index, c_id, local_coordinate_systems, affine_transforms_to_root)

        # collect material index of first face
        first_face_in_component : bmesh.types.BMFace = bm.faces[iter(curr_connected_component_faces).__next__()]
        curr_component_print_data.dominating_mat_index = first_face_in_component.material_index

        # collect cut edges, fold edges and triangles of all faces
        fold_edge_index_set = set() # we need to remember fold edges to only draw them once
        face_cog_scores = {}
        for face_index in curr_connected_component_faces:
            curr_face : bmesh.types.BMFace = bm.faces[face_index]

            # map edges to correct halfedges
            edge_to_correct_halfedge_map = {}
            face_vertex_loop = list(curr_face.verts)
            for v_i in range(len(face_vertex_loop)):
                v_j = (v_i + 1) % len(face_vertex_loop)
                curr_e = bm.edges.get([face_vertex_loop[v_i], face_vertex_loop[v_j]])
                edge_to_correct_halfedge_map[curr_e.index] = (face_vertex_loop[v_i], face_vertex_loop[v_j])

            face_cog = np.mean([scaling_factor * get_unfolded_vertex_coord(v.co, face_index) for v in face_vertex_loop], axis=0)

            # collect all edges
            dist_cog_edge_sum = 0
            for curr_edge in curr_face.edges:
                # compute edge coords in unfolding space
                vertex_coords_3d = [v.co for v in edge_to_correct_halfedge_map[curr_edge.index]]    
                vertex_coords_unfolded = [scaling_factor * get_unfolded_vertex_coord(co_3d, face_index) for co_3d in vertex_coords_3d]
                dist_cog_edge_sum += signed_point_dist_to_line(face_cog, vertex_coords_unfolded[0], vertex_coords_unfolded[1])
                if curr_edge.is_boundary or utils.mesh_edge_is_cut(curr_edge.index, edge_constraints):
                    # check if this edge has a glue flap attached to it
                    if glueflaps.check_if_edge_has_flap_geometry_attached_to_it(mesh, c_id, curr_edge.index, glue_flap_triangles):
                        # this is a fold edge of a glue flap
                        curr_component_print_data.add_fold_edges_at_flaps(FoldEdgeAtGlueFlapData(tuple(vertex_coords_unfolded), curr_edge.is_convex, curr_edge.calc_face_angle(), curr_edge.index))
                    else:
                        # this is a cut edge
                        curr_component_print_data.add_cut_edge(CutEdgeData(tuple(vertex_coords_unfolded), curr_edge.index))
                elif curr_edge.index not in fold_edge_index_set:
                    # this is a fold edge
                    fold_edge_index_set.add(curr_edge.index)
                    curr_component_print_data.add_fold_edge(FoldEdgeData(tuple(vertex_coords_unfolded), curr_edge.is_convex, curr_edge.calc_face_angle()))
            # set cog score
            face_cog_scores[face_index] = dist_cog_edge_sum / len(curr_face.edges)

            # get face material info
            mat_slots = obj.material_slots
            color = None
            text_path = None
            if curr_face.material_index < len(mat_slots):
                curr_material_slot : bpy.types.MaterialSlot = mat_slots[curr_face.material_index]
                curr_material : bpy.types.Material = curr_material_slot.material
                color = curr_material.diffuse_color

                # try to get color and texture image path from node tree
                if curr_material.use_nodes:
                    for node in curr_material.node_tree.nodes:
                        if isinstance(node, bpy.types.ShaderNodeBsdfPrincipled):
                            color = np.array(node.inputs['Base Color'].default_value)
                        if not isinstance(node, bpy.types.ShaderNodeTexImage):
                            continue
                        if not node.image:
                            continue                            
                        full_path = bpy.path.abspath(node.image.filepath, library=node.image.library)
                        norm_path = os.path.normpath(full_path)
                        text_path = norm_path

            # collect all faces and apply scaling factor
            triangles_in_unfolding_space = [tuple(scaling_factor * np.asarray(tri_coords)) for tri_coords in unfolded_face_triangles[c_id][face_index]]
            
            # uvs
            uvs_available = len(mesh.uv_layers) > 0
            uv_layer = bm.loops.layers.uv.verify()
            v_id_to_uv_in_face = {}

            for loop in curr_face.loops:
                v_id_to_uv_in_face[loop.vert.index] = loop[uv_layer].uv

            triangle_uvs = [tuple([v_id_to_uv_in_face[v_id] for v_id in triangle_indices]) if uvs_available else None for triangle_indices in face_triangulations[face_index]]

            for tri_coords, tri_uv in zip(triangles_in_unfolding_space, triangle_uvs):
                curr_component_print_data.add_texured_triangle(ColoredTriangleData(tri_coords, tri_uv, text_path, color))

        # collect edges for glue flaps
        for flap_triangles in glue_flap_triangles[c_id].values():
            assert len(flap_triangles) > 0
            curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((scaling_factor * flap_triangles[0][0],scaling_factor * flap_triangles[0][1])))
            curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((scaling_factor * flap_triangles[0][1],scaling_factor * flap_triangles[0][2])))
            if len(flap_triangles) == 2:
                curr_component_print_data.add_glue_flap_edge(GlueFlapEdgeData((scaling_factor * flap_triangles[1][1],scaling_factor * flap_triangles[1][2])))

        face_with_step_number = list(sorted(curr_connected_component_faces, key=lambda face_index : face_cog_scores[face_index], reverse=True))[0] # sorting in the end is a bit meh but whatever
        step_number_pos = np.mean([scaling_factor * get_unfolded_vertex_coord(v.co, face_with_step_number) for v in bm.faces[face_with_step_number].verts], axis=0)

        curr_component_print_data.build_step_number_position = step_number_pos
        if c_id in build_step_numbers:
            curr_component_print_data.build_step_number = build_step_numbers[c_id]

        all_print_data.append(curr_component_print_data)
    
    bm.free()
    return all_print_data

def create_new_page_partition(page_size, page_margin):
    return PagePartitionNode(np.array([page_margin, page_margin]), np.maximum(0, np.array([page_size[0] - page_margin, page_size[1] - page_margin])))

def component_fits_in_page_part_node(component : ComponentPrintData, node : PagePartitionNode):
    comp_width = component.upper_right[0] - component.lower_left[0]
    comp_height = component.upper_right[1] - component.lower_left[1]

    return comp_width <= node.width and comp_height <= node.height

def partition_node_after_component_insertion(component : ComponentPrintData, node : PagePartitionNode, component_margin):
    comp_width = component.upper_right[0] - component.lower_left[0]
    comp_height = component.upper_right[1] - component.lower_left[1]

    if comp_width >= node.width and comp_height >= node.height:
        return # no splits
    
    # split along longer edge (I dont know what is better here tbh)
    if comp_height >= comp_width:
        b1_ll = node.lower_left_corner + np.array([comp_width + component_margin, 0])
        b1_ur = node.upper_right_corner
        if b1_ll[0] < b1_ur[0]:
            node.child_one = PagePartitionNode(b1_ll, b1_ur)
        b2_ll = node.lower_left_corner + np.array([0, comp_height + component_margin])
        b2_ur = np.array([node.lower_left_corner[0] + comp_width, node.upper_right_corner[1]])
        if b2_ll[0] < b2_ur[0] and b2_ll[1] < b2_ur[1]:
            node.child_two = PagePartitionNode(b2_ll, b2_ur)
    else:
        b1_ll = node.lower_left_corner + np.array([0 , comp_height + component_margin])
        b1_ur = node.upper_right_corner
        if b1_ll[1] < b1_ur[1]:
            node.child_one = PagePartitionNode(b1_ll, b1_ur)
        b2_ll = node.lower_left_corner + np.array([comp_width + component_margin, 0])
        b2_ur = np.array([node.upper_right_corner[0], node.lower_left_corner[1] + comp_height])
        if b2_ll[0] < b2_ur[0] and b2_ll[1] < b2_ur[1]:
            node.child_two = PagePartitionNode(b2_ll, b2_ur)

def recursive_search_for_all_free_spaces(component : ComponentPrintData, node : PagePartitionNode, free_spaces):
    if node is None:
        return
    if node.contained_component_id is None and component_fits_in_page_part_node(component, node):
        free_spaces.append(node)
    recursive_search_for_all_free_spaces(component, node.child_one, free_spaces)
    recursive_search_for_all_free_spaces(component, node.child_two, free_spaces)

def recursively_collect_all_page_components(node : PagePartitionNode, component_list, target_list):
    if node is None:
        return
    if node.contained_component_id is not None:
        component_in_node : ComponentPrintData = component_list[node.contained_component_id]
        translation =  AffineTransform2D(affine_part = node.lower_left_corner - component_in_node.lower_left)
        component_in_node.concat_page_transform(translation)
        target_list.append(component_in_node)
    recursively_collect_all_page_components(node.child_one, component_list, target_list)
    recursively_collect_all_page_components(node.child_two, component_list, target_list)

def fit_components_on_pages(components, page_size, page_margin, component_margin, different_materials_on_different_pages):
    """ This funciton returns a list of pages, each page containing a subset of the given components with translations attached to them"""
    
    some_material_index = 0
    if len(components) > 0:
        some_material_index = components[0].dominating_mat_index
    page_partitions = { some_material_index : [create_new_page_partition(page_size, page_margin)]}

    #preprocess components
    component_print_data : ComponentPrintData
    for component_print_data in components:
        component_print_data.align_vertically_via_pca()

    sorted_components = sorted(components, key = lambda c : (c.upper_right[0] - c.lower_left[0]) * (c.upper_right[1] - c.lower_left[1]), reverse=True)
    # go through all components ordered by their bounding box area
    for current_component_id, component_print_data in enumerate(sorted_components):
        component_mat_id = component_print_data.dominating_mat_index
        
        # search for any existing space for this component
        node_candidates = []
        for mat_id, page_list in page_partitions.items():
            if different_materials_on_different_pages and mat_id != component_print_data.dominating_mat_index:
                continue
            # recursively search for free space
            for page_root in page_list:
                recursive_search_for_all_free_spaces(component_print_data, page_root, node_candidates)

        if len(node_candidates) == 0:
            # align horizontally and try again
            component_print_data.align_horizontally_via_pca()
            for mat_id, page_list in page_partitions.items():
                if different_materials_on_different_pages and mat_id != component_print_data.dominating_mat_index:
                    continue
                # recursively search for free space
                for page_root in page_list:
                    recursive_search_for_all_free_spaces(component_print_data, page_root, node_candidates)

        if len(node_candidates) == 0:
            component_print_data.align_vertically_via_pca() # back to vertical alignment...
            # create a new page for this component
            new_page = create_new_page_partition(page_size, page_margin)
            page_partitions.setdefault(component_mat_id, []).append(new_page)
            # add component to the created page
            if not component_fits_in_page_part_node(component_print_data, new_page):
                print("POLYZAMBONI WARNING: Unfolded connected component does not fit on one page! ")
            new_page.contained_component_id = current_component_id
            partition_node_after_component_insertion(component_print_data, new_page, component_margin)
        else:
            smallest_viable_node : PagePartitionNode = list(sorted(node_candidates, key=lambda node : node.area))[0]
            smallest_viable_node.contained_component_id = current_component_id
            partition_node_after_component_insertion(component_print_data, smallest_viable_node, component_margin)

    # collect all components per page
    page_arrangement = []
    print("POLYZAMBONI INFO: Paper model instruction fits on", sum([len(roots) for roots in page_partitions.values()]), "pages.")
    for page_root_list in page_partitions.values():
        for page_root in page_root_list:
            components_on_curr_page = []
            recursively_collect_all_page_components(page_root, sorted_components, components_on_curr_page)
            page_arrangement.append(components_on_curr_page)

    # quick sanity check
    number_of_components_on_pages = sum([len(page) for page in page_arrangement])
    assert number_of_components_on_pages == len(components)

    return page_arrangement
