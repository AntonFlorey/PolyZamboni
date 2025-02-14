import numpy as np
from .geometry import AffineTransform2D

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
        self.lower_left = np.zeros(2)
        self.upper_right = np.zeros(2)
        self.cut_edges = []
        self.fold_edges = []
        self.fold_edges_at_flaps = []
        self.glue_flap_edges = []
        self.colored_triangles = []
        self.dominating_mat_index = 0 # we need this if different materials should be printed on different pages
        self.build_step_number = 0
        self.build_step_number_position = np.zeros(2)
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
        boundary_cog, pca_basis = self.__pca()
        shift_cog_to_orig = AffineTransform2D(affine_part=-boundary_cog)
        long_axis_eigvec = pca_basis[:,1] # eigenvector of largest eigenvalue
        short_axis_eigvec = np.array([-long_axis_eigvec[1], long_axis_eigvec[0]])
        PCA_B = np.column_stack([long_axis_eigvec, short_axis_eigvec])
        rotate_x_axis_to_long_axis = AffineTransform2D(linear_part=np.linalg.inv(PCA_B))
        horizontal_alignment_transform = shift_cog_to_orig.inverse() @ rotate_x_axis_to_long_axis @ shift_cog_to_orig
        self.apply_affine_transform_to_all_coords(horizontal_alignment_transform)

    def align_vertically_via_pca(self):
        boundary_cog, pca_basis = self.__pca()
        shift_cog_to_orig = AffineTransform2D(affine_part=-boundary_cog)
        long_axis_eigvec = pca_basis[:,1] # eigenvector of largest eigenvalue
        short_axis_eigvec = np.array([-long_axis_eigvec[1], long_axis_eigvec[0]])
        rotate_y_axis_to_long_axis = AffineTransform2D(linear_part=np.linalg.inv(np.column_stack([-short_axis_eigvec, long_axis_eigvec])))
        vertical_alignment_transform = shift_cog_to_orig.inverse() @ rotate_y_axis_to_long_axis @ shift_cog_to_orig
        self.apply_affine_transform_to_all_coords(vertical_alignment_transform)

    def apply_affine_transform_to_all_coords(self, affine_transform):
        self.lower_left = np.inf * np.ones(2)
        self.upper_right = -np.inf * np.ones(2)

        # edges
        for edge_data in self.__all_edge_data():
            edge_data.coords = tuple([affine_transform * coord for coord in edge_data.coords])
            self.__update_bb_after_adding_edge(edge_data.coords)

        # triangles
        for triangle_data in self.colored_triangles:
            triangle_data.coords = tuple([affine_transform * coord for coord in triangle_data.coords])

        # build step number
        self.build_step_number_position = affine_transform * self.build_step_number_position

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
    
    # split along longer edge (I dont know what is better here)
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
        target_list.append({"print data" : component_in_node, "page coord transform" : translation})
    recursively_collect_all_page_components(node.child_one, component_list, target_list)
    recursively_collect_all_page_components(node.child_two, component_list, target_list)

def fit_components_on_pages(components, page_size, page_margin, component_margin, different_materials_on_different_pages):
    """ This funciton returns a list of pages, each page containing a subset of the given components with translations attached to them"""

    page_partitions = { 0 : [create_new_page_partition(page_size, page_margin)]}

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
            print("had to begin a new page for component")
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
    print("Unfolding fits on", sum([len(roots) for roots in page_partitions.values()]), "pages.")
    for page_root_list in page_partitions.values():
        for page_root in page_root_list:
            components_on_curr_page = []
            recursively_collect_all_page_components(page_root, sorted_components, components_on_curr_page)
            page_arrangement.append(components_on_curr_page)

    # quick sanity check
    number_of_components_on_pages = sum([len(page) for page in page_arrangement])
    print("Number of components on all pages:", number_of_components_on_pages)
    print("Number of input components:", len(components))
    assert number_of_components_on_pages == len(components)

    return page_arrangement
