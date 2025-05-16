"""
Custom geometry functions used to compute triangulation, do intersection tests and map from 3D to 2D local spaces and back.
"""

import numpy as np
import bmesh
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations

def draw_polygon(ax, vertices, number):
    polygon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='none')
    ax.add_patch(polygon)

    # Calculate the centroid of the polygon
    center_x = np.mean([v[0] for v in vertices])
    center_y = np.mean([v[1] for v in vertices])

    ax.text(center_x, center_y, str(number), color='black', ha='center', va='center', fontsize=8)

def debug_draw_polygons_2d(polygons_2d, filename):
    fig, ax = plt.subplots()

    min_x = min([min([p[0] for p in polygon]) for polygon in polygons_2d])
    min_y = min([min([p[1] for p in polygon]) for polygon in polygons_2d])
    max_x = max([max([p[0] for p in polygon]) for polygon in polygons_2d])
    max_y = max([max([p[1] for p in polygon]) for polygon in polygons_2d])

    _max = max(max_x, max_y)
    _min = min(min_x, min_y)

    ax.set_xlim(_min-0.5, _max+0.5)
    ax.set_ylim(_min-0.5, _max+0.5)

    for i, polygon in enumerate(polygons_2d):
        draw_polygon(ax, polygon, i)

    plt.axis('equal')
    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')
    plt.title('Unfolded Polygons')
    plt.grid(False)
    plt.savefig(filename)
    plt.close(fig)

def debug_draw_component_outline_with_selected_edge(lines_2d, selected_line, filename):
    fig, ax = plt.subplots()

    for line in lines_2d:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color="blue")
    ax.plot([selected_line[0][0], selected_line[1][0]], [selected_line[0][1], selected_line[1][1]], color="red")

    plt.axis('equal')
    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')
    plt.title('Connected component boundary with selected edge')
    plt.grid(False)
    plt.savefig(filename)
    plt.close(fig)


class AffineTransform2D():
    """ Affine Transformation in 2D Space """
    def __init__(self, linear_part = np.eye(2), affine_part = np.zeros(2)):
        self.A = linear_part
        self.t = affine_part

    def inverse(self):
        A_inv = np.linalg.inv(self.A)
        return AffineTransform2D(A_inv, -np.dot(A_inv, self.t))

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return np.dot(self.A, other) + self.t
        else:
            raise NotImplementedError("Cant multiply affine 2D transformation with", other)

    def __matmul__(self, other):
        if isinstance(other, AffineTransform2D):
            return AffineTransform2D(self.A @ other.A, np.dot(self.A, other.t) + self.t)
        else:
            raise NotImplementedError("Cant multiply affine 2D transformation with", other)

    def to_numpy_array(self):
        mat = np.eye(3)
        mat[:2,:2] = self.A
        mat[:2,2] = self.t
        return mat

    def __str__(self):
        return "2D Affine Transformation\nLinear Part:\n" + self.A.__str__() + "\nAffine Part:\n" + self.t.__str__()

def compute_bb_diameter(mesh : bmesh.types.BMesh):
    positions = np.asarray([np.asarray(v.co) for v in mesh.verts])
    mincorner = np.min(positions, axis=0)
    maxcorner = np.max(positions, axis=0)
    
    return np.linalg.norm(maxcorner - mincorner)

def compute_mesh_surface_area(mesh : bmesh.types.BMesh):
    return sum([f.calc_area() for f in mesh.faces])

def compute_voronoi_areas(mesh : bmesh.types.BMesh):
    voronoi_areas = {}
    face_areas = {}
    for f in mesh.faces:
        face_areas[f.index] = f.calc_area()
    for v in mesh.verts:
        voronoi_areas[v.index] = sum([face_areas[f.index] / len(f.verts) for f in v.link_faces])

    assert np.allclose(sum(face_areas.values()), sum(voronoi_areas.values()))

    return voronoi_areas

def compute_planarity_score(face_coords):
    n = len(face_coords)
    normalized_edges = []
    planarity_score = 0
    for i in range(n):
        v_a = face_coords[i]
        v_b = face_coords[(i + 1) % n]
        normalized_edges.append((v_b - v_a) / np.linalg.norm(v_b - v_a))
    for a, b, c in combinations(normalized_edges, 3):
        planarity_score += np.linalg.det(np.stack((a,b,c))) ** 2
    return planarity_score * 6 / (n * (n - 1) * (n - 2))

def to_local_coords(point_in_3d, frame_orig, base_x, base_y):
    relative_coord = np.asarray(point_in_3d) - frame_orig
    return np.array([np.dot(base_x, relative_coord), np.dot(base_y, relative_coord)])

def to_world_coords(point_in_2d, frame_orig, base_x, base_y):
    return frame_orig + point_in_2d[0] * base_x + point_in_2d[1] * base_y

def construct_orthogonal_basis_at_2d_edge(v_2d_from, v_2d_to):
    x_ax = np.array(v_2d_to - v_2d_from)
    x_ax = x_ax / np.linalg.norm(x_ax)
    y_ax = np.array([-x_ax[1], x_ax[0]]) # rotation by 90 degrees
    return x_ax, y_ax

def construct_2d_space_along_face_edge(v_3d_from, v_3d_to, n_3d):
    x_ax = np.array(v_3d_to - v_3d_from)
    x_ax = x_ax / np.linalg.norm(x_ax)
    n = np.array(n_3d)
    y_ax = np.cross(n, x_ax)
    y_ax = y_ax / np.linalg.norm(y_ax)
    orig = np.array(v_3d_from)
    return (orig, x_ax, y_ax)

def compute_angle_between_2d_vectors_atan2(vec_1, vec_2):
    v1, v1_orth = construct_orthogonal_basis_at_2d_edge(np.zeros(2), vec_1)
    v2 = np.array(vec_2)
    v2 = v2 / np.linalg.norm(v2)
    # rotate v1 to (1,0) and apply the same rotation to v2
    v2_rot = np.linalg.inv(np.array([v1, v1_orth]).T) @ v2
    return np.arctan2(v2_rot[1], v2_rot[0])

def affine_2d_transformation_between_two_2d_spaces_on_same_plane(space_a, space_b):
    """ Returns a 2d affine transformation that maps from space b to space a """
    orig_b_in_space_a = to_local_coords(space_b[0], *space_a)
    x_ax_b_in_space_a = to_local_coords(space_a[0] + space_b[1], *space_a)
    y_ax_b_in_space_a = to_local_coords(space_a[0] + space_b[2], *space_a)
    A = np.array([[x_ax_b_in_space_a[0], y_ax_b_in_space_a[0]], [x_ax_b_in_space_a[1], y_ax_b_in_space_a[1]]])
    return AffineTransform2D(A, orig_b_in_space_a)

def signed_point_dist_to_line(point, v_a, v_b):
    v_ab = v_b - v_a
    n = np.array([-v_ab[1], v_ab[0]]) # rotation by 90 degrees
    n = n / np.linalg.norm(n)
    return np.dot(n,point - v_a)

def determinant_2d(col_1, col_2):
    return col_1[0] * col_2[1] - col_1[1] * col_2[0]

def point_on_rhs_of_line_2d_exclusive(point, v_a, v_b):
    eps = 1e-4
    return determinant_2d(v_b - v_a, point - v_a) < eps

def point_in_2d_triangle(point, v_a, v_b, v_c, eps=0.0):
    #1e-8 # maybe this will fix numerical issues
    if determinant_2d(v_b - v_a, point - v_a) < eps:
        return False
    if determinant_2d(v_c - v_b, point - v_b) < eps:
        return False
    if determinant_2d(v_a - v_c, point - v_c) < eps:
        return False
    return True

def signed_triangle_area(v_a, v_b, v_c):
    return 0.5 * determinant_2d(v_b - v_a, v_c - v_a)

def compute_all_interior_triangle_angles_2d(v_a, v_b, v_c):
    ab = (v_b - v_a) / np.linalg.norm(v_b - v_a)
    bc = (v_c - v_b) / np.linalg.norm(v_c - v_b)
    ca = (v_a - v_c) / np.linalg.norm(v_a - v_c)
    return (np.arccos(np.dot(ab, -ca)), np.arccos(np.dot(bc, -ab)), np.arccos(np.dot(ca, -bc)))

def face_corner_convex_3d(v_a, v_b, v_c, normal):
    ab = (v_b - v_a) / np.linalg.norm(v_b - v_a)
    bc = (v_c - v_b) / np.linalg.norm(v_c - v_b)
    return np.linalg.det(np.array([ab, bc, normal])) >= 0

def triangulate_3d_polygon(ccw_vertex_list, normal, external_vertex_id = None, crash_on_fail = False):
    """ Triangulates a 2D polygon embedded in 3D space """

    local_2d_space = construct_2d_space_along_face_edge(ccw_vertex_list[0], ccw_vertex_list[1], normal)
    vertex_2d_list = [to_local_coords(np.array(v), *local_2d_space) for v in ccw_vertex_list]
    triangles_2d, triangles_via_ids = triangulate_2d_polygon_angle_optimal(vertex_2d_list, external_vertex_id, crash_on_fail)

    return [to_world_coords(v, *local_2d_space) for v in triangles_2d], triangles_via_ids

def triangulate_2d_polygon_angle_optimal(ccw_vertex_list, external_vertex_id = None, crash_on_fail = False):
    """ Triangulates a 2D polygon. """
    
    k = len(ccw_vertex_list)
    if external_vertex_id is None:
        external_vertex_id = list(range(k))
    
    triangles_in_2d = []
    triangles_via_ids = []
    processed_verts = [False] * len(ccw_vertex_list)
    determinants = []
    for v_id in range(k):
        curr_v = ccw_vertex_list[v_id]
        prev_v = ccw_vertex_list[(v_id + k - 1) % k]
        next_v = ccw_vertex_list[(v_id + 1) % k]
        determinants.append(determinant_2d(curr_v - prev_v, next_v - curr_v))
    next_v_index = list(range(1, k)) + [0]
    prev_v_index = [k - 1] + list(range(k - 1))

    for _ in range(k - 2):
        best_ear_vertex = None
        largest_min_angle = -np.inf
        for curr_v_id in range(k):
            # skip processed verts
            if processed_verts[curr_v_id]:
                continue
            # skip concave vertex
            if determinants[curr_v_id] < -1e-4:           
                continue
            # check if all other verts lie outside of the triangle
            v_a = ccw_vertex_list[prev_v_index[curr_v_id]]
            v_b = ccw_vertex_list[curr_v_id]
            v_c = ccw_vertex_list[next_v_index[curr_v_id]]
            is_ear = True
            for other_v_id in range(k):
                if other_v_id == curr_v_id or prev_v_index[curr_v_id] == other_v_id or next_v_index[curr_v_id] == other_v_id or processed_verts[other_v_id]:
                    continue
                if point_in_2d_triangle(ccw_vertex_list[other_v_id], v_a, v_b, v_c):
                    is_ear = False
                    break
            # skip degenerate triangles
            curr_tri_area = signed_triangle_area(v_a, v_b, v_c)
            if curr_tri_area < 0.0:
                continue
            # add the ear to candidates
            curr_min_angle = min(compute_all_interior_triangle_angles_2d(v_a, v_b, v_c))
            if is_ear and curr_min_angle > largest_min_angle:
                largest_min_angle = curr_min_angle
                best_ear_vertex = curr_v_id
        # select best triangle
        if best_ear_vertex is not None:
            v_a = ccw_vertex_list[prev_v_index[best_ear_vertex]]
            v_b = ccw_vertex_list[best_ear_vertex]
            v_c = ccw_vertex_list[next_v_index[best_ear_vertex]]
            triangles_in_2d += [v_a, v_b, v_c]
            triangles_via_ids.append((external_vertex_id[prev_v_index[best_ear_vertex]], external_vertex_id[best_ear_vertex], external_vertex_id[next_v_index[best_ear_vertex]]))
            processed_verts[best_ear_vertex] = True

            # calculate det of two adjacent verts
            determinants[prev_v_index[best_ear_vertex]] = determinant_2d(v_a - ccw_vertex_list[prev_v_index[prev_v_index[best_ear_vertex]]], v_c - v_a)
            determinants[next_v_index[best_ear_vertex]] = determinant_2d(v_c - v_a, ccw_vertex_list[next_v_index[next_v_index[best_ear_vertex]]] - v_c)

            # delete curr vertex from ring
            next_v_index[prev_v_index[best_ear_vertex]] = next_v_index[best_ear_vertex]
            prev_v_index[next_v_index[best_ear_vertex]] = prev_v_index[best_ear_vertex]

    # sanity check
    if crash_on_fail:
        # if len(triangles_in_2d) != 3 * (k - 2):
        #     tri_list_4draw = []
        #     for i in range(0, len(triangles_in_2d), 3):
        #         tri_list_4draw.append((triangles_in_2d[i], triangles_in_2d[i+1], triangles_in_2d[i+2]))
        #     debug_draw_polygons_2d(tri_list_4draw, "triangulation_crash_report.png")
        assert len(triangles_in_2d) == 3 * (k - 2)

    return triangles_in_2d, triangles_via_ids

def solve_for_weird_intersection_point(center_point_3d, prev_point_3d, next_point_3d, normal, prev_offset, next_offset):
    local_basis = construct_2d_space_along_face_edge(center_point_3d, next_point_3d, normal)
    center_2d = to_local_coords(np.asarray(center_point_3d), *local_basis)
    prev_2d = to_local_coords(np.asarray(prev_point_3d), *local_basis)
    next_2d = to_local_coords(np.asarray(next_point_3d), *local_basis)

    ptc = center_2d - prev_2d
    line_normal_prev = np.array([-ptc[1], ptc[0]])
    line_normal_prev = line_normal_prev / np.linalg.norm(line_normal_prev)
    ctn = next_2d - center_2d
    line_normal_next = np.array([-ctn[1], ctn[0]])
    line_normal_next = line_normal_next / np.linalg.norm(line_normal_next)

    line_base_prev = center_2d + prev_offset * line_normal_prev
    line_base_next = center_2d + next_offset * line_normal_next
    M = np.array([line_normal_prev, line_normal_next])
    rhs = np.array([np.dot(line_normal_prev, line_base_prev), np.dot(line_normal_next, line_base_next)]).reshape(2,1)
    weird_p_2d = np.linalg.solve(M, rhs)

    return to_world_coords(weird_p_2d, *local_basis)

def solve_2d_line_line_intersection(a0, a1, b0, b1):
    a_dir = a1 - a0
    b_dir = b1 - b0
    M = np.array([a_dir, -b_dir]).T
    det = np.linalg.det(M)
    if abs(det) <= 1e-4:
        # skip intersection test for near parallel lines
        return None, None
    intersection_times = np.linalg.solve(M, b0 - a0)
    return intersection_times[0], intersection_times[1]


def triangle_intersection_test_2d(t1_a, t1_b, t1_c, t2_a, t2_b, t2_c):
    if determinant_2d(t1_b - t1_a, t1_c - t1_a) <= 0:
        print("POLYZAMBONI ERROR: Faulty triangle detected with determinant", determinant_2d(t1_b - t1_a, t1_c - t1_a))
    if determinant_2d(t2_b - t2_a, t2_c - t2_a) <= 0:
        print("POLYZAMBONI ERROR: Faulty triangle detected with determinant", determinant_2d(t2_b - t2_a, t2_c - t2_a))

    assert determinant_2d(t1_b - t1_a, t1_c - t1_a) > 0
    assert determinant_2d(t2_b - t2_a, t2_c - t2_a) > 0
        
    for e_t1 in [(t1_a, t1_b), (t1_b, t1_c), (t1_c, t1_a)]:
        if point_on_rhs_of_line_2d_exclusive(t2_a, *e_t1) and point_on_rhs_of_line_2d_exclusive(t2_b, *e_t1) and point_on_rhs_of_line_2d_exclusive(t2_c, *e_t1):
            return False
    for e_t2 in [(t2_a, t2_b), (t2_b, t2_c), (t2_c, t2_a)]:
        if point_on_rhs_of_line_2d_exclusive(t1_a, *e_t2) and point_on_rhs_of_line_2d_exclusive(t1_b, *e_t2) and point_on_rhs_of_line_2d_exclusive(t1_c, *e_t2):
            return False
        
    return True
