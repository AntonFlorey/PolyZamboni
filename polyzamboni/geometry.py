import numpy as np
import bpy
import bmesh
import math
import mathutils
from mathutils import Vector, Matrix, Quaternion
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    def __str__(self):
        return "2D Affine Transformation\nLinear Part:\n" + self.A.__str__() + "\nAffine Part:\n" + self.t.__str__()

def generate_random_point_on_unit_circle():
    # Angle in radians
    angle = np.random.uniform(0, 2 * np.pi)
    # x and y coordinates on the unit circle
    x = np.cos(angle)
    y = np.sin(angle)
    return np.array([x, y])

def generate_random_triangle():
    triangle = [generate_random_point_on_unit_circle() for _ in range(3)]

    # ensure ccw order
    if determinant_2d(triangle[1] - triangle[0], triangle[2] - triangle[0]) < 0:
        triangle = [triangle[0], triangle[2], triangle[1]]

    return triangle

def draw_triangle(triangle, ax, color):
    triangle.append(triangle[0])  # to close the triangle
    x_coords, y_coords = zip(*triangle)
    ax.plot(x_coords, y_coords, marker='o', color=color)
    
def two_random_triangle_test():
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)
    
    colors = ['b', 'g']  # Colors for the triangles
    t1 = generate_random_triangle()
    t2 = generate_random_triangle()
    color = "g"
    if triangle_intersection_test_2d(*t1, *t2):
        color = "r"
    draw_triangle(t1, ax, color)
    draw_triangle(t2, ax, color)
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Random 2D Triangles on the Unit Circle")
    plt.show()
    plt.close(fig)

def debug_draw_polygon_2d(polygon_2d, vertex_indices):

    # Polygon zeichnen
    polygon = plt.Polygon(polygon_2d, closed=True, edgecolor='r', facecolor='orange')

    fig, ax = plt.subplots()
    ax.add_patch(polygon)
    ax.set_xlim(min(polygon_2d[:,0])-0.1, max(polygon_2d[:,0])+0.1)
    ax.set_ylim(min(polygon_2d[:,1])-0.1, max(polygon_2d[:,1])+0.1)
    ax.set_aspect('equal')  # Gleiche Skala für x und y

    for i, (x, y) in zip(vertex_indices, polygon_2d): 
        ax.text(x, y, str(i), fontsize=6, ha='center', va='center', color='black')#, bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))

    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')
    plt.title('Gezeichnetes Polygon')
    plt.grid(True)
    plt.savefig("test.png")
    plt.close(fig)

def draw_polygon(ax, vertices, number):
    polygon = patches.Polygon(vertices, closed=True, edgecolor='black', facecolor='none')
    ax.add_patch(polygon)

    # Calculate the centroid of the polygon
    center_x = np.mean([v[0] for v in vertices])
    center_y = np.mean([v[1] for v in vertices])
    
    ax.text(center_x, center_y, str(number), color='black', ha='center', va='center', fontsize=8)

def debug_draw_polygons_2d(polygons_2d):
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

    plt.xlabel('X-Achse')
    plt.ylabel('Y-Achse')
    plt.title('Unfolded Polygons')
    plt.grid(False)
    plt.savefig("debug-unfolding.png")
    plt.close(fig)

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

    # assertion
    assert np.allclose(sum(face_areas.values()), sum(voronoi_areas.values()))

    return voronoi_areas

def to_local_coords(point_in_3d, frame_orig, base_x, base_y):
    relative_coord = np.asarray(point_in_3d) - frame_orig
    # project
    return np.array([np.dot(base_x, relative_coord), np.dot(base_y, relative_coord)])

def to_world_coords(point_in_2d, frame_orig, base_x, base_y):
    return frame_orig + point_in_2d[0] * base_x + point_in_2d[1] * base_y

def construct_2d_space_along_face_edge(v_3d_from, v_3d_to, n_3d):
    x_ax = np.array(v_3d_to - v_3d_from)
    x_ax = x_ax / np.linalg.norm(x_ax)
    n = np.array(n_3d)
    y_ax = np.cross(n, x_ax)
    y_ax = y_ax / np.linalg.norm(y_ax)
    orig = np.array(v_3d_from)
    return (orig, x_ax, y_ax)

def affine_2d_transformation_between_two_2d_spaces_on_same_plane(space_a, space_b):
    """ Returns a 2d affine transformation that maps from space b to space a """
    orig_b_in_space_a = to_local_coords(space_b[0], *space_a)
    x_ax_b_in_space_a = to_local_coords(space_a[0] + space_b[1], *space_a)
    y_ax_b_in_space_a = to_local_coords(space_a[0] + space_b[2], *space_a)
    A = np.array([[x_ax_b_in_space_a[0], y_ax_b_in_space_a[0]], [x_ax_b_in_space_a[1], y_ax_b_in_space_a[1]]])
    
    #print("det of near-rotation matrix:", np.linalg.det(A))

    return AffineTransform2D(A, orig_b_in_space_a)

def determinant_2d(col_1, col_2):
    return col_1[0] * col_2[1] - col_1[1] * col_2[0]

def point_on_rhs_of_line_2d_exclusive(point, v_a, v_b):
    eps = 1e-4
    return determinant_2d(v_b - v_a, point - v_a) < eps

def point_in_2d_triangle(point, v_a, v_b, v_c):
    eps = 0 #1e-8 # maybe this will fix numerical issues
    if determinant_2d(v_b - v_a, point - v_a) < eps:
        return False
    if determinant_2d(v_c - v_b, point - v_b) < eps:
        return False
    if determinant_2d(v_a - v_c, point - v_c) < eps:
        return False
    return True

def signed_triangle_area(v_a, v_b, v_c):
    return 0.5 * determinant_2d(v_b - v_a, v_c - v_a)

def triangulate_3d_polygon(ccw_vertex_list, normal, external_vertex_id = None, crash_on_fail = False):
    """ Triangulates a 2D polygon embedded in 3D space """

    local_2d_space = construct_2d_space_along_face_edge(ccw_vertex_list[0], ccw_vertex_list[1], normal)
    vertex_2d_list = [to_local_coords(np.array(v), *local_2d_space) for v in ccw_vertex_list]
    triangles_2d, triangles_via_ids = triangulate_2d_polygon(vertex_2d_list, external_vertex_id, crash_on_fail)

    return [to_world_coords(v, *local_2d_space) for v in triangles_2d], triangles_via_ids

def triangulate_2d_polygon(ccw_vertex_list, external_vertex_id = None, crash_on_fail = False):
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
    
    triangulation_done = False
    while not triangulation_done:
        triangulation_done = True
        if np.less(np.array(determinants), 0).all():
            print("OH NONNONONO")
            break
        some_ear_clipped = False
        for curr_v_id in range(k):
            if processed_verts[curr_v_id]:
                continue
            triangulation_done = False
            
            # skip concave vertex
            if determinants[curr_v_id] < 0:                
                continue
            
            # check if all other verts lie outside of the triangle
            v_a = ccw_vertex_list[prev_v_index[curr_v_id]]
            v_b = ccw_vertex_list[curr_v_id]
            v_c = ccw_vertex_list[next_v_index[curr_v_id]]

            # skip degenerate triangles
            if abs(signed_triangle_area(v_a, v_b, v_c)) < 1e-4:
                continue

            is_ear = True
            for other_v_id in range(k):
                if other_v_id == curr_v_id or prev_v_index[curr_v_id] == other_v_id or next_v_index[curr_v_id] == other_v_id or processed_verts[other_v_id]:
                    continue
                if point_in_2d_triangle(ccw_vertex_list[other_v_id], v_a, v_b, v_c):
                    is_ear = False
                    break
            
            # clip the ear
            if is_ear:
                some_ear_clipped = True
                triangles_in_2d += [v_a, v_b, v_c]
                triangles_via_ids.append((external_vertex_id[prev_v_index[curr_v_id]], external_vertex_id[curr_v_id], external_vertex_id[next_v_index[curr_v_id]]))
                if len(triangles_in_2d) == 3 * (k - 2):
                    triangulation_done = True
                    break
                processed_verts[curr_v_id] = True

                # calculate det of two adjacent verts
                determinants[prev_v_index[curr_v_id]] = determinant_2d(v_a - ccw_vertex_list[prev_v_index[prev_v_index[curr_v_id]]], v_c - v_a)
                determinants[next_v_index[curr_v_id]] = determinant_2d(v_c - v_a, ccw_vertex_list[next_v_index[next_v_index[curr_v_id]]] - v_c)

                # delete curr vertex from ring
                next_v_index[prev_v_index[curr_v_id]] = next_v_index[curr_v_id]
                prev_v_index[next_v_index[curr_v_id]] = prev_v_index[curr_v_id]

        if len(triangles_in_2d) == 3 * (k - 3):
            # just add last triangle
            for curr_v_id in range(k):
                if processed_verts[curr_v_id]:
                    continue
                v_a = ccw_vertex_list[prev_v_index[curr_v_id]]
                v_b = ccw_vertex_list[curr_v_id]
                v_c = ccw_vertex_list[next_v_index[curr_v_id]]
                if crash_on_fail:
                    assert determinant_2d(v_b - v_a, v_c - v_a) > 0
                triangles_in_2d += [v_a, v_b, v_c]
                triangles_via_ids.append((external_vertex_id[prev_v_index[curr_v_id]], external_vertex_id[curr_v_id], external_vertex_id[next_v_index[curr_v_id]]))
                break
            triangulation_done = True
            break
        
        if not some_ear_clipped: # only for debugging
            print("OH Hell naaah")
            print(ccw_vertex_list)
            print(external_vertex_id)
            print(determinants)
            print(triangles_via_ids)
            print(processed_verts)
            debug_draw_polygon_2d(np.asarray(ccw_vertex_list), external_vertex_id)
            break

    if crash_on_fail:
        assert len(triangles_in_2d) == 3 * (k - 2)

    return triangles_in_2d, triangles_via_ids

def solve_for_weird_point(v_a, v_b, v_c, normal):
    x_ax = np.array(v_a - v_b)
    x_ax = x_ax / np.linalg.norm(x_ax)
    n = np.array(normal)
    y_ax = np.cross(n, x_ax)
    y_ax = y_ax / np.linalg.norm(y_ax)
    orig = np.array(v_b)

    a_2d = to_local_coords(np.asarray(v_a), orig, x_ax, y_ax)
    b_2d = to_local_coords(np.asarray(v_b), orig, x_ax, y_ax)
    c_2d = to_local_coords(np.asarray(v_c), orig, x_ax, y_ax)

    n_1 = a_2d - b_2d
    n_2 = c_2d - b_2d

    M = np.array([n_1, n_2])
    rhs = np.array([np.dot(n_1, a_2d), np.dot(n_2, c_2d)]).reshape(2,1)
    weird_p_2d = np.linalg.solve(M, rhs)

    return to_world_coords(weird_p_2d, orig, x_ax, y_ax)

def triangle_intersection_test_2d(t1_a, t1_b, t1_c, t2_a, t2_b, t2_c):
    if determinant_2d(t1_b - t1_a, t1_c - t1_a) <= 0:
        print("Faulty triangle detected with determinant:", determinant_2d(t1_b - t1_a, t1_c - t1_a))
    if determinant_2d(t2_b - t2_a, t2_c - t2_a) <= 0:
        print("Faulty triangle detected with determinant:", determinant_2d(t2_b - t2_a, t2_c - t2_a))

    assert determinant_2d(t1_b - t1_a, t1_c - t1_a) > 0
    assert determinant_2d(t2_b - t2_a, t2_c - t2_a) > 0
        
    for e_t1 in [(t1_a, t1_b), (t1_b, t1_c), (t1_c, t1_a)]:
        if point_on_rhs_of_line_2d_exclusive(t2_a, *e_t1) and point_on_rhs_of_line_2d_exclusive(t2_b, *e_t1) and point_on_rhs_of_line_2d_exclusive(t2_c, *e_t1):
            return False
    for e_t2 in [(t2_a, t2_b), (t2_b, t2_c), (t2_c, t2_a)]:
        if point_on_rhs_of_line_2d_exclusive(t1_a, *e_t2) and point_on_rhs_of_line_2d_exclusive(t1_b, *e_t2) and point_on_rhs_of_line_2d_exclusive(t1_c, *e_t2):
            return False
        
    return True

if __name__ == "__main__":
    print("tests...")
    import time

    # some amount of tests
    num_tests = 10000000
    print("Testing", num_tests, "random triangles...")
    time_for_instersection_tests = 0
    intersecting_cases = 0

    for _ in range(num_tests):
        t1 = generate_random_triangle()
        t2 = generate_random_triangle()

        start_t = time.time()
        intersecting = triangle_intersection_test_2d(*t1, *t2)
        end_t = time.time()
        time_for_instersection_tests += end_t - start_t

        if intersecting:
            intersecting_cases += 1
    
    print("Intersection computation took", time_for_instersection_tests, "seconds.")
    print("Avg time per test:", time_for_instersection_tests / num_tests)
    print("Percentage of intersecting triangles:", intersecting_cases / num_tests)