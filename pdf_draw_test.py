import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import numpy as np

def affine_transform_from_uv_to_vertices(vertices, uvs):
    a_t, b_t, c_t = tuple(np.asarray(v) for v in vertices)
    a_uv, b_uv, c_uv = tuple(np.asarray(uv) for uv in uvs)

    o_t = a_t
    o_uv = a_uv
    x_t = b_t - a_t
    y_t = c_t - a_t
    x_uv = b_uv - a_uv
    y_uv = c_uv - a_uv
    
    A_t = np.hstack((x_t.reshape((2,1)), y_t.reshape((2,1))))
    A_uv = np.hstack((x_uv.reshape((2,1)), y_uv.reshape((2,1))))

    # linear part
    linear_transform = A_t @ np.linalg.inv(A_uv)
    if np.linalg.det(linear_transform) < 0:
        print("triangle coords and uvs are flipped")

    # affine part
    translation = np.dot(linear_transform, -o_uv) + o_t

    # transformation matrix
    affine_mat = np.eye(3)
    affine_mat[:2, :2] = linear_transform
    affine_mat[:2, 2] = translation

    # construct affine transformation
    return mtransforms.Affine2D(matrix=affine_mat)

def normalize_np(np_arr):
    return np_arr / np.linalg.norm(np_arr)

def create_clip_path_polygon(vertices, eps):
    a_t, b_t, c_t = tuple(np.asarray(v) for v in vertices)

    ab = b_t - a_t
    bc = c_t - b_t
    ca = a_t - c_t

    a_clip = a_t + eps * normalize_np(ca - ab)
    b_clip = b_t + eps * normalize_np(ab - bc)
    c_clip = c_t + eps * normalize_np(bc - ca)

    return patches.Polygon((a_clip, b_clip, c_clip), closed=True, fill=None, transform=ax.transData)

def add_textured_triangle(ax, vertices, uvs, texture_image, outlines=True):

    #draw the triangle
    if outlines:
        draw_polygon = patches.Polygon(vertices, closed=True, fill=None, edgecolor='black', transform=ax.transData, lw=0.8, ls=(0,(10,10,2,10)), capstyle="round", )
        ax.add_patch(draw_polygon)

    # add the image texture
    texture_transform = affine_transform_from_uv_to_vertices(vertices, uvs)
    image_transform = texture_transform + ax.transData
    im = ax.imshow(texture_image, origin='upper', extent = [0,1,0,1], transform=image_transform, zorder=-1)
    im.set_clip_path(create_clip_path_polygon(vertices, 1e-1))


def add_triangle(ax, triangle):
    """
    Fügt ein Dreieck zu einer bestehenden Achse hinzu.

    :param ax: Matplotlib-Achse, auf der das Dreieck gezeichnet wird
    :param triangle: Tuple, containing x and y coordinates of the triangle's vertices
    """
    x, y = zip(*triangle)  # entpacke die Koordinaten
    ax.plot(x + (x[0],), y + (y[0],), color="black")  # schließe das Dreieck

# Beispielanwendung
fig, ax = plt.subplots()

# # Stelle Maßstäbe auf Zentimeter ein

# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_major_locator(plt.MultipleLocator(1))

ax.set_title('Textured Triangle Demo')

# Verschiebe die Achsen an die untere linke Ecke
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

fig.set_size_inches(8.27, 11.69)# Setze die x und y Achsen exakt an die untere linke Ecke des Dokuments
ax.set_xlim(0, 21)  # 21 cm für die Breite von DIN-A4 im Hochformat
ax.set_ylim(0, 29.7)  # 29.7 cm für die Höhe von DIN-A4 im Hochformat
# Stelle Maßstäbe auf Zentimeter ein
ax.set_aspect('equal')
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

a,b,c,d = ((5,5), (15,5), (10,10), (0,10))
triangle1 = [a, b, c]
triangle2 = [a, c, d]
triangle1_uvs = [(0,0), (1,0), (1,1)]
triangle2_uvs = [(0,0), (1,1), (0,1)]

img = mpimg.imread('test.png')

print(np.asarray(img).shape)

# make rgba
if img.shape[2] == 3:
    img_rgba = np.dstack((img, np.ones((img.shape[0], img.shape[1]))))
else:
    img_rgba = img

print(img_rgba.shape)

add_textured_triangle(ax, triangle1, triangle1_uvs, img_rgba, outlines=True)
add_textured_triangle(ax, triangle2, triangle2_uvs, img_rgba, outlines=True)

# for _ in range(1):
#     random_triangle = [np.random.rand(2) * np.array([21, 29.7]) for _ in range(3)]
#     add_textured_triangle(ax, random_triangle, triangle2_uvs, img)

# Speicher als PDF
with PdfPages('mehrere_dreiecke.pdf') as pdf:
    pdf.savefig(fig)

plt.close()
