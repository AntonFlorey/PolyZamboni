from abc import ABC, abstractmethod
from matplotlib import axes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import numpy as np
from .geometry import AffineTransform2D
from .printprepper import ComponentPrintData, ColoredTriangleData, CutEdgeData, FoldEdgeData, GlueFlapEdgeData

# feel free to add more paper sizes (in cm)
paper_sizes = {
    "A0" : (84.1, 118.8),
    "A1" : (59.4, 84.1),
    "A2" : (42.0, 59.4),
    "A3" : (29.7, 42.0),
    "A4" : (21.0, 29.7),
    "A5" : (14.8, 21.0),
    "A6" : (10.5, 14.8),
    "A7" : (7.4, 10.5),
    "A8" : (5.2, 7.4),
    "Letter" : (21.6, 27.9),
    "Legal" : (21.6, 35.6),
    "Tabloid" : (27.9, 43.2)
}

# feel free to add more linestyles (see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
custom_line_styles = {
    "-" : "-",
    ".." : (0, (2,4,2,4)),
    "-." : (0, (8,4,2,4)),
    "--." : (0, (8,4,8,4,2,4)),
    "-.." : (0, (8,4,2,4,2,4))
}

class PolyzamboniExporter(ABC):
    """ Base class for all export classes. """

    def __init__(self, output_format = "pdf",
                 paper_size = "A4",
                 line_width = 1, 
                 cut_edge_ls = "-",
                 convex_fold_edge_ls = ".", 
                 concave_fold_edge_ls = "-.", 
                 glue_flap_ls = "-", 
                 fold_hide_threshold_angle = 5,
                 show_edge_numbers = True,
                 edge_number_font_size = 10,
                 edge_number_offset = 0.1,
                 show_build_step_numbers = True,
                 apply_main_texture = False, 
                 print_on_inside = True, 
                 two_sided_w_texture = False,
                 color_of_lines = [0,0,0],
                 color_of_edge_numbers = [0,0,0],
                 color_of_build_steps = [0,0,0],
                 build_step_font_size = 10):
        self.output_format = output_format
        self.paper_size = paper_sizes[paper_size]
        self.line_width = line_width
        self.cut_edge_linestyle = cut_edge_ls
        self.convex_fold_edge_linestyle = convex_fold_edge_ls
        self.concave_fold_edge_linestyle = concave_fold_edge_ls
        self.glue_flap_linestyle = glue_flap_ls
        self.show_edge_numbers = show_edge_numbers
        self.edge_number_font_size = edge_number_font_size
        self.edge_number_offset = edge_number_offset
        self.show_build_step_numbers = show_build_step_numbers
        self.apply_textures = apply_main_texture
        self.prints_on_model_inside = print_on_inside
        self.two_sided_with_texture = two_sided_w_texture
        self.fold_hide_threshold_angle = fold_hide_threshold_angle
        self.color_of_lines = color_of_lines
        self.color_of_edge_numbers = color_of_edge_numbers
        self.build_step_number_font_size = build_step_font_size
        self.builf_step_number_color = color_of_build_steps

        self.texture_images = {}

    @abstractmethod
    def export(self, print_ready_pages, output_file_name_prefix):
        pass

class MatplotlibBasedExporter(PolyzamboniExporter):
    """ 
    This exporter makes use of the matplotlib package. 
    
    If anyone reads this, feel free to write your own exporter that does not require this package to be installed 
    """

    supported_formats = ["pdf", "svg"]

    def __linear_to_srgb(self, linear_color):
        linear_color = np.array(linear_color)
        srgb_color = np.where(linear_color <= 0.0031308, 12.92 * linear_color, 1.055 * np.power(linear_color, 1/2.4) - 0.055)
        return srgb_color

    def __create_new_page(self):
        fig, ax = plt.subplots()

        cm = 1/2.54  # centimeters in inches
        paper_width, paper_height = self.paper_size
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        fig.set_size_inches(paper_width * cm, paper_height * cm)
        ax.set_xlim(0, paper_width) 
        ax.set_ylim(0, paper_height) 
        ax.set_aspect('equal')
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        return fig, ax

    def __transform_component_coord_to_page_coord(self, component_coord, page_transform : AffineTransform2D, flip_along_short_side):
        page_coords = page_transform * component_coord
        if not flip_along_short_side:
            return page_coords
        paper_width, paper_height = self.paper_size
        page_flip_transform = AffineTransform2D(np.array([[-1,0], [0,1]]), np.array([paper_width,0]))
        return page_flip_transform * page_coords
    
    def __transform_component_line_coords_to_page_coord(self, component_line_coords, page_transform : AffineTransform2D, flip_along_short_side):
        page_coord_from = self.__transform_component_coord_to_page_coord(component_line_coords[0], page_transform, flip_along_short_side)
        page_coord_to = self.__transform_component_coord_to_page_coord(component_line_coords[1], page_transform, flip_along_short_side)
        return (page_coord_from, page_coord_to)
    
    def __transform_component_triangle_coords_to_page_coords(self, component_tri_coords, page_transform : AffineTransform2D, flip_along_short_side):
        page_coord_a = self.__transform_component_coord_to_page_coord(component_tri_coords[0], page_transform, flip_along_short_side)
        page_coord_b = self.__transform_component_coord_to_page_coord(component_tri_coords[1], page_transform, flip_along_short_side)
        page_coord_c = self.__transform_component_coord_to_page_coord(component_tri_coords[2], page_transform, flip_along_short_side)
        return (page_coord_a, page_coord_b, page_coord_c)

    def __draw_line(self, ax : axes.Axes, line_coords, linestyle, color):
        line = lines.Line2D([line_coords[0][0], line_coords[1][0]], [line_coords[0][1], line_coords[1][1]], linewidth=self.line_width, linestyle=custom_line_styles[linestyle], c=color, solid_capstyle="round")
        ax.add_line(line)

    def __write_text(self, ax : axes.Axes, text, coord, text_size, color, page_transform : AffineTransform2D):
        page_coord_text = self.__transform_component_coord_to_page_coord(coord, page_transform, self.prints_on_model_inside)
        if text == "6" or text == "9":
            text += "."
        ax.text(page_coord_text[0], page_coord_text[1], text, fontsize=text_size, ha="center", va="center", color=color)

    def __write_text_along_line(self, ax : axes.Axes, line_coords, text, text_size, color = "black", offset_cm=0.1, flipped=False):
        start = line_coords[0]
        end = line_coords[1]
        midpoint = (start + end) / 2
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        if flipped:
            angle += np.pi
        # Normal of line
        normal = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])
        text_position = (midpoint[0] + offset_cm * normal[0], midpoint[1] + offset_cm * normal[1])
        if text == "6" or text == "9":
            text += "."
        ax.text(text_position[0], text_position[1], text,
                fontsize=text_size,  # Schriftgröße proportional zur Seitenlänge
                ha='center', va='baseline',
                rotation=np.rad2deg(angle), rotation_mode="anchor",
                color=color)

    def __draw_cut_edge(self, ax : axes.Axes, cut_edge_data : CutEdgeData, page_transform : AffineTransform2D):
        # transform line coords and draw
        page_coords = self.__transform_component_line_coords_to_page_coord(cut_edge_data.coords, page_transform, self.prints_on_model_inside)
        self.__draw_line(ax, page_coords, self.cut_edge_linestyle, color=self.__linear_to_srgb(self.color_of_lines))

        # add edge number
        if not self.show_edge_numbers:
            return
        self.__write_text_along_line(ax, page_coords, str(cut_edge_data.edge_index), self.edge_number_font_size, color=self.__linear_to_srgb(self.color_of_edge_numbers), offset_cm=self.edge_number_offset, flipped=self.prints_on_model_inside)

    def __draw_fold_edge(self, ax : axes.Axes, fold_edge_data : FoldEdgeData, page_transform : AffineTransform2D):
        if fold_edge_data.fold_angle <= self.fold_hide_threshold_angle:
            return # dont draw almost flat folds
        # transform line coords
        page_coords = self.__transform_component_line_coords_to_page_coord(fold_edge_data.coords, page_transform, self.prints_on_model_inside)
        self.__draw_line(ax, page_coords, self.convex_fold_edge_linestyle if fold_edge_data.is_convex else self.concave_fold_edge_linestyle, color=self.__linear_to_srgb(self.color_of_lines))

    def __draw_glue_flap_edge(self, ax : axes.Axes, glue_flap_edge_data : GlueFlapEdgeData, page_transform : AffineTransform2D):
        # transform line coords and draw
        page_coords = self.__transform_component_line_coords_to_page_coord(glue_flap_edge_data.coords, page_transform, self.prints_on_model_inside)
        self.__draw_line(ax, page_coords, self.glue_flap_linestyle, color=self.__linear_to_srgb(self.color_of_lines))

    def __create_thickened_triangle_coords(self, triangle_coords, eps):
        a_t, b_t, c_t = tuple(np.asarray(v) for v in triangle_coords)

        ab = b_t - a_t
        bc = c_t - b_t
        ca = a_t - c_t

        def normalize_np(np_arr):
            return np_arr / np.linalg.norm(np_arr)

        a_thick = a_t + eps * normalize_np(ca - ab)
        b_thick = b_t + eps * normalize_np(ab - bc)
        c_thick = c_t + eps * normalize_np(bc - ca)

        return (a_thick, b_thick, c_thick)

    def __affine_transform_from_uv_to_vertices(self, vertices, uvs):
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

        # affine part
        translation = np.dot(linear_transform, -o_uv) + o_t

        # transformation matrix
        affine_mat = np.eye(3)
        affine_mat[:2, :2] = linear_transform
        affine_mat[:2, 2] = translation

        # construct affine transformation
        return mtransforms.Affine2D(matrix=affine_mat)

    def __retrieve_texture_image(self, image_path):
        if image_path not in self.texture_images.keys():
            img = mpimg.imread(image_path)
            self.texture_images[image_path] = img
        return self.texture_images[image_path]

    def __draw_textured_triangle(self, ax : axes.Axes, triangle_coords, triangle_uvs, image_path):
        clip_polygon = patches.Polygon(triangle_coords, closed=True, fill=None, transform=ax.transData)
        # add the image texture

        texture_image = self.__retrieve_texture_image(image_path)
        texture_transform = self.__affine_transform_from_uv_to_vertices(triangle_coords, triangle_uvs)
        image_transform = texture_transform + ax.transData
        im = ax.imshow(texture_image, origin='upper', extent = [0,1,0,1], transform=image_transform, zorder=-1)
        im.set_clip_path(clip_polygon)

    def __draw_solid_color_triangle(self, ax : axes.Axes, triangle_coords, color):
        draw_polygon = patches.Polygon(triangle_coords, closed=True, fill=True, facecolor=tuple(color))
        ax.add_patch(draw_polygon)

    def __draw_colored_triangle(self, ax : axes.Axes, colored_tri_data : ColoredTriangleData, page_transform : AffineTransform2D):
        if colored_tri_data.absolute_texture_path is None and colored_tri_data.color is None:
            return # nothing to draw here
        
        triangle_page_coords = self.__transform_component_triangle_coords_to_page_coords(colored_tri_data.coords, page_transform, self.prints_on_model_inside)
        thickened_triangle_coords = self.__create_thickened_triangle_coords(triangle_page_coords, 0.08) # make triangles one millimeter thicker

        if colored_tri_data.absolute_texture_path is not None:
            # draw texture
            self.__draw_textured_triangle(ax, thickened_triangle_coords, colored_tri_data.uvs, colored_tri_data.absolute_texture_path)
        elif colored_tri_data.color is not None:
            # fill with solid color
            srgb_color = self.__linear_to_srgb(np.array(colored_tri_data.color))
            self.__draw_solid_color_triangle(ax, thickened_triangle_coords, srgb_color)

    def __create_page_figure(self, components_on_page, draw_textures = True, only_textures=False):
        # create a new figure for this page
        fig, ax = self.__create_new_page()
        
        if only_textures:
            for component_on_page in components_on_page:
                component_print_data : ComponentPrintData = component_on_page["print data"]
                component_page_transform : AffineTransform2D = component_on_page["page coord transform"]
                for colored_triangle_data in component_print_data.colored_triangles:
                    self.__draw_colored_triangle(ax, colored_triangle_data, component_page_transform)
            return fig, ax

        # draw all lines and text and maybe colored triangles
        for component_on_page in components_on_page:
            component_print_data : ComponentPrintData = component_on_page["print data"]
            component_page_transform : AffineTransform2D = component_on_page["page coord transform"]
            
            if self.show_build_step_numbers:
                self.__write_text(ax, str(component_print_data.build_step_number), 
                                component_print_data.build_step_number_position,
                                self.build_step_number_font_size,
                                self.__linear_to_srgb(self.builf_step_number_color),
                                component_page_transform)

            for cut_edge_print_data in component_print_data.cut_edges:
                self.__draw_cut_edge(ax, cut_edge_print_data, component_page_transform)
            
            for fold_edge_print_data in component_print_data.fold_edges:
                self.__draw_fold_edge(ax, fold_edge_print_data, component_page_transform)

            for glue_flap_edge_data in component_print_data.glue_flap_edges:
                self.__draw_glue_flap_edge(ax, glue_flap_edge_data, component_page_transform)

            if draw_textures:
                for colored_triangle_data in component_print_data.colored_triangles:
                    self.__draw_colored_triangle(ax, colored_triangle_data, component_page_transform)
        return fig, ax

    def export(self, print_ready_pages, output_file_name_prefix):
        
        if self.output_format not in self.supported_formats:
            print("POLYZAMBONI ERROR: Output format not supported!")
            return

        if self.output_format == "pdf":
            with PdfPages(output_file_name_prefix + ".pdf") as pdf:
                for components_on_page in print_ready_pages:
                    # create a new figure for this page
                    fig, ax = self.__create_page_figure(components_on_page, self.apply_textures and not self.two_sided_with_texture, False)
                    # save everything and close current figure
                    pdf.savefig(fig)
                    plt.close(fig)
                    # two sided printing
                    if self.apply_textures and self.two_sided_with_texture:
                        fig, ax = self.__create_page_figure(components_on_page, True, True)
                        # save everything and close current figure
                        pdf.savefig(fig)
                        plt.close(fig)

                # some metadata
                d = pdf.infodict()
                d['Title'] = 'Papercraft instructions by PolyZamboni'
                d['Author'] = 'PolyZamboni'
                d['Subject'] = 'Have fun crafting!'

        if self.output_format == "svg":
            # create a svg file for each page
            page_number = 1
            for components_on_page in print_ready_pages:
                # create a new figure for this page
                    fig, ax = self.__create_page_figure(components_on_page, self.apply_textures and not self.two_sided_with_texture, False)
                    # save everything and close current figure
                    fig.savefig(output_file_name_prefix + "_page" + str(page_number) + ".svg")
                    plt.close(fig)
                    page_number += 1
                    # two sided printing
                    if self.apply_textures and self.two_sided_with_texture:
                        fig, ax = self.__create_page_figure(components_on_page, True, True)
                        # save everything and close current figure
                        fig.savefig(output_file_name_prefix + "_page" + str(page_number) + ".svg")
                        plt.close(fig)
                        page_number += 1
        
        print("Called matplotlib based exporter with output filepath:", output_file_name_prefix)
