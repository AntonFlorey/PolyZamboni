"""
All classes inheriting from PolyzamboniExporter generate paper model instructions from a list of ComponentPrintData objects.
So far only one exporter that uses matplotlib to create PDS as SVG
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing
import numpy.typing
from typing import NamedTuple
import platform
from matplotlib import axes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as lines
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
import numpy as np
from .geometry import AffineTransform2D
from .printprepper import ComponentPrintData, ColoredTriangleData, CutEdgeData, FoldEdgeData, GlueFlapData, FoldEdgeAtGlueFlapData

import gpu
import bpy
from gpu_extras.presets import draw_circle_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector
from .shaders import create_2d_textured_triangle_shader_no_color

# when testing on macosx, closing a matplotlib plot crashes blender. This is a workaround that kind of works. 
# if anyone knows a better way of dealing with this problem let me know :)
if platform.system() == "Darwin":
    import matplotlib
    matplotlib.use("agg") # try to use a different backend 

# feel free to add more paper sizes (in cm)
paper_sizes : dict[str, tuple[int, int]] = {
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

# lookup table for texture quality
texture_resolutions : dict[str, int] = {
    "low" : 512,
    "medium" : 1024,
    "high" : 2048,
    "super" : 4096,
    "ultra" : 8192
}

# feel free to add more linestyles (see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
custom_line_styles = {
    "-" : "-",
    ".." : (0, (2,4,2,4)),
    "-." : (0, (8,4,2,4)),
    "--." : (0, (8,4,8,4,2,4)),
    "-.." : (0, (8,4,2,4,2,4))
}

class ExportSettings(NamedTuple):
    paper_size : tuple[float, float] = paper_sizes["A4"],
    line_width : float = 1.0, 
    cut_edge_ls : str = "-",
    convex_fold_edge_ls : str = ".", 
    concave_fold_edge_ls : str = "-.", 
    glue_flap_ls : str = "-", 
    fold_hide_threshold_angle : float = 5,
    show_edge_numbers : bool = True,
    edge_number_font_size : int = 10,
    edge_number_offset : float = 0.1,
    show_build_step_numbers : bool = True,
    prints_on_model_inside : bool = True, 
    color_of_cut_edges : list[float] = [0,0,0],
    color_of_convex_fold_edges : list[float] = [0,0,0],
    color_of_concave_fold_edges : list[float] = [0,0,0],
    color_of_edge_numbers : list[float] = [0,0,0],
    color_of_build_steps : list[float] = [0,0,0],
    build_step_font_size : int = 10,
    texture_quality : str = "high"
    color_of_glue_flaps : list[float] = [0.5,0.1,0.1],
    two_sided_pages : bool = False,
    apply_texture_front : bool = True
    apply_texture_back : bool = False
    color_glue_flaps_front : bool = False,
    color_glue_flaps_back : bool = False
    

class PolyzamboniExporter(ABC):
    """ Base class for all export classes. """

    def __init__(self, output_format : str, export_settings : ExportSettings):
        self.output_format = output_format
        self.export_settings = export_settings
        self.textures : dict[str, bpy.types.Image] = {}

    @abstractmethod
    def export(self, print_ready_pages, output_file_name_prefix):
        pass

@dataclass
class ColoredTriangleBatch:
    texture_path : str | None
    coords : list[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]
    uvs : list[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]
    colors : list[tuple[float, float, float , float]]

class MatplotlibBasedExporter(PolyzamboniExporter):
    """ 
    This exporter makes use of the matplotlib package. 
    
    If anyone reads this, feel free to write your own exporter that does not require this package to be installed.
    """

    supported_formats = ["pdf", "svg"]

    def __linear_to_srgb(self, linear_color):
        linear_color = np.array(linear_color, dtype=np.float64)
        srgb_color = np.where(linear_color <= 0.0031308, 12.92 * linear_color, 1.055 * np.power(linear_color, 1/2.4) - 0.055)
        return srgb_color

    def __create_new_page(self):
        fig, ax = plt.subplots()

        cm = 1/2.54  # centimeters in inches
        paper_width, paper_height = self.export_settings.paper_size
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        fig.set_size_inches(paper_width * cm, paper_height * cm)
        ax.set_xlim(0, paper_width) 
        ax.set_ylim(0, paper_height) 
        ax.set_aspect('equal')
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.set_dpi(600)

        return fig, ax

    def __transform_component_coord_to_page_coord(self, component_coord, page_transform : AffineTransform2D, flip_along_short_side):
        page_coords = page_transform * component_coord
        if not flip_along_short_side:
            return page_coords
        paper_width, paper_height = self.export_settings.paper_size
        page_flip_transform = AffineTransform2D(np.array([[-1,0], [0,1]], dtype=np.float64), np.array([paper_width,0], dtype=np.float64))
        return page_flip_transform * page_coords

    def __transform_component_line_coords_to_page_coord(self, component_line_coords, page_transform : AffineTransform2D, flip_along_short_side):
        page_coord_from = self.__transform_component_coord_to_page_coord(component_line_coords[0], page_transform, flip_along_short_side)
        page_coord_to = self.__transform_component_coord_to_page_coord(component_line_coords[1], page_transform, flip_along_short_side)
        return (page_coord_from, page_coord_to)
    
    def __transform_component_coords_tuple_to_page_coords_tuple(self, component_coords_tuple, page_transform : AffineTransform2D, flip_along_short_side):
        return tuple([self.__transform_component_coord_to_page_coord(coord, page_transform, flip_along_short_side) for coord in component_coords_tuple])

    def __create_glue_flap_mask_of_component(self, ax : axes.Axes, component : ComponentPrintData, page_transform : AffineTransform2D, page_flipped = False):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        full_page_rect = mpath.Path([
            [xlim[0], ylim[0]],
            [xlim[1], ylim[0]],
            [xlim[1], ylim[1]],
            [xlim[0], ylim[1]],
            [xlim[0], ylim[0]],
        ], closed=True)
        triangle_masks_coords = [list(self.__transform_component_coords_tuple_to_page_coords_tuple(triangle.coords, page_transform, page_flipped)) for triangle in component.colored_triangles]
        triangle_masks_coords = [(coords + [coords[0]]) for coords in triangle_masks_coords]
        if not page_flipped:
            triangle_masks_coords = [coords[::-1] for coords in triangle_masks_coords]
        triangle_masks = [mpath.Path(coords, closed=True) for coords in triangle_masks_coords]

        return mpath.Path.make_compound_path(full_page_rect, *triangle_masks)

    def __draw_line(self, ax : axes.Axes, line_coords, linestyle, color, zorder=None, clip_path=None):
        line = lines.Line2D([line_coords[0][0], line_coords[1][0]], [line_coords[0][1], line_coords[1][1]], linewidth=self.export_settings.line_width, linestyle=custom_line_styles[linestyle], c=color, solid_capstyle="round")
        if zorder is not None:
            line.zorder = zorder
        if clip_path is not None:
            line.set_clip_path(clip_path, transform=ax.transData)
        ax.add_line(line)

    def __write_section_with_build_step_number(self, ax : axes.Axes, section_name, step_number, text_position, page_transform : AffineTransform2D, page_flipped = False):
        combined_text = ""
        if section_name is not None:
            combined_text += section_name + " "
        combined_text += str(step_number)
        if step_number == 6 or step_number == 9:
            combined_text += "."
        page_coord_text = self.__transform_component_coord_to_page_coord(text_position, page_transform, page_flipped)
        ax.text(page_coord_text[0], 
                page_coord_text[1], 
                combined_text, 
                fontsize=self.export_settings.build_step_font_size, 
                ha="center", 
                va="center", 
                color=self.__linear_to_srgb(self.export_settings.color_of_build_steps))

    def __write_text_along_line(self, ax : axes.Axes, line_coords, text, text_size, color = "black", offset_cm=0.1, flipped=False):
        start = line_coords[0]
        end = line_coords[1]
        midpoint = (start + end) / 2
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        if flipped:
            angle += np.pi
        # Normal of line
        normal = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)], dtype=np.float64)
        text_position = (midpoint[0] + offset_cm * normal[0], midpoint[1] + offset_cm * normal[1])
        if text == "6" or text == "9":
            text += "."
        ax.text(text_position[0], text_position[1], text,
                fontsize=text_size,  # Schriftgröße proportional zur Seitenlänge
                ha='center', va='baseline',
                rotation=np.rad2deg(angle), rotation_mode="anchor",
                color=color)

    def __draw_cut_edge(self, ax : axes.Axes, cut_edge_data : CutEdgeData, page_transform : AffineTransform2D, page_flipped = False):
        # transform line coords and draw
        page_coords = self.__transform_component_line_coords_to_page_coord(cut_edge_data.coords, page_transform, page_flipped)
        self.__draw_line(ax, page_coords, self.export_settings.cut_edge_ls, color=self.__linear_to_srgb(self.export_settings.color_of_cut_edges))

        # add edge number
        if not self.export_settings.show_edge_numbers or cut_edge_data.is_boundary:
            return
        self.__write_text_along_line(ax, page_coords, 
                                     str(cut_edge_data.edge_index), 
                                     self.export_settings.edge_number_font_size, 
                                     color=self.__linear_to_srgb(self.export_settings.color_of_edge_numbers), 
                                     offset_cm=self.export_settings.edge_number_offset, 
                                     flipped=page_flipped)

    def __draw_fold_edge(self, ax : axes.Axes, fold_edge_data : FoldEdgeData, page_transform : AffineTransform2D, page_flipped = False):
        if fold_edge_data.fold_angle <= self.export_settings.fold_hide_threshold_angle:
            return # dont draw almost flat folds
        # transform line coords
        page_coords = self.__transform_component_line_coords_to_page_coord(fold_edge_data.coords, page_transform, page_flipped)
        ls = self.export_settings.convex_fold_edge_ls if fold_edge_data.is_convex else self.export_settings.concave_fold_edge_ls
        color = self.__linear_to_srgb(self.export_settings.color_of_convex_fold_edges if fold_edge_data.is_convex else self.export_settings.color_of_concave_fold_edges)  
        self.__draw_line(ax, page_coords, ls, color)

    def __draw_fold_edge_at_glue_flap(self, ax : axes.Axes, fold_edge_at_flap_data : FoldEdgeAtGlueFlapData, page_transform : AffineTransform2D, page_flipped = False):
        page_coords = self.__transform_component_line_coords_to_page_coord(fold_edge_at_flap_data.coords, page_transform, page_flipped)
        if fold_edge_at_flap_data.fold_angle <= self.export_settings.fold_hide_threshold_angle:
            pass
        else:
            ls = self.export_settings.convex_fold_edge_ls if fold_edge_at_flap_data.is_convex else self.export_settings.concave_fold_edge_ls
            color = self.__linear_to_srgb(self.export_settings.color_of_convex_fold_edges if fold_edge_at_flap_data.is_convex else self.export_settings.color_of_concave_fold_edges)
            self.__draw_line(ax, page_coords, ls, color)
        # add edge number
        if not self.export_settings.show_edge_numbers:
            return
        self.__write_text_along_line(ax, page_coords, 
                                     str(fold_edge_at_flap_data.edge_index), 
                                     self.export_settings.edge_number_font_size, 
                                     color=self.__linear_to_srgb(self.export_settings.color_of_edge_numbers), 
                                     offset_cm=self.export_settings.edge_number_offset, 
                                     flipped=page_flipped)

    def __draw_glue_flap_faces(self, ax : axes.Axes, glue_flap_data : GlueFlapData, mask_path : mpath.Path, page_transform : AffineTransform2D, page_flipped = False):
        if len(glue_flap_data.tris) == 1:
            flap_coords_page = self.__transform_component_coords_tuple_to_page_coords_tuple(glue_flap_data.tris[0], page_transform, page_flipped)
        else:
            flap_coords_page = self.__transform_component_coords_tuple_to_page_coords_tuple(list(glue_flap_data.tris[0]) + [glue_flap_data.tris[1][2]], page_transform, page_flipped)
        draw_polygon = patches.Polygon(flap_coords_page, closed=True, fill=True, facecolor=self.__linear_to_srgb(self.export_settings.color_of_glue_flaps))
        draw_polygon.set_clip_path(mask_path, transform=ax.transData)
        draw_polygon.set_zorder(-3) 
        ax.add_patch(draw_polygon)

    def __draw_glue_flap_edges(self, ax : axes.Axes, glue_flap_data : GlueFlapData, mask_path : mpath.Path, page_transform : AffineTransform2D, page_flipped = False):
        # transform line coords and draw
        for flap_edge_coords in glue_flap_data.edge_coords:
            page_coords = self.__transform_component_line_coords_to_page_coord(flap_edge_coords, page_transform, page_flipped)
            self.__draw_line(ax, page_coords, self.export_settings.glue_flap_ls, color=self.__linear_to_srgb(self.export_settings.color_of_cut_edges), zorder=-2, clip_path=mask_path)

    def __create_page_render_data(self, components_on_page : list[ComponentPrintData], page_flipped = False):
        solid_color_triangle_batch = None
        textured_triangle_batches : dict[str, ColoredTriangleBatch]= {}

        for component_data in components_on_page:
            page_transform = component_data.page_transform
            for triangle_data in component_data.colored_triangles:
                if triangle_data.absolute_texture_path is None:
                    if triangle_data.color is None:
                        continue
                    if solid_color_triangle_batch is None:
                        solid_color_triangle_batch = ColoredTriangleBatch(None, [], [], [])
                    solid_color_triangle_batch.coords += list(self.__transform_component_coords_tuple_to_page_coords_tuple(triangle_data.coords, page_transform, page_flipped))
                    solid_color_triangle_batch.colors += [self.__linear_to_srgb(triangle_data.color)] * 3
                else:
                    curr_batch = textured_triangle_batches.setdefault(triangle_data.absolute_texture_path, 
                                                                      ColoredTriangleBatch(triangle_data.absolute_texture_path, [], [], []))
                    curr_batch.coords += list(self.__transform_component_coords_tuple_to_page_coords_tuple(triangle_data.coords, page_transform, page_flipped))
                    curr_batch.uvs += triangle_data.uvs
        return solid_color_triangle_batch, textured_triangle_batches

    def __draw_colored_triangles(self, ax : axes.Axes, 
                                 colored_tris : ColoredTriangleBatch | None, 
                                 textured_tris : dict[str, ColoredTriangleBatch]):
        TEXTURE_SIZE = texture_resolutions[self.export_settings.texture_quality]
        # Render batches into off-screen buffer and then let a np array read from that buffer
        offscreen = gpu.types.GPUOffScreen(TEXTURE_SIZE, TEXTURE_SIZE)
        np_image = np.empty(shape=(TEXTURE_SIZE, TEXTURE_SIZE, 4), dtype=np.uint8)  
        np_buffer = gpu.types.Buffer("UBYTE", np_image.shape, np_image)
        max_page_len = max(self.export_settings.paper_size)

        with offscreen.bind():
            fb : gpu.types.GPUFrameBuffer = gpu.state.active_framebuffer_get()
            fb.clear(color=(1.0, 1.0, 1.0, 0.0))
            with gpu.matrix.push_pop():
                # Page coords -> normalized device coordinates [-1, 1].
                ndc_scaling = Matrix.Scale(2.0 / max_page_len, 4)
                ndc_translation = Matrix.Translation(Vector([-1,-1,0]))
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(ndc_translation @ ndc_scaling)
                
                if colored_tris is not None:
                    shader = gpu.shader.from_builtin("FLAT_COLOR")
                    data = {
                        "pos" : colored_tris.coords,
                        "color" : colored_tris.colors
                    }
                    batch = batch_for_shader(shader, 'TRIS', data)
                    shader.bind()
                    batch.draw(shader)

                for triangle_batch in textured_tris.values():
                    if triangle_batch.texture_path not in self.textures:
                        self.textures[triangle_batch.texture_path] = bpy.data.images.load(filepath=triangle_batch.texture_path, check_existing=True)
                    texture = gpu.texture.from_image(self.textures[triangle_batch.texture_path])
                    shader = create_2d_textured_triangle_shader_no_color()
                    # prepare and draw batch
                    data = {
                        "pos" : triangle_batch.coords,
                        "uv" : triangle_batch.uvs
                    }
                    batch = batch_for_shader(shader, 'TRIS', data)
                    shader.bind()
                    shader.uniform_sampler("image", texture)
                    shader.uniform_float("viewProjectionMatrix", gpu.matrix.get_projection_matrix())
                    batch.draw(shader)
                        
            fb.read_color(0, 0, TEXTURE_SIZE, TEXTURE_SIZE, 4, 0, 'UBYTE', data=np_buffer)

        offscreen.free()
        # draw image
        ax.imshow(np_image, zorder=-1, extent=(0, max_page_len, 0, max_page_len), origin="lower")


    def __create_page_figure(self, components_on_page : list[ComponentPrintData], draw_textures = True, fill_glue_flaps = True, only_textures=False):
        # create a new figure for this page
        fig, ax = self.__create_new_page()

        flip_drawings = (only_textures != self.export_settings.prints_on_model_inside)

        if draw_textures:
            colored_tris, textured_tris = self.__create_page_render_data(components_on_page, page_flipped=flip_drawings)
            self.__draw_colored_triangles(ax, colored_tris, textured_tris)

        if fill_glue_flaps:
            for component_print_data in components_on_page:
                component_mask = self.__create_glue_flap_mask_of_component(ax, component_print_data, component_print_data.page_transform, flip_drawings)
                for glue_flap_data in component_print_data.glue_flaps:
                    self.__draw_glue_flap_faces(ax, glue_flap_data, component_mask, component_print_data.page_transform, flip_drawings)

        if only_textures:
            return fig, ax

        # draw all lines and text
        for component_print_data in components_on_page:
            component_mask = self.__create_glue_flap_mask_of_component(ax, component_print_data, component_print_data.page_transform, self.export_settings.prints_on_model_inside)
            if self.export_settings.show_build_step_numbers:
                self.__write_section_with_build_step_number(ax, component_print_data.build_section_name, 
                                                            component_print_data.build_step_number, component_print_data.build_step_number_position, 
                                                            component_print_data.page_transform, self.export_settings.prints_on_model_inside)
            for cut_edge_print_data in component_print_data.cut_edges:
                self.__draw_cut_edge(ax, cut_edge_print_data, component_print_data.page_transform, self.export_settings.prints_on_model_inside)
            
            for fold_edge_print_data in component_print_data.fold_edges:
                self.__draw_fold_edge(ax, fold_edge_print_data, component_print_data.page_transform, self.export_settings.prints_on_model_inside)

            for fold_edge_at_flap_print_data in component_print_data.fold_edges_at_flaps:
                self.__draw_fold_edge_at_glue_flap(ax, fold_edge_at_flap_print_data, component_print_data.page_transform, self.export_settings.prints_on_model_inside)

            for glue_flap_data in component_print_data.glue_flaps:
                self.__draw_glue_flap_edges(ax, glue_flap_data, component_mask, component_print_data.page_transform, self.export_settings.prints_on_model_inside)

        return fig, ax

    def export(self, print_ready_pages, output_file_name_prefix):
        if self.output_format not in self.supported_formats:
            print("POLYZAMBONI ERROR: Output format not supported!")
            return

        if self.output_format == "pdf":
            with PdfPages(output_file_name_prefix + ".pdf") as pdf:
                for components_on_page in print_ready_pages:
                    # create a new figure for this page
                    fig, _ = self.__create_page_figure(components_on_page, self.export_settings.apply_texture_front, self.export_settings.color_glue_flaps_front, False)
                    # save everything and close current figure
                    pdf.savefig(fig)
                    fig.clear()
                    plt.close(fig)
                    # two sided printing
                    if self.export_settings.two_sided_pages:
                        fig, _ = self.__create_page_figure(components_on_page, self.export_settings.apply_texture_back, self.export_settings.color_glue_flaps_back, True)
                        # save everything and close current figure
                        pdf.savefig(fig)
                        fig.clear()
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
                fig, _ = self.__create_page_figure(components_on_page, self.export_settings.apply_texture_front, self.export_settings.color_glue_flaps_front, False)
                # save everything and close current figure
                fig.savefig(output_file_name_prefix + "_page" + str(page_number) + ".svg")
                fig.clear()
                plt.close(fig)
                page_number += 1
                # two sided printing
                if self.export_settings.two_sided_pages:
                    fig, _ = self.__create_page_figure(components_on_page, self.export_settings.apply_texture_back, self.export_settings.color_glue_flaps_back, True)
                    # save everything and close current figure
                    fig.savefig(output_file_name_prefix + "_page" + str(page_number) + ".svg")
                    fig.clear()
                    plt.close(fig)
                    page_number += 1

        plt.close("all")
        print("POLYZAMBONI INFO: Paper model instructions successfully created!")
