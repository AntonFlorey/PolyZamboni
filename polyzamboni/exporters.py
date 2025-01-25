from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
# from .printprepper import ComponentPrintData, ColoredTriangleData
# from .geometry import AffineTransform2D

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
}

# feel free to add more linestyles (see https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html)
custom_line_styles = {
    "--." : (0, (5,2,5,2,1,2))
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
                 show_edge_numbers = True,
                 show_build_step_numbers = True,
                 apply_main_texture = False, 
                 print_on_inside = True, 
                 two_sided_w_texture = False):
        self.output_format = output_format
        self.paper_size = paper_sizes[paper_size]
        self.line_width = line_width
        self.cut_edge_linestyle = cut_edge_ls
        self.convex_fold_edge_linestyle = convex_fold_edge_ls
        self.concave_fold_edge_linestyle = concave_fold_edge_ls
        self.glue_flap_linestyle = glue_flap_ls
        self.show_edge_numbers = show_edge_numbers
        self.show_build_step_numbers = show_build_step_numbers
        self.apply_textures = apply_main_texture
        self.prints_on_model_inside = print_on_inside
        self.two_sided_with_texture = two_sided_w_texture

    @abstractmethod
    def export(self, print_ready_pages, output_file_name_prefix):
        pass


class MatplotlibBasedExporter(PolyzamboniExporter):
    """ 
    This exporter makes use of the matplotlib package. 
    
    If anyone reads this, feel free to write your own exporter that does not require this package to be installed 
    """

    def export(self, print_ready_pages, output_file_name_prefix):
        
        if self.output_format == "pdf":
            with PdfPages(output_file_name_prefix + "." + self.output_format) as pdf:
                # todo save all pages
                pass

        print("this is a test")


if __name__ == "__main__":
    testExportet = MatplotlibBasedExporter()


    testExportet.export("test", "test")
