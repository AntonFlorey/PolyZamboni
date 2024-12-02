import bpy
import gpu
import bgl
from gpu_extras.batch import batch_for_shader
from . import pz_globals

# Keep track of active draw callbacks 
_drawing_handle_errors = None

class ColorGenerator():
    """ A simple color generator"""

    def __init__(self):
        self.colors = {
            "BLUE" :        (  0 / 255,  84 / 255, 159 / 255, 1),
            "MAGENTA" :     (227 / 255,   0 / 255, 102 / 255, 1),
            "YELLOW" :      (255 / 255, 237 / 255,   0 / 255, 1),
            "PETROL" :      (  0 / 255,  97 / 255, 101 / 255, 1),
            "TEAL" :        (  0 / 255, 152 / 255, 161 / 255, 1),
            "GREEN" :       ( 87 / 255, 171 / 255,  39 / 255, 1),
            "MAY_GREEN" :   (189 / 255, 205 / 255,   0 / 255, 1),
            "ORANGE" :      (246 / 255, 168 / 255,   0 / 255, 1),
            "RED" :         (204 / 255,   7 / 255,  30 / 255, 1),
            "BORDEAUX" :    (161 / 255,  16 / 255,  53 / 255, 1),
            "PURPLE" :      ( 97 / 255,  33 / 255,  88 / 255, 1),
            "LILAC" :       (122 / 255, 111 / 255, 172 / 255, 1)
        }
        self.index = 0

    def next_color(self):
        col = list(self.colors.values())[self.index]
        self.index = (self.index + 1) % len(self.colors)
        return col


def draw_errors():
    print("drawing not implemented yet")