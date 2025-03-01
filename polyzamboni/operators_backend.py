"""
This file contains several high level functions that meaningful combine multiple polyzamboni operations.
Still need a better name for this file...
"""

import bpy
from bpy.types import Mesh
import bmesh
import numpy as np
from . import utils
from . import io
from . import cutgraph
from . import unfolding
from . import glueflaps





def compactify_polyzamboni_data(mesh : Mesh):
    """ This may take some time so call this rarely """
    pass # TODO

def update_all_connected_components(mesh : Mesh, preserve_old_indices=True):
    """ Most data stored for the new components might still be usable. If this is NOT the case, set preserve_old_indices to False to save some time. """
    pass # TODO

def update_connected_components_around_edge(mesh : Mesh, edge_index):
    pass # TODO