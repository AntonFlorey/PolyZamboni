import numpy as np
import bpy
import bmesh
import math
import mathutils
from itertools import combinations
import sympy as sp
import time
import torch
import torch.nn as nn
from . import geometry


class ObjectiveFunction(nn.Module):
    """
    A class that stores the flattening objective function for a mesh.
    It offers evaluation of
    - Value
    - Gradient
    - Hessian
    """

    def __init__(self, ao : bpy.types.Object, closeness_weight, angle_weight, det_weight):
        super().__init__()
        self.mesh : bmesh.types.BMesh = bmesh.new()
        bpy.ops.object.mode_set(mode="EDIT")
        self.mesh = bmesh.from_edit_mesh(ao.data).copy()
        bpy.ops.object.mode_set(mode="OBJECT")
        self.closeness_w = closeness_weight
        self.angle_w = angle_weight
        self.det_w = det_weight
        self.bb_diameter = geometry.compute_bb_diameter(self.mesh)
        self.voronoi_areas = geometry.compute_voronoi_areas(self.mesh)

        # create variables
        self.vertex_to_variables = {}
        self.initial_vertex_positions = []
        self.vertex_weights = []

        for vertex in self.mesh.verts:
            self.vertex_to_variables[vertex.index] = len(self.vertex_to_variables)
            v_pos = np.asarray(vertex.co).reshape((3,))
            self.initial_vertex_positions.append(v_pos)
            self.vertex_weights.append(self.voronoi_areas[vertex.index])

        self.initial_vertex_positions = torch.Tensor(np.asarray(self.initial_vertex_positions))
        self.vertex_weights = torch.Tensor(np.asarray(self.vertex_weights))

        self.weights = nn.Parameter(self.initial_vertex_positions.detach(), requires_grad=True)  # [vertex, xyz]

    def forward(self):
        # add objective function elements 
        # face_vertices: [face_id, vertex_id] padded with -1

        dist = (self.weights - self.initial_vertex_positions.detach())  # [vertex, xyz]
        dist = torch.einsum("i,ik->ik", torch.sqrt(self.vertex_weights.detach()), dist)
        closeness = torch.einsum("ij,ik->", dist, dist)

        angle_term = 0
        det_term = 0
        for face in self.mesh.faces:
            if len(face.verts) == 3:
                continue

            v_ids = [v.index for v in face.verts]
            n = len(v_ids)

            if self.angle_w != 0:
                planarity_f = - (n - 2) * np.pi
                for i in range(n):
                    a_id = v_ids[(i + n - 1) % n]
                    b_id = v_ids[i]
                    c_id = v_ids[(i + 1) % n]

                    v_a = self.weights[self.vertex_to_variables[a_id]]
                    v_b = self.weights[self.vertex_to_variables[b_id]]
                    v_c = self.weights[self.vertex_to_variables[c_id]]
                    
                    ba = (v_a - v_b) / torch.sqrt((v_a - v_b).dot(v_a - v_b))
                    bc = (v_c - v_b) / torch.sqrt((v_c - v_b).dot(v_c - v_b))

                    angle = torch.acos(ba.dot(bc))
                    planarity_f += angle
                angle_term += planarity_f**2
            
            if self.det_w == 0:
                continue
                
            normalized_edges = []
            for i in range(n):
                a_id = v_ids[i]
                b_id = v_ids[(i + 1) % n]
                v_a = self.weights[self.vertex_to_variables[a_id]]
                v_b = self.weights[self.vertex_to_variables[b_id]]
                normalized_edges.append((v_b - v_a) / torch.linalg.norm(v_b - v_a))
            for a, b, c in combinations(normalized_edges, 3):
                det_term += torch.square(torch.linalg.det(torch.stack((a,b,c))))


        final_energy = self.closeness_w * closeness / (geometry.compute_mesh_surface_area(self.mesh)**2)
        if self.angle_w != 0:
            final_energy += self.angle_w * angle_term / len(self.mesh.faces)
        if self.det_w != 0:
            final_energy += self.det_w * det_term / len(self.mesh.faces)

        return final_energy

    def apply_back_to_mesh(self, ao : bpy.types.Object):
        for vertex in self.mesh.verts:
            new_pos = self.weights[self.vertex_to_variables[vertex.index]]
            vertex.co = tuple(new_pos.tolist())
            
        bpy.ops.object.mode_set(mode="OBJECT")
        self.mesh.to_mesh(ao.data)
        bpy.ops.object.mode_set(mode="EDIT")
