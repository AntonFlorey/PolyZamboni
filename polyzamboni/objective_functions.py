import numpy as np
import bpy
import bmesh
import math
import mathutils
import sympy as sp
import time
import torch

print("all imports working!")

class ObjectiveFunction():
    """
    A class that stores the flattening objective function for a mesh.
    It offers evaluation of
    - Value
    - Gradient
    - Hessian
    """

    def __init__(self, ao : bpy.types.Object):
        self.mesh : bmesh.types.BMesh = bmesh.new()
        bpy.ops.object.mode_set(mode="EDIT")
        self.mesh = bmesh.from_edit_mesh(ao.data).copy()
        bpy.ops.object.mode_set(mode="OBJECT")

        # create variables
        self.vertex_to_variables = {}
        self.variables = []

        for vertex in self.mesh.verts:
            suffix = str(vertex.index)
            self.vertex_to_variables[vertex.index] = len(self.variables)
            self.variables.append(sp.symbols("x_" + suffix))
            self.variables.append(sp.symbols("y_" + suffix))
            self.variables.append(sp.symbols("z_" + suffix))

        # add objective function elements
        self.closeness_terms = []

        closeness_start_time = time.time()
        for vertex in self.mesh.verts:
            current_elem = {}
            v_id = vertex.index
            v_pos = np.asarray(vertex.co).reshape((3,1))

            x_var = self.variables[self.vertex_to_variables[v_id] + 0]
            y_var = self.variables[self.vertex_to_variables[v_id] + 1]
            z_var = self.variables[self.vertex_to_variables[v_id] + 2]

            v_var = sp.Matrix([x_var, y_var, z_var])

            # value
            closeness_f = (v_var - v_pos).dot(v_var - v_pos)
            closeness_f_lambdified = sp.lambdify((x_var, y_var, z_var), closeness_f, "numpy")
            # gradient
            closeness_gradient = sp.Matrix([closeness_f]).jacobian([x_var, y_var, z_var])
            closeness_gradient_lambdified = sp.lambdify((x_var, y_var, z_var), closeness_gradient, "numpy")
            # hessian
            closeness_hessian = sp.hessian(closeness_f, [x_var, y_var, z_var])
            closeness_hessian_lambdified = sp.lambdify((x_var, y_var, z_var), closeness_hessian, "numpy")

            current_elem["f"] = closeness_f
            current_elem["g"] = closeness_gradient
            current_elem["H"] = closeness_hessian
            current_elem["f_np"] = closeness_f_lambdified
            current_elem["g_np"] = closeness_gradient_lambdified
            current_elem["H_np"] = closeness_hessian_lambdified
            current_elem["variables"] = [x_var, y_var, z_var]
            current_elem["variable_indices"] = [self.vertex_to_variables[v_id] + 0,
                                                self.vertex_to_variables[v_id] + 1,
                                                self.vertex_to_variables[v_id] + 2]

            self.closeness_terms.append(current_elem)

        closeness_time = time.time() - closeness_start_time
        print("Closeness term preperation took ", closeness_time, "seconds.")


        self.angle_based_planarity_terms = []

        planarity_start_time = time.time()
        for face in self.mesh.faces:
            if (len(face.verts) == 3):
                continue

            v_ids = [v.index for v in face.verts]
            used_variables = []    
            used_variable_indices = []

            for v_id in v_ids:
                used_variables.append(self.variables[self.vertex_to_variables[v_id] + 0])
                used_variables.append(self.variables[self.vertex_to_variables[v_id] + 1])
                used_variables.append(self.variables[self.vertex_to_variables[v_id] + 2])
                used_variable_indices.append(self.vertex_to_variables[v_id] + 0)
                used_variable_indices.append(self.vertex_to_variables[v_id] + 1)
                used_variable_indices.append(self.vertex_to_variables[v_id] + 2)

            
            n = len(v_ids)
            planarity_f = - (n - 2) * sp.pi
            for i in range(len(v_ids)):
                a_id = v_ids[(i + n - 1) % n]
                b_id = v_ids[i]
                c_id = v_ids[(i + 1) % n]

                v_a = sp.Matrix([self.variables[self.vertex_to_variables[a_id] + 0], 
                                 self.variables[self.vertex_to_variables[a_id] + 1],
                                 self.variables[self.vertex_to_variables[a_id] + 2]])
                v_b = sp.Matrix([self.variables[self.vertex_to_variables[b_id] + 0], 
                                 self.variables[self.vertex_to_variables[b_id] + 1],
                                 self.variables[self.vertex_to_variables[b_id] + 2]])
                v_c = sp.Matrix([self.variables[self.vertex_to_variables[c_id] + 0], 
                                 self.variables[self.vertex_to_variables[c_id] + 1],
                                 self.variables[self.vertex_to_variables[c_id] + 2]])
                
                ba = (v_a - v_b) / sp.sqrt((v_a - v_b).dot(v_a - v_b))
                bc = (v_c - v_b) / sp.sqrt((v_c - v_b).dot(v_c - v_b))

                angle = sp.acos(ba.dot(bc))
                planarity_f += angle

            planarity_f = planarity_f ** 2
            planarity_f_lambdified = sp.lambdify(used_variables, planarity_f, "numpy")
            print(planarity_f)
            # gradient
            planarity_gradient = sp.Matrix([planarity_f]).jacobian(used_variables)
            planarity_gradient_lambdified = sp.lambdify(used_variables, planarity_gradient, "numpy")
            print(planarity_gradient)
            # hessian
            planarity_hessian = sp.hessian(planarity_f, used_variables)
            planarity_hessian_lambdified = sp.lambdify(used_variables, planarity_hessian, "numpy")
            print(planarity_hessian)

        planarity_time = time.time() - planarity_start_time
        print("Planarity term preperation took ", planarity_time, "seconds.")


    def eval_passive(self, x):
        print("TODO")

    def eval_with_derivatives(self, x):
        print("TODO")


