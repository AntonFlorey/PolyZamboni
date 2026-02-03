import gpu

def create_2d_textured_triangle_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("custom_interface")
    vert_out.smooth('VEC2', "uvInterp")
    vert_out.smooth('VEC4', "colorInterp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "viewProjectionMatrix")
    shader_info.sampler(0, 'FLOAT_2D', "image")
    shader_info.vertex_in(0, 'VEC2', "pos")
    shader_info.vertex_in(1, 'VEC2', "uv")
    shader_info.vertex_in(2, 'VEC4', "color")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "FragColor")

    shader_info.vertex_source(
        "void main()"
        "{"
        "  uvInterp = uv;"
        "  gl_Position = viewProjectionMatrix * vec4(pos, 0.0, 1.0);"
        "  colorInterp = color;"
        "}"
    )

    shader_info.fragment_source(
        "void main()"
        "{"
        "  vec4 tex = texture(image, uvInterp);"
        "  FragColor = colorInterp * tex;"
        "}"
    )

    colored_triangle_shader = gpu.shader.create_from_info(shader_info)
    return colored_triangle_shader

def create_dashed_lines_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("custom_interface")
    vert_out.smooth('FLOAT', "v_ArcLengthInterp")
    vert_out.smooth('VEC4', "v_ColorInterp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "viewProjectionMatrix")
    shader_info.push_constant('FLOAT', "dashLength")
    shader_info.vertex_in(0, 'VEC3', "pos")
    shader_info.vertex_in(1, 'FLOAT', "arcLength")
    shader_info.vertex_in(2, 'VEC4', "color")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "FragColor")

    shader_info.vertex_source(
        "void main()"
        "{"
        "  v_ArcLengthInterp = arcLength;"
        "  v_ColorInterp = color;"
        "  gl_Position = viewProjectionMatrix * vec4(pos, 1.0);"
        "}"
    )

    shader_info.fragment_source(
        "const float PI = 3.14159265359;"
        "void main()"
        "{"
        "  if (step(sin(v_ArcLengthInterp * PI / dashLength), 0) == 1) discard;"
        "  FragColor = v_ColorInterp;"
        "}"
    )

    shader = gpu.shader.create_from_info(shader_info)
    return shader

def create_uniform_color_dashed_lines_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("custom_interface")
    vert_out.smooth('FLOAT', "v_ArcLengthInterp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "viewProjectionMatrix")
    shader_info.push_constant('FLOAT', "dashLength")
    shader_info.push_constant('VEC4', "color")
    shader_info.vertex_in(0, 'VEC3', "pos")
    shader_info.vertex_in(1, 'FLOAT', "arcLength")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "FragColor")

    shader_info.vertex_source(
        "void main()"
        "{"
        "  v_ArcLengthInterp = arcLength;"
        "  gl_Position = viewProjectionMatrix * vec4(pos, 1.0);"
        "}"
    )

    shader_info.fragment_source(
        "void main()"
        "{"
        "  if (step(sin(v_ArcLengthInterp * 3.14159265359 / dashLength), 0.0) == 1) discard;"
        "  FragColor = color;"
        "}"
    )

    shader = gpu.shader.create_from_info(shader_info)
    return shader

def create_dash_dot_lines_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("custom_interface")
    vert_out.smooth('FLOAT', "v_ArcLengthInterp")
    vert_out.smooth('VEC4', "v_ColorInterp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "viewProjectionMatrix")
    shader_info.push_constant('FLOAT', "dashLength")
    shader_info.vertex_in(0, 'VEC3', "pos")
    shader_info.vertex_in(1, 'FLOAT', "arcLength")
    shader_info.vertex_in(2, 'VEC4', "color")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "FragColor")

    shader_info.vertex_source(
        "void main()"
        "{"
        "  v_ArcLengthInterp = arcLength;"
        "  v_ColorInterp = color;"
        "  gl_Position = viewProjectionMatrix * vec4(pos, 1.0);"
        "}"
    )

    shader_info.fragment_source(
        "const float PI = 3.14159265359;"
        "void main()"
        "{"
        "  float s1 = sin(v_ArcLengthInterp * PI / dashLength);"
        "  float s2 = sin(v_ArcLengthInterp * 3.0 * PI / dashLength);"
        "  if (step(s1, 0) == 0 && step(s2, 0) == 0) discard;"
        "  FragColor = v_ColorInterp;"
        "}"
    )

    shader = gpu.shader.create_from_info(shader_info)
    return shader

textured_triangle_shader_2d = create_2d_textured_triangle_shader()
dashed_lines_shader = create_dashed_lines_shader()
dash_dot_lines_shader = create_dash_dot_lines_shader()
uniform_color_dashed_lines_shader = create_uniform_color_dashed_lines_shader()
