# spritekit/shader.py
#
# Copyright (C) 2025 George Watson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pyglsl.glsl import *
from pyglsl import ShaderDef

class VsAttrs(AttributeBlock):
    position = vec2()
    texcoords = vec2()
    in_color = vec4()

class VsUniforms(UniformBlock):
    view = mat4()
    world = mat4()

class VsOut(ShaderInterface):
    gl_Position = vec4()
    out_texcoords = vec2()
    out_color = vec4()

def default_vertex(attr: VsAttrs, uniforms: VsUniforms) -> VsOut:
    return VsOut(gl_Position=uniforms.view * uniforms.world * vec4(attr.position, 0., 1.),
                    out_texcoords=attr.texcoords,
                    out_color=attr.in_color)

class FsUniforms(UniformBlock):
    in_buffer = sampler2D()
    use_texture = int()

class FsOut(FragmentShaderOutputBlock):
    out_color = vec4()

def default_fragment(vs_out: VsOut, uniforms: FsUniforms) -> FsOut:
    if uniforms.use_texture != 0:
        color = vec4(texture(uniforms.in_buffer, vs_out.out_texcoords) * vec4(1., 1., 1., vs_out.out_color.a))
        # noinspection PyTypeChecker
        if color.a <= 0: # bug in pyglsl, can't use `==` in if statement
            # noinspection PyStatementEffect
            discard
        return FsOut(out_color=color)
    else:
        return FsOut(out_color=vs_out.out_color)

default_shader = ShaderDef(vertex_shader=default_vertex, fragment_shader=default_fragment)

__all__ = ['default_shader']