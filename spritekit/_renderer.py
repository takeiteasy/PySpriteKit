# spritekit/renderer.py
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

import platform
from typing import Optional

from .window import window_size
from .shader import default_vertex, default_fragment

import moderngl
import glm
from pyglsl import VertexStage, FragmentStage
import numpy as np

class Batch:
    def __init__(self, program, view, world, texture=None):
        self._ctx = moderngl.get_context()
        self._program = program
        self._view = view
        self._world = world
        self._texture = texture
        self._vertices = []
    
    def add(self, vertices):
        assert len(vertices) % 8 == 0
        if self._texture is None:
            self._vertices.extend(vertices)
        else:
            vertices.extend(self._vertices)
            self._vertices = vertices
    
    @property
    def has_texture(self):
        return self._texture is not None
    
    def flush(self):
        if not self._vertices:
            return
        vbo = self._ctx.buffer(np.array(self._vertices, dtype=np.float32).tobytes())
        vao = self._ctx.vertex_array(self._program, [(vbo, '2f 2f 4f', 'position', 'texcoords', 'in_color')])
        self._ctx.enable(self._ctx.BLEND)
        if self._texture is not None:
            self._ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
            vao.program['use_texture'] = 1 
            self._texture.use()
        else:
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            vao.program['use_texture'] = 0
        self._ctx.disable(self._ctx.DEPTH_TEST)
        vao.program['view'].write(self._view)
        vao.program['world'].write(self._world)
        vao.render()

class Renderer:
    def __init__(self,
                 viewport: Optional[tuple[int | float, int | float]] = None,
                 clear_color: tuple[float, float, float, float] = (0, 0, 0, 1)):
        self._ctx = moderngl.get_context()
        self._program = self._ctx.program(vertex_shader=VertexStage(default_vertex).compile(),
                                          fragment_shader=FragmentStage(default_fragment).compile())
        self.size = window_size() if viewport is None else viewport
        self._view = glm.mat4()
        self._world = glm.mat4()
        self._dirty = True
        self._update_view()
        self.clear_color = glm.vec4(*clear_color)
        self._batches = []
        self._current_batch = None
    
    def _update_view(self):
        halfw, halfh = self._size[0] / 2, self._size[1] / 2
        self._view = glm.ortho(-halfw, halfw, -halfh, halfh, -1., 1.)
        self._dirty = False
    
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value: tuple | list):
        assert len(value) == 2, "Size must be a tuple or list of 2 integers"
        self._size = tuple(value)
        if platform.system() == "Darwin":
            self._size = (self._size[0] * 2, self._size[1] * 2)
        self._dirty = True
    
    @property
    def clear_color(self):
        return self._clear_color
    
    @clear_color.setter
    def clear_color(self, color: tuple | list):
        assert 3 <= len(color) <= 4, "Color must be a list of 3 or 4 floats"
        self._clear_color = tuple(min(max(v if isinstance(v, float) else float(v) / 255., 0.), 1.) for v in (color if len(color) == 4 else (*color, 1.)))
    
    @property
    def view(self):
        return self._view
    
    @property
    def world(self):
        return self._world

    @world.setter
    def world(self, value: glm.mat4):
        self._world = value
    
    def draw(self, vertices, texture=None):
        if self._current_batch is None or self._current_batch._texture != texture:
            if self._current_batch is not None:
                self._batches.append(self._current_batch)
            self._current_batch = Batch(self._program, self._view, self._world, texture)
        self._current_batch.add(vertices)

    def flush(self):
        if self._current_batch is not None:
            self._batches.append(self._current_batch)
        if self._dirty:
            self._update_view()
        self._ctx.clear(viewport=self._size, color=self._clear_color)
        for batch in self._batches:
            batch.flush()
        self._batches = []
        self._current_batch = None

__renderer__ = None

def _check_renderer(func):
    def wrapper(*args, **kwargs):
        if __renderer__ is None:
            raise RuntimeError("Renderer not initialized")
        return func(*args, **kwargs)
    return wrapper

def init(viewport: Optional[tuple[int | float, int | float]] = None,
         clear_color: tuple[float, float, float, float] = (0, 0, 0, 1)):
    global __renderer__
    assert __renderer__ is None, "Renderer already initialized"
    __renderer__ = Renderer(viewport, clear_color)

@_check_renderer
def get_viewport():
    return __renderer__.size

@_check_renderer
def get_clear_color():
    return __renderer__.clear_color

@_check_renderer
def set_viewport(viewport: tuple[int | float, int | float]):
    __renderer__.size = viewport

@_check_renderer
def set_clear_color(clear_color: tuple[float, float, float, float]):
    __renderer__.clear_color = clear_color

@_check_renderer
def set_world_matrix(world: glm.mat4):
    __renderer__.world = world

@_check_renderer
def draw(vertices, texture=None):
    __renderer__.draw(vertices, texture)

@_check_renderer
def flush():
    __renderer__.flush()