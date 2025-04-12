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

import math
import platform

import moderngl
import quickwindow
import glm
from pyglsl import VertexStage, FragmentStage
import numpy as np

from spritekit.shader import *
from spritekit.cache import *

def _rotate_point(x, y, c, s):
    return glm.vec2(x * c - y * s, x * s + y * c)

def _normalise_texcoords(clip, texture_size):
    tx, ty = texture_size
    cx, cy, cw, ch = clip
    nx = cx * cw
    ny = cy * ch
    return (nx / tx,
            1.0 - (ny + ch) / ty,
            (nx + cw) / tx,
            1.0 - ny / ty)

def rect_vertices(x, y, w, h, rotation=0., scale=1., clip=(0, 0, 1, 1), texture_size=(1, 1), color=(1, 1, 1, 1)):
    hw = w / 2
    hh = h / 2
    x1 = (x - hw) * scale
    x2 = (x + hw) * scale
    y1 = (y - hh) * scale
    y2 = (y + hh) * scale
    c = math.cos(rotation)
    s = math.sin(rotation)
    p1 = _rotate_point(x1, y1, c, s)
    p2 = _rotate_point(x2, y1, c, s)
    p3 = _rotate_point(x1, y2, c, s)
    p4 = _rotate_point(x2, y2, c, s)
    tc = _normalise_texcoords(clip, texture_size)
    return [*p1, tc[0], tc[1], *color,
            *p2, tc[2], tc[1], *color,
            *p3, tc[0], tc[3], *color,
            *p3, tc[0], tc[3], *color,
            *p4, tc[2], tc[3], *color,
            *p2, tc[2], tc[1], *color]

class Batch:
    def __init__(self, program, mvp, texture=None):
        self._ctx = moderngl.get_context()
        self._program = program
        self._mvp = mvp
        self._texture = texture
        self._vertices = []
    
    def add(self, vertices):
        assert len(vertices) % 8 == 0
        if self._texture is None:
            self._vertices.extend(vertices)
        else:
            vertices.extend(self._vertices)
            self._vertices = vertices
    
    def flush(self):
        if not self._vertices:
            return
        vbo = self._ctx.buffer(np.array(self._vertices, dtype=np.float32).tobytes())
        vao = self._ctx.vertex_array(self._program, [(vbo, '2f 2f 4f', 'position', 'texcoords', 'in_color')])
        if self._texture is not None:
            self._ctx.enable(self._ctx.DEPTH_TEST)
            self._ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            vao.program['use_texture'] = 1 
            self._texture.use()
        else:
            self._ctx.disable(self._ctx.DEPTH_TEST)
            vao.program['use_texture'] = 0
        vao.program['mvp'].write(self._mvp)
        vao.render()

class Renderer:
    def __init__(self):
        self._ctx = moderngl.get_context()
        self._program = self._ctx.program(vertex_shader=VertexStage(default_vertex).compile(),
                                          fragment_shader=FragmentStage(default_fragment).compile())
        self.size = quickwindow.size()
        self._view = None
        self._dirty = False
        self._update_view()
        self._clear_color = glm.vec4(0., 0., 1., 1.)
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
    def size(self, value):
        self._size = value
        if platform.system() == "Darwin":
            self._size = (self._size[0] * 2, self._size[1] * 2)
        self._dirty = True
    
    @property
    def clear_color(self):
        return self._clear_color
    
    @clear_color.setter
    def clear_color(self, value):
        self._clear_color = tuple(min(max(float(v) / 255. if isinstance(v, int) else v, 0.), 1.) for v in [*value])
    
    @property
    def view(self):
        return self._view
    
    def draw(self, vertices, texture=None):
        if self._current_batch is None or self._current_batch._texture != texture:
            if self._current_batch is not None:
                self._batches.append(self._current_batch)
            self._current_batch = Batch(self._program, self._view, texture)
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