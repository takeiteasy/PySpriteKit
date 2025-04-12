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
from typing import Optional

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

def line_vertices(x1, y1, x2, y2, color=(1, 1, 1, 1), thickness=1.):
    v1 = glm.vec2(x1, y1)
    v2 = glm.vec2(x2, y2)
    ab = v2 - v1
    n = glm.normalize(glm.vec2(-ab.y, ab.x)) * (thickness / 2.)
    p1 = v1 + n
    p2 = v1 - n
    p3 = v2 + n
    p4 = v2 - n
    return [*p1, 0, 0, *color,
            *p2, 0, 0, *color,
            *p3, 0, 0, *color,
            *p2, 0, 0, *color,
            *p4, 0, 0, *color,
            *p3, 0, 0, *color]

def ellipse_vertices(x, y, width, height, rotation=0., scale=1., color=(1, 1, 1, 1), segments=32):
    centre = [x, y, 0., 0., *color]
    step = 2 * math.pi / segments
    rx = width / 2. * scale
    ry = height / 2. * scale
    c = math.cos(rotation)
    s = math.sin(rotation)
    p = glm.vec2([x, y])
    vertices = []
    for i in range(segments):
        a1 = i * step
        a2 = (i + 1) * step
        p1 = p + _rotate_point(rx * math.cos(a1), ry * math.sin(a1), c, s)
        p2 = p + _rotate_point(rx * math.cos(a2), ry * math.sin(a2), c, s)
        vertices.extend([*centre,
                         *p1, 0., 0., *color,
                         *p2, 0., 0., *color])
    return vertices

def circle_vertices(x, y, radius, rotation=0., scale=1., color=(1, 1, 1, 1), segments=32):
    return ellipse_vertices(x, y, radius, radius, rotation, scale, color, segments)

def _polygon_centroid(points):
    assert len(points) > 0
    area = 0.
    centroid = glm.vec2(0., 0.)
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        p = (x1 * y2) - (x2 * y1)
        area += p
        centroid += glm.vec2((x1 + x2) * p, (y1 + y2) * p)
    return points[0] if n == 1 else sum(points) / n if area / 2. == 0. else centroid / (6. * area)

def polygon_vertices(x, y, points, rotation=0., scale=1., color=(1, 1, 1, 1)):
    c = math.cos(rotation)
    s = math.sin(rotation)
    position = glm.vec2([x, y])
    t = [glm.vec2(*p) - position * scale for p in points]
    f = [glm.vec2(*_rotate_point(*p, c, s)) + p for p in t]
    centroid_vertex = [*_polygon_centroid(f), 0, 0, *color]
    vertices = []
    for i in range(len(f)):
        vertices.extend([*centroid_vertex,
                         *f[i], 0, 0, *color,
                         *f[(i + 1) % len(f)], 0, 0, *color])
    return vertices

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
    def __init__(self,
                 viewport: Optional[tuple[int | float, int | float]] = None,
                 clear_color: tuple[float, float, float, float] = (0, 0, 0, 1)):
        self._ctx = moderngl.get_context()
        self._program = self._ctx.program(vertex_shader=VertexStage(default_vertex).compile(),
                                          fragment_shader=FragmentStage(default_fragment).compile())
        self.size = quickwindow.size() if viewport is None else viewport
        self._view = None
        self._dirty = False
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