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

def _rotate_point(x, y, c, s):
    return glm.vec2(x * c - y * s, x * s + y * c)

def _normalise_texcoords(clip):
    cx, cy, cw, ch = clip
    nx = cx * cw
    ny = cy * ch
    return (nx / 1.,
            1.0 - (ny + ch) / 1.,
            (nx + cw) / 1.,
            1.0 - ny / 1.)

def _rect_points(x, y, w, h, rotation=0., scale=1.):
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
    return p1, p2, p3, p4

def rect_vertices(x, y, w, h, rotation=0., scale=1., clip=(0, 0, 1, 1), color=(1, 1, 1, 1)):
    p1, p2, p3, p4 = _rect_points(x, y, w, h, rotation, scale)
    tc = _normalise_texcoords(clip)
    return [*p1, tc[0], tc[1], *color,
            *p2, tc[2], tc[1], *color,
            *p3, tc[0], tc[3], *color,
            *p3, tc[0], tc[3], *color,
            *p4, tc[2], tc[3], *color,
            *p2, tc[2], tc[1], *color]

def rect_outline_vertices(x, y, w, h, rotation=0., scale=1., clip=(0, 0, 1, 1), color=(1, 1, 1, 1)):
    p1, p2, p3, p4 = _rect_points(x, y, w, h, rotation, scale)
    return [*line_vertices(*p1, *p2, color),
            *line_vertices(*p3, *p4, color),
            *line_vertices(*p1, *p3, color),
            *line_vertices(*p2, *p4, color)]

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

def ellipse_outline_vertices(x, y, width, height, rotation=0., scale=1., color=(1, 1, 1, 1), segments=32):
    centre = [x, y, 0., 0., *color]
    step = 2 * math.pi / segments
    rx = width / 2. * scale
    ry = height / 2. * scale
    c = math.cos(rotation)
    s = math.sin(rotation)
    p = glm.vec2(x, y)
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

def circle_outline_vertices(x, y, radius, rotation=0., scale=1., color=(1, 1, 1, 1), segments=32):
    return ellipse_outline_vertices(x, y, radius, radius, rotation, scale, color, segments)

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
    position = glm.vec2(x, y)
    t = [glm.vec2(*p) - position * scale for p in points]
    f = [glm.vec2(*_rotate_point(*p, c, s)) + p for p in t]
    centroid_vertex = [*_polygon_centroid(f), 0, 0, *color]
    vertices = []
    for i in range(len(f)):
        vertices.extend([*centroid_vertex,
                         *f[i], 0, 0, *color,
                         *f[(i + 1) % len(f)], 0, 0, *color])
    return vertices

def polygon_outline_vertices(x, y, points, rotation=0., scale=1., color=(1, 1, 1, 1)):
    c = math.cos(rotation)
    s = math.sin(rotation)
    position = glm.vec2(x, y)
    t = [glm.vec2(*p) - position * scale for p in points]
    f = [glm.vec2(*_rotate_point(*p, c, s)) + p for p in t]
    vertices = []
    for i in range(len(f)):
        vertices.extend(line_vertices(*f[i], *f[(i + 1) % len(f)], color))
    return vertices

class _Batch:
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
    
    def draw(self, vertices, texture=None):
        if self._current_batch is None or self._current_batch._texture != texture:
            if self._current_batch is not None:
                self._batches.append(self._current_batch)
            self._current_batch = _Batch(self._program, self._view, texture)
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

def init_renderer(viewport: Optional[tuple[int | float, int | float]] = None,
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
def draw(vertices, texture=None):
    __renderer__.draw(vertices, texture)

@_check_renderer
def flush():
    __renderer__.flush()

__all__ = [
    'rect_vertices',
    'rect_outline_vertices',
    'line_vertices',
    'ellipse_vertices',
    'ellipse_outline_vertices',
    'circle_vertices',
    'circle_outline_vertices',
    'polygon_vertices',
    'polygon_outline_vertices',
    'Renderer'
]