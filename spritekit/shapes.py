# spritekit/shapes.py
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

from . import _drawable as drawable

from pyglm import glm

def _line_vertices(x1, y1, x2, y2, color=(1, 1, 1, 1), thickness=1.):
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

class LineActor(drawable.Drawable):
    def __init__(self,
                 end: glm.vec2 | list[float] | tuple[float, float] = (0., 0.),
                 **kwargs):
        super().__init__(**kwargs)
        self._end = end

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value: glm.vec2 | tuple | list):
        assert len(value) == 2, "End must be a 2D vector"
        self._end = glm.vec2(*value)
        self._dirty = True

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value: float):
        self._thickness = value
        self._dirty = True
    
    def _generate_vertices(self):
        return _line_vertices(*self._position, *self._end, self._color, self._thickness)
    
    def _generate_outline_vertices(self):
        return self._generate_vertices()

class RectActor(drawable.Drawable):
    def __init__(self,
                 size: glm.vec2 | list | tuple = (1., 1.),
                 **kwargs):
        super().__init__(**kwargs)
        assert len(size) == 2, "Size must be a 2D vector"
        self.size = size
    
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: glm.vec2 | list | tuple):
        assert len(value) == 2, "Size must be a 2D vector"
        self._size = glm.vec2(*value)
        self._dirty = True

    @property
    def points(self):
        hw = self._size.x / 2
        hh = self._size.y / 2
        x1 = (self._position.x - hw) * self._scale
        x2 = (self._position.x + hw) * self._scale
        y1 = (self._position.y - hh) * self._scale
        y2 = (self._position.y + hh) * self._scale
        c = math.cos(self._rotation)
        s = math.sin(self._rotation)
        p1 = _rotate_point(x1, y1, c, s)
        p2 = _rotate_point(x2, y1, c, s)
        p3 = _rotate_point(x1, y2, c, s)
        p4 = _rotate_point(x2, y2, c, s)
        return p1, p2, p3, p4
    
    def _generate_vertices(self):
        p1, p2, p3, p4 = self.points
        tc = _normalise_texcoords((0, 0, 1, 1))
        return [*p1, tc[0], tc[1], *self._color,
                *p2, tc[2], tc[1], *self._color,
                *p3, tc[0], tc[3], *self._color,
                *p3, tc[0], tc[3], *self._color,
                *p4, tc[2], tc[3], *self._color,
                *p2, tc[2], tc[1], *self._color]
    
    def _generate_outline_vertices(self):
        p1, p2, p3, p4 = self.points
        return [*_line_vertices(*p1, *p2, *self._color, self._thickness),
                *_line_vertices(*p3, *p4, *self._color, self._thickness),
                *_line_vertices(*p1, *p3, *self._color, self._thickness),
                *_line_vertices(*p2, *p4, *self._color, self._thickness)]

class EllipseActor(drawable.Drawable):
    def __init__(self,
                 width: float = 1.,
                 height: float = 1.,
                 segments: int = 32,
                 **kwargs):
        super().__init__(**kwargs)
        self._width = width
        self._height = height
        assert segments >= 3, "Segments must be at least 3"
        self._segments = segments

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = value
        self._dirty = True

    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value: float):
        self._height = value
        self._dirty = True

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, value: int):
        assert value >= 3, "Segments must be at least 3"
        self._segments = value
        self._dirty = True

    @property
    def points(self):
        step = 2 * math.pi / self._segments
        rx = self._width / 2. * self._scale
        ry = self._height / 2. * self._scale
        c = math.cos(self._rotation)
        s = math.sin(self._rotation)
        p = glm.vec2(self._position.x, self._position.y)
        return [p + _rotate_point(rx * math.cos(i * step), ry * math.sin(i * step), c, s) for i in range(self._segments)]

    def _generate_vertices(self):
        centre = [*self._position, 0., 0., *self._color]
        points = self.points
        vertices = []
        for i in range(len(points)):
            vertices.extend([*centre,
                             *points[i], 0., 0., *self._color,
                             *points[(i + 1) % len(points)], 0., 0., *self._color])
        return vertices

    def _generate_outline_vertices(self):
        points = self.points
        vertices = []
        for i in range(len(points)):
            vertices.extend(*_line_vertices(*points[i], *(points[(i + 1) % len(points)]), *self._color, self._thickness))
        return vertices

class CircleActor(EllipseActor):
    def __init__(self,
                 radius: float = 1.,
                 **kwargs):
        kwargs['width'] = radius
        kwargs['height'] = radius
        super().__init__(**kwargs)
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value: float):
        self._radius = value
        self._dirty = True
    
    
    @property
    def width(self):
        return self._radius

    @width.setter
    def width(self, value: float):
        self._radius = value
        self._dirty = True
    
    @property
    def height(self):
        return self._radius

    @height.setter
    def height(self, value: float):
        self._radius = value
        self._dirty = True

class PolygonActor(drawable.Drawable):
    def __init__(self,
                 points: list | tuple,
                 sort: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self._sort = sort
        self._points = []
        self._set_points(points)
        assert len(self._points) == len(points), "All points must be 2D vectors"
        if sort:
            self._sort_points()
    
    def _set_points(self, points: list | tuple):
        self._points = [glm.vec2(*p) - self._position * self._scale for p in points if len(p) == 2]
        self._dirty = True

    def _sort_points(self):
        self._points = sorted(self._points, key=lambda p: math.atan2(p.y, p.x))
        self._dirty = True

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value: list | tuple):
        assert len(value) >= 3, "Polygon must have at least 3 points"
        self._set_points(value)
        if self._sort:
            self._sort_points()
    
    def add_point(self, points: list | tuple):
        for p in points:
            assert len(p) == 2, "Point must be a 2D vector"
            self._points.extend(glm.vec2(*p))
        if self._sort:
            self._sort_points()
        self._dirty = True

    @property
    def centroid(self):
        area = 0.
        centroid = glm.vec2(0., 0.)
        n = len(self._points)
        for i in range(n):
            x1, y1 = self._points[i]
            x2, y2 = self._points[(i + 1) % n]
            p = (x1 * y2) - (x2 * y1)
            area += p
            centroid += glm.vec2((x1 + x2) * p, (y1 + y2) * p)
        return self._points[0] if n == 1 else sum(self._points) / n if area / 2. == 0. else centroid / (6. * area)

    def _generate_vertices(self):
        c = math.cos(self._rotation)
        s = math.sin(self._rotation)
        t = [glm.vec2(*p) for p in self._points]
        f = [glm.vec2(*_rotate_point(*p, c, s)) + p for p in t]
        centroid_vertex = [*self.centroid, 0, 0, *self._color]
        vertices = []
        for i in range(len(f)):
            vertices.extend([*centroid_vertex,
                             *f[i], 0, 0, *self._color,
                             *f[(i + 1) % len(f)], 0, 0, *self._color])
        return vertices


    def _generate_outline_vertices(self):
        sorted_points = sorted(self._points, key=lambda p: math.atan2(p.y, p.x))
        vertices = []
        for i in range(len(sorted_points)):
            vertices.extend(*_line_vertices(*sorted_points[i], *(sorted_points[(i + 1) % len(sorted_points)]), *self._color, self._thickness))
        return vertices

__all__ = ['LineActor', 'RectActor', 'CircleActor', 'EllipseActor', 'PolygonActor']