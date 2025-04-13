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
from typing import Optional

from .renderer import (line_vertices, rect_vertices, ellipse_vertices, circle_vertices,
                       polygon_vertices, rect_outline_vertices, ellipse_outline_vertices,
                       circle_outline_vertices, polygon_outline_vertices)
from .drawable import Drawable

import glm

class Line(Drawable):
    _generator = line_vertices
    _outline_generator = line_vertices

    def __init__(self,
                 position: glm.vec2 | list[float] | tuple[float, float],
                 end: glm.vec2 | list[float] | tuple[float, float],
                 thickness: float = 1.,
                 **kwargs):
        super().__init__(position=position, **kwargs)
        self._end = end
        self._thickness = thickness

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

    def draw(self):
        self._draw([*self._position, *self.end, self._color, self.thickness])
        super().draw()

class Rect(Drawable):
    _generator = rect_vertices
    _outline_generator = rect_outline_vertices

    def __init__(self,
                 position: glm.vec2 | list | tuple,
                 size: glm.vec2 | list | tuple,
                 **kwargs):
        super().__init__(position=position, **kwargs)
        assert len(size) == 2, "Size must be a 2D vector"
        self._size = glm.vec2(*size)
    
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: glm.vec2 | list | tuple):
        assert len(value) == 2, "Size must be a 2D vector"
        self._size = glm.vec2(*value)
        self._dirty = True

    def draw(self):
        self._draw([*self._position, *self._size, self._rotation, self._scale, (0., 0., 1., 1.), self._color])
        super().draw()

class Circle(Drawable):
    _generator = circle_vertices
    _outline_generator = circle_outline_vertices
    
    def __init__(self,
                 position: glm.vec2 | list | tuple,
                 diameter: float = 1.,
                 radius: Optional[float] = None,
                 segments: int = 32,
                 **kwargs):
        super().__init__(position=position, **kwargs)
        self._diameter = diameter if radius is None else radius * 2.
        assert self._diameter > 0., "Diameter must be greater than 0"
        assert segments >= 3, "Segments must be at least 3"
        self._segments = segments

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, value: float | int):
        self._diameter = value
        self._dirty = True

    @property
    def radius(self):
        return self._diameter / 2.

    @radius.setter
    def radius(self, value: float | int):
        self._diameter = value * 2.
        self._dirty = True

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, value: int):
        assert value >= 3, "Segments must be at least 3"
        self._segments = value
        self._dirty = True

    def draw(self):
        self._draw([*self._position, self._diameter, self._rotation, self._scale, self._color, self.segments])
        super().draw()

class Ellipse(Drawable):
    _generator = ellipse_vertices
    _outline_generator = ellipse_outline_vertices

    def __init__(self,
                 position: glm.vec2 | list | tuple,
                 width: float = 1.,
                 height: float = 1.,
                 segments: int = 32,
                 **kwargs):
        super().__init__(position=position, **kwargs)
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

    def draw(self):
        self._draw([*self._position, self._width, self._height, self._rotation, self._scale, self._color, self.segments])
        super().draw()

class Polygon(Drawable):
    _generator = polygon_vertices
    _outline_generator = polygon_outline_vertices
 
    def __init__(self,
                 position: glm.vec2 | list | tuple,
                 points: list | tuple,
                 sort: bool = False,
                 **kwargs):
        super().__init__(position=position, **kwargs)
        self._sort = sort
        self._set_points(points)
        assert len(self._points) == len(points), "All points must be 2D vectors"
        if sort:
            self.sort()
    
    def _set_points(self, points: list | tuple):
        self._points = [glm.vec2(*p) for p in points if len(p) == 2]
        self._dirty = True
    
    def sort(self):
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
            self.sort()
    
    def add_point(self, points: list | tuple):
        for p in points:
            assert len(p) == 2, "Point must be a 2D vector"
            self._points.extend(glm.vec2(*p))
        if self._sort:
            self.sort()
        self._dirty = True

    def draw(self):
        self._draw([*self._position, self._points, self._rotation, self._scale, self._color])
        super().draw()

__all__ = ['Line', 'Rect', 'Circle', 'Ellipse', 'Polygon']