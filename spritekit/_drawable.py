# spritekit/actors.py
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

from .actor import Actor
from . import _renderer as renderer

import glm

def convert_color(color: tuple | list):
    assert 3 <= len(color) <= 4, "Color must be a list of 3 or 4 values"
    return tuple(min(max(v if isinstance(v, float) else float(v) / 255., 0.), 1.) for v in (color if len(color) == 4 else (*color, 1.)))

class Drawable(Actor):
    def __init__(self,
                 position: glm.vec2 | list | tuple = (0., 0.),
                 rotation: float = 0.,
                 degrees: bool = False,
                 scale: float = 1.,
                 color: list | tuple = (1., 1., 1., 1.),
                 wireframe: bool = False,
                 **kwargs):
        self._dirty = True
        self._vertices = []
        self._texture = None
        super().__init__(**kwargs)
        assert len(position) == 2, "Position must be a 2D vector"
        self._position = glm.vec2(*position)
        self._rotation = math.radians(rotation) if degrees else rotation
        self._degrees = degrees
        self._scale = scale
        self._color = convert_color(color)
        self._wireframe = wireframe

    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value: glm.vec2 | tuple[float, float]):
        self._position = glm.vec2(*value)
        self._dirty = True

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value: float):
        self._rotation = value if not self._degrees else math.radians(value)
        self._dirty = True

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value: float):
        self._scale = value
        self._dirty = True

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value: list | tuple):
        self._color = convert_color(value)
        self._dirty = True

    @property
    def wireframe(self):
        return self._wireframe

    @wireframe.setter
    def wireframe(self, value: bool):
        self._wireframe = value
        self._dirty = True
    
    def _generate_vertices(self):
        return [*self._position, 0., 0., *self._color]

    def _generate_outline_vertices(self):
        return [*self._position, 0., 0., *self._color]

    def draw(self):
        if self._dirty:
            if self._wireframe:
                self._vertices = self._generate_outline_vertices()
            else:
                self._vertices = self._generate_vertices()
            self._dirty = False
        renderer.draw(self._vertices, None if self._wireframe else self._texture)
        super().draw()

