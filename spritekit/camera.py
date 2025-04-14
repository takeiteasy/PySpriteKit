# spritekit/camera.py
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

import glm

class Camera:
    def __init__(self,
                 position: glm.vec2 | tuple | list = (0., 0.),
                 rotation: float = 0.,
                 zoom: float = 1.):
        assert len(position) == 2, "Position must be a 2D vector"
        self._position = position
        self._rotation = rotation
        self._zoom = zoom
        self._dirty = True
        self._matrix = None

    def _to_matrix(self):
        mat = glm.mat4()
        mat = glm.translate(mat, glm.vec3(*self._position, 0))
        mat = glm.rotate(mat, self._rotation, glm.vec3(0, 0, 1))
        mat = glm.scale(mat, glm.vec3(self._zoom, self._zoom, 1))
        return mat

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: glm.vec2 | tuple | list):
        assert len(value) == 2, "Position must be a 2D vector"
        self._position = value
        self._dirty = True

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value: float):
        self._rotation = value
        self._dirty = True

    @property
    def zoom(self):
        return self._zoom

    @zoom.setter
    def zoom(self, value: float):
        self._zoom = value
        self._dirty = True

    @property
    def dirty(self):
        return self._dirty

    @property
    def matrix(self):
        if self._dirty:
            self._matrix = self._to_matrix()
            self._dirty = False
        return self._matrix

__all__ = ["Camera"]
