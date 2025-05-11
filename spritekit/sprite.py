# spritekit/sprite.py
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

from typing import Optional
from dataclasses import dataclass, field
import json

from .actor import Actor
from .cache import load_texture, find_file, __image_folders__
from .shapes import RectNode
from .texture import Texture
from .timer import TimerNode

from pyglm import glm
import moderngl
from PIL import Image

class SpriteNode(RectNode):
    def __init__(self,
                 texture: Texture | str = None,
                 clip: glm.vec4 | tuple | list = None,
                 **kwargs):
        super().__init__(**kwargs)
        match texture:
            case str():
                self._texture = load_texture(texture)
            case Image.Image():
                self._texture = Texture(texture)
            case moderngl.Texture() | Texture():
                self._texture = texture
            case _:
                raise ValueError("Invalid texture type")
        self._set_clip(clip)

    def _set_clip(self, clip: glm.vec4 | tuple | list):
        if clip is None:
            self._size = glm.vec2(*self._texture.size)
            self._clip = glm.vec2(0, 0)
        else:
            match len(clip):
                case 2:
                    clip = (0, 0, *clip)
                case 4:
                    clip = tuple(clip)
                case _:
                    raise ValueError("Clip must be a 2D vector or a 4D vector")
            self._size = glm.vec2(clip[2], clip[3])
            self._clip = glm.vec2(clip[0], clip[1])
        self._dirty = True

    @property
    def clip(self):
        return *self._clip, *self._size

    @clip.setter
    def clip(self, value: glm.vec4 | tuple | list):
        self._set_clip(value)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value: glm.vec2 | tuple | list):
        self._set_clip(value)

    def _generate_vertices(self):
        p1, p2, p3, p4 = self.points
        tx, ty = self._clip / glm.vec2(*self._texture.size)
        tw, th = self._size / glm.vec2(*self._texture.size)
        return [*p1, tx,      ty,      *self._color,
                *p2, tx + tw, ty,      *self._color,
                *p3, tx,      ty + th, *self._color,
                *p3, tx,      ty + th, *self._color,
                *p4, tx + tw, ty + th, *self._color,
                *p2, tx + tw, ty,      *self._color]

# TODO: Total rework
class AnimatedSpriteNode(SpriteNode):
    pass

__all__ = ["SpriteNode", "AnimatedSpriteNode"]