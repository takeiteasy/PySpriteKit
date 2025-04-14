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

from .cache import load_texture
from .shapes import RectActor

import glm
import moderngl

class SpriteActor(RectActor):
    def __init__(self,
                 position: glm.vec2 | list | tuple,
                 texture: str | moderngl.Texture,
                 size: Optional[glm.vec2 | list | tuple] = None,
                 clip: Optional[tuple | list] = None,
                 **kwargs):
        tmp = texture if isinstance(texture, moderngl.Texture) else load_texture(texture)
        size = size if size is not None else tmp.size
        super().__init__(position=position,
                         size=size,
                         **kwargs)
        self._texture = tmp
        if clip is not None:
            self._clip = self._convert_clip(clip)
        else:
            self._clip = (0., 0., 1., 1.)
    
    def _convert_clip(self, clip: tuple | list):
        assert len(clip) == 2, "Clip must be a tuple or list of 2 floats or ints"
        tx, ty = self._texture.size
        cx, cy = self._size
        x, y = glm.vec2(*clip) * glm.vec2(cx, cy)
        assert x >= 0 and x <= tx, "Clip x must be between 0 and the texture width"
        assert y >= 0 and y <= ty, "Clip y must be between 0 and the texture height"
        assert x + cx >= 0 and x + cx <= tx, "Clip x + width must be between 0 and the texture width"
        assert y + cy >= 0 and y + cy <= ty, "Clip y + height must be between 0 and the texture height"
        return (x / tx,
                1.0 - (y + cy) / ty,
                (x + cx) / tx,
                1.0 - y / ty)
    
    @property
    def clip(self):
        return self._clip

    @clip.setter
    def clip(self, value: tuple | list):
        self._clip = self._convert_clip(value)
        self._dirty = True

__all__ = ["SpriteActor"]
