# spritekit/texture.py
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

from .cache import load_image

import moderngl
from PIL import Image

def _convert_color(color: tuple | list):
    assert 3 <= len(color) <= 4, "Color must be a list of 3 or 4 values"
    return tuple(min(max(v if isinstance(v, int) else int(v * 255.), 0), 255) for v in (color if len(color) == 4 else (*color, 255)))

class Texture:
    def __init__(self, image: str | Image.Image):
        self._image = load_image(image) if isinstance(image, str) else image
        self._texture = None
        self._dirty = True
        self._load()
    
    def _load(self):
        self._texture = moderngl.get_context().texture(self._image.size, 4, self._image.tobytes())
        self._dirty = False
    
    @classmethod
    def checkered(cls,
                  width: int,
                  height: int,
                  square_width: int,
                  square_height: Optional[int] = None,
                  color_1: tuple | list = (1., 1., 1., 1.),
                  color_2: tuple | list = (0., 0., 0., 1.),
                  offset_x: int = 0,
                  offset_y: int = 0):
        color_1 = _convert_color(color_1)
        color_2 = _convert_color(color_2)
        if square_height is None:
            square_height = square_width
        img = Image.new("RGBA", (width, height), color_1)
        offset_x = offset_x % square_width
        offset_y = offset_y % square_height
        rows = (height + square_height + offset_y) // square_height
        cols = (width + square_width + offset_x) // square_width
        for row in range(rows):
            for col in range(cols):
                if (row + col) % 2 == 0:
                    img.paste(color_2, (col * square_width - offset_x, row * square_height - offset_y, (col + 1) * square_width - offset_x, (row + 1) * square_height - offset_y))
        return Texture(img)
    
    @classmethod
    def solid(cls,
              width: int,
              height: int,
              color: tuple | list = (1., 1., 1., 1.)):
        return Texture(Image.new("RGBA", (width, height), _convert_color(color)))
    
    @property
    def size(self):
        return self._image.size
    
    @property
    def image(self):
        return self._image

    @property
    def raw(self):
        return self._texture
    
    def use(self):
        if self._dirty:
            self._load()
        self._texture.use()
