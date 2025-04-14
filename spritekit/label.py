# spritekit/label.py
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

from functools import reduce

from .drawable import Drawable
from .cache import load_font
from .renderer import rect_vertices

import moderngl
import glm
from PIL import Image
import freetype as ft

def _is_monospaced_metrics(face: ft.Face):
    return bool(face.face_flags & ft.FT_FACE_FLAG_FIXED_WIDTH)

def _get_char_width(face: ft.Face, char: str):
    face.load_char(char, ft.FT_LOAD_RENDER)
    return face.glyph.advance.x

def _is_monospaced_font(face: ft.Face, test_chars):
    if not _is_monospaced_metrics(face):
        return None
    if not test_chars:
        test_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    widths = set(_get_char_width(face, char) for char in test_chars)
    return widths[0] if len(widths) == 1 else None

class _Glyph:
    def __init__(self, bitmap: ft.Bitmap, glyph: ft.Glyph, clip: tuple[float, float, float, float]):
        self.bitmap = bitmap
        self.glyph = glyph
        self.clip = clip

class Label(Drawable):
    def __init__(self, text: str, font: str | ft.Face, font_size: int = 48, **kwargs):
        self._generator = self._regenerate
        super().__init__(**kwargs)
        self._font = font if isinstance(font, ft.Face) else load_font(font)
        self._width = None
        assert self._test_monospaced(), "Font must be monospaced"
        self._font_size = None
        self._cache = {}
        self._sorted_keys = []
        self._text = None
        self._texture = None
        self._dirty = True
        self._dirty_texture = True
        self.font_size = font_size
        self.text = text
    
    def _test_monospaced(self):
        self._width = _is_monospaced_font(self._font)
        return self._width is not None
    
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, text: str):
        uniq = list(set(list(text)))
        keys = list(self._cache.keys())
        new = [c for c in uniq if c not in keys]
        if new:
            for char in new:
                self._font.load_char(char)
                self._cache[char] = _Glyph(self._font.glyph.bitmap, self._font.glyph, (0., 0., 1., 1.))
            self._sorted_keys = sorted(self._cache.keys())
            self._dirty_texture = True
        if self._text != text:
            self._text = text
            self._dirty = True
        assert len(self._text) > 0, "Label must have text"
    
    @property
    def font_size(self):
        return self._font_size
    
    @font_size.setter
    def font_size(self, font_size: int):
        self._font_size = font_size
        self._font.set_char_size(font_size * 64)
        self._cache = {}
        self._texture = None
        self.text = self._text
        self._dirty = True
        self._dirty_texture = True
    
    def _convert_clip(self, x, y, tx, ty, cx, cy):
        x *= cx
        y *= cy
        return (x / tx,
                1.0 - (y + cy) / ty,
                (x + cx) / tx,
                1.0 - y / ty)

    def _regenerate_texture(self):
        width = self._width * len(self._sorted_keys)
        height = max(self._cache[x].glyph.rows for x in self._sorted_keys)
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for x, char in enumerate(self._sorted_keys):
            image.paste(self._cache[char].bitmap, (x * self._width, 0))
            self._cache[char].clip = self._convert_clip(x * self._width, 0,
                                                        width, height,
                                                        self._width, height)
        self._texture = moderngl.get_context().texture(image.size, 4, image.tobytes())
        self._dirty_texture = False
    
    def _regenerate_vertices(self):
        self._vertices = []
        lines = self._text.split("\n")
        max_height = reduce(lambda x, y: x + y, [max(self._cache[x].height for x in line) for line in lines])
        y = self._position.y - max_height / 2
        for line in lines:
            # TODO: Text alignment
            x = self._position.x - (len(line) * self._width / 2)
            height = 0
            for char in line:
                self._vertices.extend(rect_vertices(x, y, self._cache[char].width, self._cache[char].height, 0., 1., self._cache[char].clip, self._color))
                x += self._cache[char].width
                height = max(height, self._cache[char].height)
            y += height
        
    def _regenerate(self):
        if self._dirty_texture or self._texture is None:
            self._regenerate_texture()
        self._regenerate_vertices()
        self._dirty = False
    
    def draw(self):
        self._draw([])
        super().draw()

__all__ = ["Label"]