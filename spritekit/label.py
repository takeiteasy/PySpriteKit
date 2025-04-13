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

from .drawable import Drawable
from .cache import load_font

import freetype as ft

class _Glyph:
    def __init__(self, char: str, font: ft.Face):
        self._font = font
        self._font.load_char(char)
        glyph = self._font.glyph
        bitmap = glyph.bitmap
        self._width   = bitmap.width
        self._height  = bitmap.rows

class Label(Drawable):
    def __init__(self, text: str, font: str | ft.Face, size: int = 48, **kwargs):
        super().__init__(**kwargs)
        self._text = text
        self._font = font if isinstance(font, ft.Face) else load_font(font)
        self._font.set_char_size(size * 64)
        self._size = size
        self._cache = {}
    
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, text: str):
        self._text = text
        self._dirty = True
    
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, size: int):
        self._size = size
        self._font.set_char_size(size * 64)
        self._dirty = True

    def _ensure_char(self, char):
        if not char in self._cache:
            glyph = _Glyph(char, self._font)
            self._cache[char] = glyph
            return glyph
        else:
            return self._cache[char]
    
    def measure(self):
        width = 0
        height = 0
        line_height = 0
        for char in self._text:
            glyph = self._ensure_char(char)
            line_height = max(line_height, glyph.height)
            if char == '\n':
                height += line_height
                width = 0
            else:
                width += glyph.width
        return width, height if height > 0 else line_height
    
    def _regenerate(self):
        self.clear()
        for char in self._text:
            glyph = self._ensure_char(char)