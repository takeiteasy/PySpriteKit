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

import moderngl
from PIL import Image
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
    def __init__(self, text: str, font: str | ft.Face, font_size: int = 48, **kwargs):
        super().__init__(**kwargs)
        self._text = text
        self._font = font if isinstance(font, ft.Face) else load_font(font)
        self._font.set_char_size(font_size * 64)
        self._font_size = font_size
    
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, text: str):
        self._text = text
        self._dirty = True
    
    @property
    def font_size(self):
        return self._font_size
    
    @font_size.setter
    def font_size(self, font_size: int):
        self._font_size = font_size
        self._font.set_char_size(font_size * 64)
        self._dirty = True

    def _regenerate(self):
        glyphs = { c: _Glyph(c, self._font) for c in list(set(list(self._text))) }
        max_width = max(glyph.width for glyph in glyphs.values())
        max_height = max(glyph.height for glyph in glyphs.values())
        lines = self._text.split("\n")
        longest_line = max(lines, key=len)
        image = Image.new("RGBA", (max_width * longest_line, max_height * len(lines)), (0, 0, 0, 0))
        keys = { k: v for k, v in enumerate(sorted([int(c) for c in glyphs.keys()])) }
        for k, v in keys.items():
            glyph = glyphs[v]
            image.paste(glyph.bitmap, (k * max_width, 0))
        ctx = moderngl.get_context()
        self._texture = ctx.texture(image.size, 4, image.tobytes())
        # TODO: Check texture atlas and stuff
        # TODO: Generate vertices for each glyph
    
    def draw(self):
        self._draw([])
        super().draw()