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

import os

from .actor import Actor
from .cache import _find_file
from .drawable import Drawable
from .renderer import rect_vertices

import moderngl
from PIL import ImageFont, ImageDraw, Image
from typing import Optional

__pil_fonts__ = [".pil", ".pbm"]
__other_fonts__ = [".ttf", ".ttc", ".otf", ".pfa", ".pfb", ".cff", ".fnt", ".fon", ".bdf", ".pcf", ".woff", ".woff2", ".dfont"]
__font_extensions__ = __pil_fonts__ + __other_fonts__
__font_folders__ = ("fonts",)

def _convert_color(color: tuple | list):
    assert 3 <= len(color) <= 4, "Color must be a list of 3 or 4 values"
    return tuple(min(max(v if isinstance(v, int) else int(v * 255.), 0), 255) for v in (color if len(color) == 4 else (*color, 255)))

# TODO: Rewrite to replace ImageFont
# TODO: Text-alignment

class Label(Drawable):
    def __init__(self,
                 text: str,
                 font: str | ImageFont.FreeTypeFont | ImageFont.ImageFont,
                 font_size: int = 48,
                 background: Optional[tuple | list] = None,
                 **kwargs):
        color = kwargs.pop("color", (255, 255, 255, 255))
        self._generator = self._rebuild
        super().__init__(**kwargs)
        self._color = _convert_color(color)
        self._font_size = font_size
        self._text = text
        self.font = font
        self._background_color = _convert_color(background) if background is not None else (0, 0, 0, 0)
        self._texture = None
        self._dirty = True
    
    @property
    def text(self):
        return self._text
    
    @text.setter
    def text(self, value: str):
        self._text = value
        self._dirty = True
    
    @property
    def font(self):
        return self._font
    
    @font.setter
    def font(self, value: str | ImageFont.FreeTypeFont | ImageFont.ImageFont):
        match value:
            case str():
                found = _find_file(value, __font_folders__, __font_extensions__)
                _, ext = os.path.splitext(found)
                if ext in __pil_fonts__:
                    self._font = ImageFont.load(found)
                elif ext in __other_fonts__:
                    self._font = ImageFont.truetype(found, self._font_size)
                else:
                    raise ValueError(f"Unsupported font extension: {ext}, supported extensions: {', '.join(__font_extensions__)}")
            case ImageFont.FreeTypeFont() | ImageFont.ImageFont():
                self._font = value
            case _:
                raise ValueError(f"Unsupported font type")
        self._is_truetype = isinstance(self._font, ImageFont.FreeTypeFont)
        self._dirty = True
    
    @property
    def font_size(self):
        return self._font_size
    
    @font_size.setter
    def font_size(self, value: int):
        self._font_size = value
        self._dirty = True

    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value: tuple | list):
        self._color = _convert_color(value)
        self._dirty = True
    
    @property
    def background_color(self):
        return self._background_color
    
    @background_color.setter
    def background_color(self, value: tuple | list):
        self._background_color = _convert_color(value)
        self._dirty = True
    
    def _rebuild(self):
        if self._is_truetype:
            bbox = self._font.getbbox(self._text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = self._font.getsize(self._text)
        image = Image.new("RGBA", (text_width + 20, text_height + 20), self._background_color)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), self._text, font=self._font, fill=self._color)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if self._texture is not None:
            del self._texture
        self._texture = moderngl.get_context().texture(image.size, 4, image.tobytes())
        return rect_vertices(*self._position, *self._texture.size, self._rotation, self._scale, (0., 0., 1., 1.), self._color)
    
    def draw(self):
        self._draw([])
        Actor.draw(self)