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
from functools import reduce
import platform

from .actor import Actor
from .cache import find_file, load_font
from . import _drawable as drawable
from . import _renderer as renderer

import moderngl
from PIL import ImageFont, ImageDraw, Image
from typing import Optional

def _system_font_paths():
    def _clean(paths):
        return [path for path in paths if os.path.isdir(path)]
    match platform.system():
        case "Windows":
            return _clean([os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")])
        case "Darwin":
            return _clean(["/Library/Fonts",
                           "/System/Library/Fonts",
                           os.path.expanduser("~/Library/Fonts")])
        case "Linux":
            return _clean(["/usr/share/fonts",
                           "/usr/local/share/fonts",
                           os.path.expanduser("~/.fonts"),
                           os.path.expanduser("~/.local/share/fonts")])
        case _:
            return []

__pil_fonts__ = [".pil", ".pbm"]
__other_fonts__ = [".ttf", ".ttc", ".otf", ".pfa", ".pfb", ".cff", ".fnt", ".fon", ".bdf", ".pcf", ".woff", ".woff2", ".dfont"]
__font_extensions__ = __pil_fonts__ + __other_fonts__
__font_folders__ = ("fonts", *_system_font_paths())

def _convert_color(color: tuple | list):
    assert 3 <= len(color) <= 4, "Color must be a list of 3 or 4 values"
    return tuple(min(max(v if isinstance(v, int) else int(v * 255.), 0), 255) for v in (color if len(color) == 4 else (*color, 255)))

class LabelActor(drawable.Drawable):
    def __init__(self,
                 text: str,
                 font: str | ImageFont.FreeTypeFont | ImageFont.ImageFont,
                 font_size: Optional[int] = None,
                 background: Optional[tuple | list] = None,
                 align: str = "left",
                 **kwargs):
        color = kwargs.pop("color", (255, 255, 255, 255))
        self._generator = self._rebuild
        super().__init__(**kwargs)
        self._color = _convert_color(color)
        self._font_size = font_size
        self._font = None
        self._is_truetype = False
        self._load_font(font, font_size)
        self._text = text
        self._background_color = _convert_color(background) if background is not None else (0, 0, 0, 0)
        self._texture = None
        self._align = align
        self._dirty = True
    
    def _load_font(self, font: str | ImageFont.FreeTypeFont | ImageFont.ImageFont, font_size: Optional[int] = None):
        if isinstance(font, str):
            self._font = load_font(font, font_size)
        elif isinstance(font, ImageFont.FreeTypeFont) or isinstance(font, ImageFont.ImageFont):
            self._font = font
        else:
            raise ValueError(f"Unsupported font type: {type(font)}")
        self._is_truetype = isinstance(self._font, ImageFont.FreeTypeFont)

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
        self._load_font(value, self._font_size)
        self._dirty = True
    
    @property
    def font_size(self):
        return self._font_size
    
    @font_size.setter
    def font_size(self, value: int):
        self._font_size = value
        self._load_font(self._font, value)
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
    
    @property
    def align(self):
        return self._align
    
    @align.setter
    def align(self, value: str):
        match value.lower():
            case "left" | "l":
                self._align = "left"
            case "right" | "r":
                self._align = "right"
            case "center" | "middle" | "c" | "m":
                self._align = "center"
            case _:
                raise ValueError(f"Unsupported align value: {value}")
    
    def _calculate_size(self, text: str):
        if self._is_truetype:
            bbox = self._font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = self._font.getsize(text)
        return text_width, text_height
    
    def _rebuild(self):
        lines = self._text.split("\n")
        sizes = [self._calculate_size(line) for line in lines]
        max_width = max(size[0] for size in sizes)
        total_height = reduce(lambda x, y: x + y, [size[1] for size in sizes])
        image = Image.new("RGBA", (max_width, total_height), self._background_color)
        draw = ImageDraw.Draw(image)
        for y, line in enumerate(lines):
            match self._align:
                case "left":
                    x = 0
                case "right":
                    x = max_width - sizes[y][0]
                case "center":
                    x = (max_width - sizes[y][0]) / 2
            draw.text((x, y * sizes[y][1]), line, font=self._font, fill=self._color)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if self._texture is not None:
            del self._texture
        self._texture = moderngl.get_context().texture(image.size, 4, image.tobytes())
        return renderer.rect_vertices(*self._position, *self._texture.size, self._rotation, self._scale, (0., 0., 1., 1.), self._color)
    
    def draw(self):
        self._draw([])
        Actor.draw(self)

__all__ = ["LabelActor"]
