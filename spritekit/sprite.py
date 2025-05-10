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
from dataclasses import dataclass
import json

from .cache import load_texture, find_file, __image_folders__
from .shapes import RectActor
from .texture import Texture

from pyglm import glm
import moderngl
from PIL import Image

class SpriteActor(RectActor):
    def __init__(self,
                 texture: str | Image.Image | moderngl.Texture | Texture,
                 position: glm.vec2 | list | tuple = (0., 0.),
                 size: Optional[glm.vec2 | list | tuple] = None,
                 clip: Optional[tuple | list] = None,
                 **kwargs):
        # noinspection PyUnreachableCode
        match texture:
            case str():
                tmp = load_texture(texture)
            case Image.Image():
                tmp = Texture(texture)
            case moderngl.Texture() | Texture():
                tmp = texture
            case _:
                raise ValueError("Invalid texture type")
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
        assert 0 <= x <= tx, "Clip x must be between 0 and the texture width"
        assert 0 <= y <= ty, "Clip y must be between 0 and the texture height"
        assert 0 <= x + cx <= tx, "Clip x + width must be between 0 and the texture width"
        assert 0 <= y + cy <= ty, "Clip y + height must be between 0 and the texture height"
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

def _pop_name(data: dict):
    name = data.pop("name")
    return name, data

def _fix_dict(data: dict):
    match len(data):
        case 2:
            return data["w"], data["h"]
        case 4:
            return data["x"], data["y"], data["w"], data["h"]
        case _:
            raise ValueError("Invalid dict length")

@dataclass
class SpriteSheetFrame:
    frame: tuple[int, int, int, int]
    rotated: bool
    trimmed: bool
    spriteSourceSize: tuple[int, int, int, int]
    sourceSize: tuple[int, int]
    duration: int

class SpriteSheetActor(SpriteActor):
    def __init__(self,
                 json_path: str,
                 initial_frame: Optional[int] = None,
                 initial_animation: Optional[str] = None,
                 auto_start: bool = True,
                 loop: bool = True,
                 **kwargs):
        path = find_file(json_path, __image_folders__, "json")
        with open(path, "r") as f:
            data = json.load(f)
        assert "frames" in data and "meta" in data, "Invalid JSON data"
        _texture = load_texture(data["meta"]["image"])
        self._layers = dict(_pop_name(layer) for layer in data["meta"]["layers"])
        self._tags = dict(_pop_name(frame) for frame in data["meta"]["frameTags"])
        self._frames = {}
        for name, frame in data["frames"].items():
            name = name.split(" ")[-1][:-1]
            frame = {k: _fix_dict(v) if isinstance(v, dict) else v for k, v in frame.items()}
            self._frames[name] = SpriteSheetFrame(**frame)
        assert len(self._frames) > 0, "No frames found"
        self._frame = next(iter(self._frames))
        super().__init__(texture=_texture, **kwargs)
        self._set_frame(initial_frame if initial_frame is not None else self._frame)
    
    def _convert_frame(self, frame: tuple[int, int, int, int]):
        return (frame[0] / self._texture.size[0],
                1.0 - (frame[1] + frame[3]) / self._texture.size[1],
                (frame[0] + frame[2]) / self._texture.size[0],
                1.0 - frame[1] / self._texture.size[1])
    
    def _set_frame(self, frame: str | int):
        name = str(frame) if isinstance(frame, int) else frame
        assert name in self._frames, "Invalid frame"
        self._frame = name
        self._size = self._frames[name].sourceSize
        self._clip = self._convert_frame(self._frames[name].frame)
        self._dirty = True
    
    @property
    def frame(self):
        return self._frame
    
    @frame.setter
    def frame(self, value: str | int):
        self._set_frame(value)
    
__all__ = ["SpriteActor", "SpriteSheetActor"]
