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

from . import _drawable as drawable
from .shapes import _rect_points
from .cache import load_texture, find_file, __image_folders__
from .timer import TimerNode

from pyglm import glm
import moderngl
from PIL import Image

class SpriteNode(drawable.Drawable):
    def __init__(self,
                 texture: Image.Image | str | moderngl.Texture,
                 clip: glm.vec4 | tuple | list = None,
                 **kwargs):
        super().__init__(**kwargs)
        match texture:
            case str():
                self._texture = load_texture(texture)
            case Image.Image():
                self._texture = moderngl.get_context().texture(texture.size, 4, texture.tobytes())
            case moderngl.Texture():
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

    @property
    def points(self):
        return _rect_points(*self._position, *self._size, self._rotation, self._scale)

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

@dataclass
class _AtlasLayer:
    name: str
    opacity: int
    blendMode: str

@dataclass
class _AtlasFrameTag:
    name: str
    start: int
    to: int
    direction: str
    color: str

@dataclass
class _AtlasFrame:
    frame: tuple[int, int, int, int]
    rotated: bool
    trimmed: bool
    spriteSourceSize: tuple[int, int, int, int]
    sourceSize: tuple[int, int]
    duration: int

class AtlasNode(SpriteNode):
    def __init__(self,
                 atlas: str | dict,
                 initial_frame: Optional[str | int] = None,
                 initial_animation: Optional[str] = None,
                 **kwargs):
        if isinstance(atlas, str):
            path = find_file(atlas, __image_folders__, [".json"])
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = atlas
        assert data is not None, "Invalid atlas data"
        self._texture = None
        self._layers = {}
        self._tags = {}
        self._frames = {}
        self._animations = {}
        self._current_animation = None
        self._animation_timer = None
        self._animation_cursor = None
        self._load_aseprite(data)
        super().__init__(texture=self._texture, **kwargs)
        if initial_animation is not None:
            self._set_animation(initial_animation)
        elif initial_frame is not None:
            self._set_frame(str(initial_frame))

    def _load_aseprite(self, data: dict):
        assert "frames" in data and "meta" in data, "Invalid JSON data"
        self._texture = load_texture(data["meta"]["image"])
        for layer in data["meta"]["layers"]:
            l = _AtlasLayer(**layer)
            self._layers[l.name] = l
        for tag in data["meta"]["frameTags"]:
            tag["start"] = tag.pop("from")
            self._tags[tag["name"]] = _AtlasFrameTag(**tag)
        for frame_name, frame in data["frames"].items():
            name = frame_name.split(" ")[-1][:-1]
            def _fix_dict(_data: dict):
                match len(_data):
                    case 2:
                        return _data["w"], _data["h"]
                    case 4:
                        return _data["x"], _data["y"], _data["w"], _data["h"]
                    case _:
                        raise ValueError("Invalid dict length")
            frame["duration"] = int(frame["duration"]) / 1000.
            self._frames[name] = _AtlasFrame(**{k: _fix_dict(v) if isinstance(v, dict) else v for k, v in frame.items()})
        assert len(self._frames) > 0, "No frames found in atlas"
        for _, tag in self._tags.items():
            self._animations[tag.name] = [self._frames[str(i)] for i in range(tag.start, tag.to + 1)]

    def _set_animation(self, name: str):
        assert name in self._animations, f"Invalid animation name: {name}"
        self._current_animation = name
        self._animation_cursor = 0
        self._animation_timer = TimerNode(duration=self._animations[name][0].duration,
                                          repeat=True,
                                          auto_start=True,
                                          on_complete=self.next_frame)
        self._set_frame(self._animations[name][0])

    @property
    def animation(self):
        return self._current_animation

    @animation.setter
    def animation(self, value: str):
        self._set_animation(value)

    @property
    def frame(self):
        if not self._current_animation:
            return None
        else:
            return self._animations[self._current_animation][self._animation_cursor]

    @frame.setter
    def frame(self, value: str | int):
        self._set_frame(str(value) if isinstance(value, int) else value)

    def next_frame(self):
        if not self._current_animation:
            return
        self._animation_cursor += 1
        if self._animation_cursor >= len(self._animations[self._current_animation]):
            self._animation_cursor = 0
        self._set_frame(self._animations[self._current_animation][self._animation_cursor])

    def _set_frame(self, frame: str | _AtlasFrame):
        if isinstance(frame, str):
            assert frame in self._frames, f"Invalid frame name: {frame}"
            frame = self._frames[frame]
        self._animation_timer.duration = frame.duration
        self._set_clip(glm.vec4(*frame.frame))

    def step(self, delta: float):
        if self._animation_timer is not None:
            self._animation_timer.step(delta)
        super().step(delta)

    def start(self, animation: Optional[str] = None):
        if animation is not None:
            self._set_animation(animation)
        elif self._current_animation is not None:
            self._set_animation(self._current_animation)
        else:
            raise ValueError("No animation to start")

    def stop(self):
        self._animation_timer = None
        self._animation_cursor = 0
        if self._current_animation:
            self._set_frame(self._animations[self._current_animation][0])
        else:
            self._set_frame(next(iter(self._frames.keys())))

    def pause(self):
        if self._animation_timer:
            self._animation_timer.pause()

    def resume(self):
        if self._animation_timer:
            self._animation_timer.resume()

__all__ = ["SpriteNode", "AtlasNode"]