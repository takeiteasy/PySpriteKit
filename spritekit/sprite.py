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

from dataclasses import dataclass, field
from typing import Optional

from .renderer import load_texture
from .shapes import Rect

import glm
import moderngl

@dataclass
class SpriteAnimation:
    frames: list[tuple[int, int]]
    fps: int

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int):
        return self.frames[index]

@dataclass
class SpriteSheet:
    cell_size: tuple[int, int]
    animations: dict[str, SpriteAnimation] = field(default_factory=dict)

    def __getitem__(self, name: str):
        assert name in self.animations, "Animation must be in animations"
        return self.animations[name]

class Sprite(Rect):
    def __init__(self,
                 position: glm.vec2 | list | tuple,
                 texture: str | moderngl.Texture,
                 clip: Optional[tuple | list] = None,
                 sprite_sheet: Optional[SpriteSheet] = None,
                 initial_animation: Optional[str] = None,
                 loop: bool = True,
                 auto_start: bool = True,
                 **kwargs):
        tmp = texture if isinstance(texture, moderngl.Texture) else load_texture(texture)
        self._sprite_sheet = sprite_sheet
        size = kwargs.pop("size", None)
        if size is None:
            size = tmp.size if self._sprite_sheet is None else self._sprite_sheet.cell_size
        super().__init__(position=position,
                         size=size,
                         on_added=self.play if auto_start else None,
                         **kwargs)
        self._texture = tmp
        if clip is not None:
            self._clip = self._convert_clip(clip)
        else:
            self._clip = (0., 0., 1., 1.)
        self._current_frame = 0
        self._current_time = 0.
        self._loop = loop
        self._playing = False
        if self._sprite_sheet is not None:
            animation = initial_animation
            if animation is None:
                if len(self._sprite_sheet.animations) > 0:
                    animation = next(iter(self._sprite_sheet.animations))
            self._current_animation = animation
            self.clip = self.current_animation[self._current_frame]
        else:
            self._current_animation = None
    
    def _convert_clip(self, clip: tuple | list):
        assert len(clip) == 2, "Clip must be a tuple or list of 2 floats or ints"
        tx, ty = self._texture.size
        cx, cy = self._size
        x, y = glm.vec2(*clip) * glm.vec2(cx, cy)
        return (x / tx,
                1.0 - (y + cy) / ty,
                (x + cx) / tx,
                1.0 - y / ty)
    
    def play(self):
        if not self._playing and self._current_animation is not None and self._sprite_sheet is not None:
            self._current_time = 0.
            self._current_frame = 0
            self.clip = self._sprite_sheet[self._current_animation][self._current_frame]
    
    def pause(self):
        if self._playing:
            self._playing = False
    
    def resume(self):
        if not self._playing:
            self._playing = True
    
    def stop(self):
        self._playing = False
        self._current_time = 0.
        self._current_frame = 0
    
    @property
    def clip(self):
        return self._clip

    @clip.setter
    def clip(self, value: tuple | list):
        self._clip = self._convert_clip(value)
        self._dirty = True

    @property
    def animation(self):
        return self._current_animation
    
    @animation.setter
    def animation(self, value: str):
        assert value in self._sprite_sheet.animations, "Animation must be in sprite sheet"
        self._current_animation = value
    
    @property
    def current_animation(self):
        return self._sprite_sheet[self._current_animation]
    
    @property
    def frame(self):
        return self._current_frame
    
    @frame.setter
    def frame(self, value: int):
        assert value < len(self._sprite_sheet[self._current_animation]), "Frame must be in range of animation"
        self._current_frame = value
    
    @property
    def loop(self):
        return self._loop
    
    @loop.setter
    def loop(self, value: bool):
        self._loop = value
    
    @property
    def playing(self):
        return self._playing
    
    @playing.setter
    def playing(self, value: bool):
        if value:
            self.play()
        else:
            self.stop()
    
    def step(self, dt: float):
        super().step(dt)
        if self._playing:
            self._current_time += dt
            if self._current_time >= 1. / self.current_animation.fps:
                self._current_frame += 1
                if self._current_frame >= len(self.current_animation):
                    if self._loop:
                        self._current_frame = 0
                        self._current_time = 0.
                    else:
                        self.clip = self.current_animation[-1]
                        self._playing = False
                self.clip = self.current_animation[self._current_frame]

    def draw(self):
        self._draw([*self._position, *self._size, self._rotation, self._scale, self._clip, self._color])
        super().draw()