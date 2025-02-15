# spritekit/actor.py
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

from .vector import Vector2
from dataclasses import dataclass, field
from typing import Optional, override
from raylib.colors import *
import raylib as rl
import pyray as r

class ActorType:
    pass

class ActorParent:
    def _add_child(self, node: ActorType):
        if not hasattr(self, '_children'):
            self._children = []
        self._children.append(node)

    def add_child(self, node: ActorType):
        self._add_child(node)

    def add_children(self, nodes: ActorType | list[ActorType]):
        for node in nodes if isinstance(nodes, list) else [nodes]:
            self.add_child(node)

    def find_children(self, name: Optional[str] = ""):
        if not hasattr(self, '_children'):
            return []
        return [x for x in self._children if x.name == name]
    
    def all_children(self):
        return self._children if hasattr(self, '_children') else []
    
    def children(self, name: Optional[str] = ""):
        for child in self.find_children(name):
            yield child

    def remove_children(self, name: Optional[str] = ""):
        self._children = [x for x in (self._children if hasattr(self, '_children') else []) if x.name != name]

    def remove_all_children(self):
        self._children = []

@dataclass
class Actor(ActorType, ActorParent):
    name: str = ""

    def __str__(self):
        return f"(Node({self.__class__.__name__}) {" ".join([f"{key}:{getattr(self, key)}" for key in list(vars(self).keys())])})"
    
    @override
    def add_child(self, node: ActorType):
        node.parent = self
        self._add_child(node)

    def step(self, delta: float):
        for child in self.all_children():
            child.step(delta)

    def draw(self):
        for child in self.all_children():
            child.draw()

@dataclass
class Actor2D(Actor):
    position: Vector2 = field(default_factory=Vector2)
    rotation: float = 0.
    scale: float = 1.
    origin: Vector2 = field(default_factory=lambda: Vector2([0.5, 0.5]))
    color: r.Color = RAYWHITE

    def _offset(self):
        return self.position + self.origin * Vector2([-self.width, -self.height])

class BaseShape:
    draw_func = None
    draw_wire_func = None

@dataclass
class ShapeActor2D(Actor2D):
    wireframe: bool = False
    line_thickness: float = 1.

    def _draw(self, *args, **kwargs):
        if self.wireframe:
            self.__class__.draw_wire_func(*args, **kwargs)
        else:
            self.__class__.draw_func(*args, **kwargs)

@dataclass
class LineActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawLine
    draw_wire_func = rl.DrawLine
    end: Vector2 = field(default_factory=Vector2)

    @override
    def draw(self):
        self._draw(self.position.x, self.position.y, self.end.x, self.end.y, self.color)

@dataclass
class RectangleActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawRectangleRec
    draw_wire_func = rl.DrawRectangleLinesEx
    width: float = 1.
    height: float = 1.

    @override
    def draw(self):
        rec = r.Rectangle(*self._offset(), self.width, self.height)
        if self.wireframe:
            self._draw(rec, self.line_thickness, self.color)
        else:
            self._draw(rec, self.color)

@dataclass
class CircleActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawCircle
    draw_wire_func = rl.DrawCircleLines
    radius: float = 1.

    @override
    def draw(self):
        self._draw(int(self.position.x), int(self.position.y), self.radius, self.color)

@dataclass
class TriangleActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawTriangle
    draw_wire_func = rl.DrawTriangleLines
    position2: Vector2 = field(default_factory=Vector2)
    position3: Vector2 = field(default_factory=Vector2)

    @override
    def draw(self):
        tri = self.position, self.position2, self.position3
        stri = sorted(tri, key=lambda x: x.x)
        def cross(o, a, b):
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        if cross(stri[0], stri[1], stri[2]) > 0:
            stri[1], stri[2] = stri[2], stri[1]
        self._draw([*stri[0]], [*stri[1]], [*stri[2]], self.color)

@dataclass
class EllipseActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawEllipse
    draw_wire_func = rl.DrawEllipseLines
    width: float = 1.
    height: float = 1.

    @override
    def draw(self):
        self._draw(self.position.x, self.position.y, self.width, self.height, self.color)

@dataclass
class SpriteActor2D(Actor2D):
    texture: r.Texture2D = None
    source: r.Rectangle = r.Rectangle(0, 0, 0, 0)
    dst: r.Rectangle = r.Rectangle(0, 0, 0, 0)
    scale: Vector2 = field(default_factory=lambda: Vector2([1., 1.]))

    @property
    def width(self):
        return self.texture.width

    @property
    def height(self):
        return self.texture.height

    @override
    def draw(self):
        if not self.texture:
            return
        if self.source.width == 0 or self.source.height == 0:
            self.source = r.Rectangle(0, 0, self.texture.width, self.texture.height)
        if self.dst.width == 0 or self.dst.height == 0:
            self.dst = r.Rectangle(self.position.x, self.position.y, self.width, self.height)
        r.draw_texture_pro(self.texture,
                           [self.source.x, self.source.y, self.source.width, self.source.height],
                           [self.dst.x, self.dst.y, self.dst.width * self.scale.x, self.dst.height * self.scale.y],
                           [*(-self._offset() * self.scale)], self.rotation, self.color)

@dataclass
class LabelActor2D(Actor2D):
    text: str = ""
    font: r.Font = None
    font_size: float = 16.
    spacing: float = 2.
    color: r.Color = RAYWHITE

    def _size(self):
        size = r.measure_text_ex(self.font, self.text.encode('utf-8'), self.font_size, self.spacing)
        if not hasattr(self, '_width'):
            self._width = size.x
        if not hasattr(self, '_height'):
            self._height = size.y
        return size
    
    @property
    def width(self):
        if not hasattr(self, '_width'):
            return self._size().x
        else:
            return self._width

    @property
    def height(self):
        if not hasattr(self, '_height'):
            return self._size().y
        else:
            return self._height

    def draw(self):
        if not self.font:
            self.font = r.get_font_default()
        r.draw_text_pro(self.font, self.text.encode('utf-8'), [0,0], [*-self._offset()], self.rotation, self.font_size, self.spacing, self.color)

class Line2DNode(LineActor2D):
    pass

class RectangleNode(RectangleActor2D):
    pass

class CircleNode(CircleActor2D):
    pass

class TriangleNode(TriangleActor2D):
    pass

class EllipseNode(EllipseActor2D):
    pass

class SpriteNode(SpriteActor2D):
    pass

class LabelNode(LabelActor2D):
    pass

class AudioActor(Actor):
    volume: float = 1.
    pitch: float = 1.
    pan: float = .5
    play_func = None
    stop_func = None
    pause_func = None
    resume_func = None
    set_volume_func = None
    set_pitch_func = None
    set_pan_func = None
    is_playing_func = None

    @property
    def audio(self):
        return None

    def play(self):
        if self.audio:
            self.__class__.play_func(self.audio)

    def stop(self):
        if self.audio:
            self.__class__.stop_func(self.audio)

    def pause(self):
        if self.audio:
            self.__class__.pause_func(self.audio)

    def resume(self):
        if self.audio:
            self.__class__.resume_func(self.audio)

    def set_volume(self, volume: float):
        if self.audio:
            self.__class__.set_volume_func(self.audio, max(0., min(volume, 1.)))

    def set_pitch(self, pitch: float):
        if self.audio:
            self.__class__.set_pitch_func(self.audio, max(0., min(pitch, 1.)))

    def set_pan(self, pan: float):
        if self.audio:
            self.__class__.set_pan_func(self.audio, max(0., min(pan, 1.)))
    
    @property
    def playing(self):
        if not self.audio:
            return False
        return self.__class__.is_playing_func(self.audio)
    
    @playing.setter
    def playing(self, value: bool):
        if self.audio:
            if value:
                self.play()
            else:
                self.pause()

@dataclass
class SoundActor(AudioActor):
    sound: r.Sound = None
    play_func = r.play_sound
    stop_func = r.stop_sound
    pause_func = r.pause_sound
    resume_func = r.resume_sound
    set_volume_func = r.set_sound_volume
    set_pitch_func = r.set_sound_pitch
    set_pan_func = r.set_sound_pan
    is_playing_func = r.is_sound_playing

    @property
    def audio(self):
        return self.sound

@dataclass
class MusicActor(AudioActor):
    music: r.Music = None
    loop: bool = False
    autostart: bool = False
    play_func = r.play_music_stream
    stop_func = r.stop_music_stream
    pause_func = r.pause_music_stream
    resume_func = r.resume_music_stream
    set_volume_func = r.set_music_volume
    set_pitch_func = r.set_music_pitch
    set_pan_func = r.set_music_pan
    is_playing_func = r.is_music_stream_playing

    @property
    def audio(self):
        return self.music

    def __init__(self, **kwargs):
        hax = ["music", "loop", "autostart"]
        a = {a: kwargs[a] for a in kwargs if not a in hax}
        for k in hax:
            if k in kwargs:
                self.__dict__[k] = kwargs[k]
        super().__init__(**a)
        if self.autostart:
            self.play()

    def seek(self, position: float):
        r.seek_music_stream(self.audio, position)
    
    @property
    def length(self):
        return r.get_music_time_length(self.audio)
    
    @property
    def position(self):
        return r.get_music_time_played(self.audio)
    
    @position.setter
    def position(self, value: float):
        r.seek_music_stream(self.audio, value)

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()
    
    def step(self, _: float):
        if not self.playing:
            return
        r.update_music_stream(self.audio)
        if not self.playing:
            self.position = 0
            self.play()

class MusicNode(MusicActor):
    pass

class SoundNode(SoundActor):
    pass