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
from typing import Optional, override, Callable
from raylib.colors import *
import raylib as rl
import pyray as r
from .easing import ease_linear_in_out
from contextlib import contextmanager
from queue import Queue

class ActorType:
    pass

class ActorParent:
    def _add_child(self, node: ActorType):
        if not hasattr(self, '_children'):
            self._children = []
        self._children.insert(0, node)

    def add_child(self, node: ActorType):
        self._add_child(node)

    def add_children(self, nodes: ActorType | list[ActorType]):
        for node in nodes if isinstance(nodes, list) else [nodes]:
            self.add_child(node)

    def find_children(self, name: Optional[str] = ""):
        if not hasattr(self, '_children'):
            return []
        return [x for x in self._children if x.name == name]
    
    def find_child(self, name: Optional[str] = ""):
        for child in self.find_children(name):
            return child
        return None
    
    def all_children(self):
        return self._children if hasattr(self, '_children') else []
    
    def children(self, name: Optional[str] = ""):
        for child in self.find_children(name):
            yield child
    
    def remove_child(self, child: str | ActorType = None):
        if isinstance(child, str):
            self.remove_children(name=child)
        else:
            for i in range(len(self._children)):
                if self._children[i] == child:
                    self._children.pop(i)
                    return

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
    
    def remove_me(self):
        if hasattr(self, "scene") and self.scene is not None:
            self.scene.remove_child(self)

    def step(self, delta: float):
        for child in self.all_children():
            child.step(delta)

    def draw(self):
        for child in self.all_children():
            child.draw()

@dataclass
class BaseTimer(Actor):
    interval: float = 1.
    repeat: bool = False
    on_complete: Callable[[], None] = None
    on_tick: Callable[[float], None] = None
    auto_start: bool = True
    remove_on_complete: Optional[bool] = None 
    cursor: Optional[float] = None

class TimerNode(BaseTimer):
    def __init__(self, **kwargs):
        BaseTimer.__init__(self, **kwargs)
        self._completed = False
        self._running = self.auto_start
        if not self.cursor:
            self.cursor = self.interval if self._running else 0
        if not self.remove_on_complete:
            self.remove_on_complete = not self.repeat

    def step(self, delta: float):
        if not self._running:
            return
        if self.cursor > 0:
            self.cursor -= delta
            if self.cursor <= 0:
                if self.repeat:
                    self.cursor = self.interval
                else:
                    self.cursor = 0
                    self._running = False
                    self._completed = True
                if self.on_complete:
                    self.on_complete()
                if self.remove_on_complete:
                    self.remove_me()
            else:
                if self.on_tick:
                    self.on_tick(self.cursor)
    
    def reset(self):
        self._completed = False
        self.cursor = self.interval
    
    def start(self):
        if not self._running:
            self._running = True
            self._completed = False
            self.cursor = self.interval
    
    def stop(self):
        self._running = False
        self._completed = True
        self.cursor = 0
    
    def pause(self):
        if not self._completed:
            self._running = False

    def resume(self):
        if not self._completed:
            self._running = True

class ActionType:
    pass

@dataclass
class ActionNode(ActionType, TimerNode):
    easing_fn: Callable[[float, float, float, float], float] = staticmethod(ease_linear_in_out)
    actor: Actor = None
    field: str = None
    target: any = None

    @contextmanager
    def _initial_value(self):
        obj = self.actor
        for i in range(len(self.field)):
            if i == len(self.field) - 1:
                f = getattr(obj, self.field[i])
                if f is None:
                    raise RuntimeError(f"Object has no field {self.field[i]}")
                if not isinstance(f, type(self.target)):
                    raise RuntimeError(f"Field {self.field[i]} is not of type {type(self.target)}")
                yield f
            else:
                if not hasattr(obj, self.field[i]):
                    raise RuntimeError(f"Object has no field {self.field[i]}")
                obj = getattr(obj, self.field[i])

    def __init__(self, **kwargs):
        new_kwargs = {
            "interval": kwargs.pop("interval", 1.),
            "cursor": kwargs.pop("cursor", None),
        }
        TimerNode.__init__(self, **new_kwargs)
        self.actor = kwargs.pop("actor", None)
        self.field = kwargs.pop("field", None)
        self.target = kwargs.pop("target", None)
        if self.actor is None:
            raise RuntimeError("Actor is not set")
        if self.field is None:
            raise RuntimeError("Field is not set")
        if self.target is None:
            raise RuntimeError("Target is not set")
        self.field = self.field.split(".") if "." in self.field else [self.field]
        self._initial_value()
        if "easing_fn" in kwargs:
            self.easing_fn = staticmethod(kwargs.pop("easing_fn"))
        self._start = None
        self.on_complete = self.remove_me
        self.on_tick = self._step

    @property
    def completed(self):
        return self._completed
    
    @property
    def running(self):
        return self._running
    
    def _step(self, delta: float):
        if not self._start:
            with self._initial_value() as start:
                self._start = start
        elapsed = self.interval - self.cursor
        delta = self.target - self._start
        obj = self.actor
        for i in range(len(self.field)):
            if i == len(self.field) - 1:
                def fn(x, y, z, w):
                    return self.easing_fn(x, y, z, w)
                f = getattr(obj, self.field[i])
                if isinstance(f, float):
                    setattr(obj, self.field[i], fn(elapsed, self._start, delta, self.interval))
                else:
                    z = list(zip(self._start, delta))
                    for start, delta in z:
                        print(start, delta)
                    v = [fn(elapsed, start, delta, self.interval) for start, delta in z]
                    setattr(obj, self.field[i], v)
            else:
                obj = getattr(obj, self.field[i])

class WaitAction(ActionType, TimerNode):
    def __init__(self, duration: float, auto_start: bool = True):
        TimerNode.__init__(self, interval=duration, auto_start=auto_start)
        self.on_complete = staticmethod(self._on_complete)
        self._completed = False
    
    @property
    def completed(self):
        return self._completed
    
    def _on_complete(self):
        self._completed = True
        self.remove_me()
    
class ActionSequence(ActionType, TimerNode, Queue):
    def __init__(self, actions: list[ActionType], **kwargs):
        Queue.__init__(self)
        for action in actions:
            self.put(action)
        auto_start = kwargs.pop("auto_start", True)
        timer_kwargs = {
            "interval": 0.,
            "remove_on_complete": kwargs.pop("remove_on_complete", True),
            "auto_start": False,
        }
        TimerNode.__init__(self, **timer_kwargs)
        self._head = None
        if auto_start:
            self.start()
    
    def _complete(self):
        if self.on_complete:
            self.on_complete()
        if self.remove_on_complete:
            self.remove_me()
        self._completed = True
        self._running = False

    @override
    def step(self, delta: float):
        if not self._running or (self.empty() and not self._head):
            return
        if self._head is not None:
            self._head.step(delta)
            if self._head.completed:
                if self.empty():
                    self._complete()
                else:
                    self._head = None 
        else:
            self._head = self.get()
            self._head.start()
    
    @override
    def reset(self):
        raise NotImplementedError("ActionSequence cannot be reset")
    
    @override
    def start(self):
        if self._running:
            raise RuntimeError("ActionSequence is already running")
        if self.empty():
            raise RuntimeError("ActionSequence is empty")
        self._running = True
        self._completed = False
        self._head = None
    
    @override
    def stop(self):
        self._complete()

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
class Line2DNode(ShapeActor2D, BaseShape):
    draw_func = rl.DrawLine
    draw_wire_func = rl.DrawLine
    end: Vector2 = field(default_factory=Vector2)

    @override
    def draw(self):
        self._draw(self.position.x, self.position.y, self.end.x, self.end.y, self.color)

@dataclass
class RectangleNode(ShapeActor2D, BaseShape):
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
class CircleNode(ShapeActor2D, BaseShape):
    draw_func = rl.DrawCircle
    draw_wire_func = rl.DrawCircleLines
    radius: float = 1.

    @override
    def draw(self):
        self._draw(int(self.position.x), int(self.position.y), self.radius, self.color)

@dataclass
class TriangleNode(ShapeActor2D, BaseShape):
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
class EllipseNode(ShapeActor2D, BaseShape):
    draw_func = rl.DrawEllipse
    draw_wire_func = rl.DrawEllipseLines
    width: float = 1.
    height: float = 1.

    @override
    def draw(self):
        self._draw(self.position.x, self.position.y, self.width, self.height, self.color)

@dataclass
class SpriteNode(Actor2D):
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
class LabelNode(Actor2D):
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
class SoundNode(AudioActor):
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
class MusicNode(AudioActor):
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
