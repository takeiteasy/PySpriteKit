# spritekit/scene.py
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

from .actor import ActorType, ActorParent
from typing import override
import atexit
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

import transitions as t
import quickwindow as qw
import raudio as r
import OpenGL.GL as GL

__all__ = ["Scene", "main", "Transition", "FiniteStateMachine"]

__scene__ = []
__next_scene = None
__drop_scene = None

@dataclass
class Transition:
    trigger: str
    source: str | Enum | list
    dest: str | Enum
    conditions: Optional[str | list[str]] = None
    unless: Optional[str | list[str]] = None
    before: Optional[str | list[str]] = None
    after: Optional[str | list[str]] = None
    prepare: Optional[str | list[str]] = None
    kwargs: dict = field(default_factory=dict)

    def explode(self):
        transition_args = {k: v for k, v in self.__dict__.items() if k != 'kwargs' and v is not None}
        transition_args.update(**self.kwargs)  # unpack kwargs
        return transition_args

class FiniteStateMachine:
    states: list[str | t.State] = []
    transitions: list[dict | Transition] = []

    def __init__(self, **kwargs):
        if self.__class__.states:
            if not "initial" in kwargs:
                kwargs["initial"] = self.__class__.states[0]
            self.fsm = t.Machine(model=self,
                                 states=self.__class__.states,
                                 transitions=[t.explode() if isinstance(t, Transition) else t for t in self.__class__.transitions],
                                 **kwargs)
        else:
            self.fsm = None

class Scene(FiniteStateMachine, ActorParent):
    config: dict = {}

    def __init__(self, **kwargs):
        FiniteStateMachine.__init__(self, **kwargs)
        self.target = (0, 0)
        self.offset = (0, 0)
        self.zoom = 1.
        self.clear_color = (1., 1., 1., 1.)
        self.run_in_background = False
        self.assets = {} # TODO: Store and restore assets to __cache in raylib.py

    @override
    def add_child(self, node: ActorType):
        if node and isinstance(node, ActorType):
            node.__dict__["scene"] = self
            self._add_child(node)
        else:
            raise RuntimeError("Invalid Node")

    def enter(self):
        pass

    def restored(self):
        pass

    def background(self):
        pass

    def exit(self):
        pass

    def step(self, delta):
        for child in reversed(self.all_children()):
            child.step(delta)

    def step_background(self, delta):
        if self.run_in_background:
            self.step(delta)

    def draw(self):
        GL.glClearColor(*self.clear_color)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT) # type: ignore

    def draw_background(self):
        if self.run_in_background:
            self.draw()

    @classmethod
    def push_scene(cls, scene):
        if not isinstance(scene, Scene):
            raise RuntimeError("Invalid Scene")
        global __next_scene
        if __next_scene is not None:
            raise RuntimeError("Next scene already queued")
        __next_scene = scene

    @classmethod
    def drop_scene(cls):
        global __scene__, __drop_scene
        if __drop_scene is not None:
            raise RuntimeError("Drop scene already queued")
        __drop_scene = __scene__[-1:]

    @classmethod
    def first_scene(cls):
        global __scene__, __drop_scene
        __drop_scene = __scene__[1:]

    @classmethod
    def current_scene(cls):
        if not __scene__:
            raise RuntimeError("No active Scene")
        return __scene__[-1]

    @classmethod
    def main_scene(cls):
        global __next_scene
        if __next_scene is None:
            raise RuntimeError("No next scene queued")
        return __scene__[0]
    
    @property
    def width(self):
        return r.get_screen_width()
    
    @property
    def height(self):
        return r.get_screen_height()

def main(cls):
    global __scene__, __drop_scene, __next_scene
    if __scene__:
        raise RuntimeError("There can only be one @main_scene")
    with qw.quick_window(**cls.config) as wnd:
        r.initialize()
        atexit.register(r.shutdown)
        scn = cls()
        __scene__.append(scn)
        scn.enter()

        for dt in wnd.loop():
            if not __scene__:
                break
            for e in wnd.events():
                pass

            if len(__scene__) > 1:
                for _scn in __scene__[:-1]:
                    _scn.step_background(dt)
            scn.step(dt)
            if len(__scene__) > 1:
                for _scn in __scene__[:-1]:
                    _scn.draw_background()
            scn.draw()
            if __drop_scene:
                if isinstance(__drop_scene, list):
                    for _scn in reversed(__drop_scene):
                        _scn.exit()
                elif isinstance(__drop_scene, Scene):
                    __drop_scene.exit()
                else:
                    raise RuntimeError("Invalid Scene")
                __scene__ = __scene__[:-len(__drop_scene)]
                if __scene__:
                    scn = __scene__[-1]
                    scn.restored()
                __drop_scene = None
            if __next_scene:
                if isinstance(__next_scene, Scene):
                    if __scene__:
                        __scene__[-1].background()
                    __scene__.append(__next_scene)
                    scn = __next_scene
                    scn.enter()
                    __next_scene = None
                else:
                    raise RuntimeError("Invalid Scene")
    return cls
