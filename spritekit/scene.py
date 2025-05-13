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

from typing import Optional
from enum import Enum
from dataclasses import dataclass, field

from .actor import ActorParent
from . import _renderer as renderer
from . import _drawable as drawable
from .window import window_size
from .camera import Camera

import transitions

class SceneType:
    pass

class Scene(SceneType, ActorParent):
    background_color = (0., 0., 0., 1.)
    window_size = (640, 480)
    window_title = "SpriteKit"
    window_hints = None
    frame_limit = None

    def __init__(self):
        self.parent = None
        ActorParent.__init__(self)
        self._camera = Camera()
        self.viewport = window_size()
        self.clear_color = self.__class__.background_color
    
    @property
    def viewport(self):
        return renderer.get_viewport()
    
    @viewport.setter
    def viewport(self, value: tuple[int, int]):
        renderer.set_viewport(value)
    
    @property
    def clear_color(self):
        return renderer.get_clear_color()
    
    @clear_color.setter
    def clear_color(self, value: tuple | list):
        renderer.set_clear_color(drawable.convert_color(value))
    
    @property
    def camera(self):
        return self._camera
    
    @camera.setter
    def camera(self, value: Camera):
        self._camera = value
        self._camera.dirty = True
    
    def remove_me(self):
        if not self.parent:
            quit()
        else:
            self.parent.remove_scene(self)
    
    def enter(self):
        pass

    def exit(self):
        pass

    def step(self, delta):
        for child in self.children:
            child.step(delta)

    def draw(self):
        if self.camera.dirty:
            renderer.set_world_matrix(self.camera.matrix)
        for child in reversed(self.children):
            child.draw()

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

def _explode(t: Transition):
    a = {k: t.__dict__[k] for k in t.__annotations__.keys() if not k.startswith("_")}
    for k, v in a.items():
        if v is None:
            a.pop(k)
    b = a.pop("kwargs")
    a.update(**b)
    return a

class FiniteStateMachine:
    states: list[str | Enum] = []
    transitions: list[dict | Transition] = []

    def __init__(self, **kwargs):
        if self.states:
            if not "initial" in kwargs:
                kwargs["initial"] = self.states[0]
            self._machine = transitions.Machine(model=self,
                                                states=self.states,
                                                transitions=[_explode(t) if isinstance(t, Transition) else t for t in self.transitions],
                                                **kwargs)
        else:
            self._machine = None

class Director(Scene, FiniteStateMachine):
    def __init__(self, **kwargs):
        Scene.__init__(self)
        FiniteStateMachine.__init__(self, **kwargs)

__all__ = ["Scene", "SceneType", "Transition", "FiniteStateMachine", "Director"]