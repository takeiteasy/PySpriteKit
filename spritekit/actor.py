# spritekit/actors.py
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
from typing import Optional, Callable, Any
from uuid import uuid4

class ActorType:
    pass

class ActorParent:
    def __init__(self):
        self.children = []
        self.parent = None
    
    def add(self, children: ActorType | list[ActorType]):
        if not isinstance(children, list):
            children = [children]
        for child in children:
            self.children.insert(0, child)
            if hasattr(child, 'on_added') and callable(child.on_added):
                child.on_added()
            child.parent = self
    
    def _remove_child(self, child: str | ActorType):
        matching = []
        not_matching = []
        for _child in self.children:
            match = _child.name == child if isinstance(child, str) else _child == child
            if match:
                matching.append(_child)
            else:
                not_matching.append(_child)
        for _child in matching:
            if hasattr(_child, 'on_removed') and callable(_child.on_removed):
                _child.on_removed()
        self.children = not_matching

    def remove(self, child: list[ActorType] | ActorType | str):
        if isinstance(child, str):
            child = self.find(child)
        if isinstance(child, ActorType):
            child = [child]
        for c in child:
            self._remove_child(c)

    def find(self, name: str):
        return [child for child in self.children if child.name == name]
    
    def sort(self, key: Callable[[ActorType], Any]):
        self.children.sort(key=key)
    
    def clear(self):
        self.children = []

@dataclass
class Actor(ActorType, ActorParent):
    name: str = field(default_factory=lambda: uuid4().hex)
    on_added: Optional[Callable[[], Any]] = None
    on_removed: Optional[Callable[[], Any]] = None

    def __init__(self, **kwargs):
        ActorType.__init__(self)
        ActorParent.__init__(self)

    def __str__(self):
        return f"(Node(\"{self.name}\", {self.__class__.__name__}))"

    def clone(self, clone_children: bool = False):
        new_actor = self.__class__(**self.__dict__)
        if clone_children:
            for child in self.children:
                new_actor.add(child.clone())
        return new_actor

    def remove_me(self):
        if self.parent is not None and issubclass(self.parent, ActorParent):
            self.parent.remove(child=self)
    
    def step(self, delta: float):
        for child in self.children:
            child.step(delta)
    
    def draw(self):
        for child in self.children:
            child.draw()

__all__ = ["Actor", "ActorParent", "ActorType"]