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

@dataclass
class ActorType:
    name: str = field(default_factory=lambda: uuid4().hex)
    on_added: Optional[Callable[[], Any]] = None
    on_removed: Optional[Callable[[], Any]] = None

class ActorParent:
    def __init__(self):
        self.children = []
        self.parent = None
    
    def add(self, children: ActorType | list[ActorType]):
        if isinstance(children, ActorType):
            children = [children]
        for child in children:
            self.children.insert(0, child)
            if hasattr(child, 'on_added') and callable(child.on_added):
                child.on_added()
            child.parent = self
    
    def _remove_child(self, child: str | ActorType):
        matching = []
        not_matching = []
        for child in self.children:
            match = child.name == child if isinstance(child, str) else child == child
            if match:
                matching.append(child)
            else:
                not_matching.append(child)
        for child in matching:
            if hasattr(child, 'on_removed') and callable(child.on_removed):
                child.on_removed()
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

class Actor(ActorType, ActorParent):
    def __init__(self, **kwargs):
        ActorType.__init__(self, **kwargs)
        ActorParent.__init__(self)

    def __str__(self):
        return f"(Node(\"{self.name}\", {self.__class__.__name__}))"

    def remove(self):
        if self.parent is not None and issubclass(self.parent, ActorParent):
            self.parent.remove(self)
    
    def step(self, delta: float):
        for child in self.children:
            child.step(delta)
    
    def draw(self):
        for child in self.children:
            child.draw()

__all__ = ["Actor", "ActorParent"]