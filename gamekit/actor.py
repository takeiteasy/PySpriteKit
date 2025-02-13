from .vector import Vector3
from .scene import Parent
from dataclasses import dataclass
from typing import Optional

class ActorType:
    pass

@dataclass
class Actor(ActorType, Parent):
    name: str = ""
    position: Vector3 = Vector3()
    rotation: Vector3 | float = 0.
    scale: Vector3 | float = 1.

    def __str__(self):
        return f"(Node name:\"{self.name}\" position:self.position, rotation:{self.rotation}, scale:{self.scale})"

    def __eq__(self, other: ActorType):
        return self._name == other.name

    def add_child(self, node: ActorType):
        node.parent = self
        self.children.append(node)

    def draw(self, indent: Optional[int] = 0):
        if indent:
            print(" " * (indent * 4) + "⤷ ", end="")
        print(str(self))
        for child in self.children:
            child.draw(indent=indent+1)