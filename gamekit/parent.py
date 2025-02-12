from typing import Optional
from .actor import ActorType

class Parent:
    def __init__(self):
        self.children = []

    def add_child(self, node: ActorType):
        self.children.append(node)

    def add_children(self, nodes: ActorType | list[ActorType]):
        for node in nodes if isinstance(nodes, list) else [nodes]:
            self.add_child(node)

    def get_children(self, name: Optional[str] = ""):
        return [x for x in self.children if x.name == name]

    def rem_children(self, name: Optional[str] = ""):
        self.children = [x for x in self.children if x.name != name]

    def clear_children(self):
        self.children = []
