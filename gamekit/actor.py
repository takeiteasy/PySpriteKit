from .vector import Vector2
from .parent import Parent, ActorType
from dataclasses import dataclass, field
from typing import Optional
from raylib.colors import *
import raylib as rl
import pyray as r

@dataclass
class Actor(Parent):
    name: str = ""
    
    def __str__(self):
        return f"(Node({self.__class__.__name__}) {" ".join([f"{key}:{getattr(self, key)}" for key in list(vars(self).keys())])})"
    
    def add_child(self, node: ActorType):
        node.parent = self
        self.children.append(node)

    def draw(self, indent: Optional[int] = 0):
        if indent:
            print(" " * (indent * 4) + "⤷ ", end="")
        print(str(self))
        for child in self.children:
            child.draw(indent=indent+1 )

@dataclass
class Actor2D(Actor):
    position: Vector2 = field(default_factory=Vector2)
    rotation: float = 0.
    scale: float = 1.
    origin: Vector2 = field(default_factory=lambda: Vector2([0.5, 0.5]))
    color: r.Color = RAYWHITE

    def _offset(self):
        return self.origin * Vector2([-self.width, -self.height])

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

    def draw(self):
        self._draw(self.position.x, self.position.y, self.end.x, self.end.y, self.color)

@dataclass
class RectangleActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawRectangleRec
    draw_wire_func = rl.DrawRectangleLinesEx
    width: float = 1.
    height: float = 1.
    origin: Vector2 = field(default_factory=lambda: Vector2([0.5, 0.5]))

    def draw(self):
        pos = self._offset()
        rec = r.Rectangle(pos.x, pos.y, self.width, self.height)
        if self.wireframe:
            self._draw(rec, self.line_thickness, self.color)
        else:
            self._draw(rec, self.color)

@dataclass
class CircleActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawCircle
    draw_wire_func = rl.DrawCircleLines
    radius: float = 1.

    def draw(self):
        self._draw(int(self.position.x), int(self.position.y), self.radius, self.color)

@dataclass
class TriangleActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawTriangle
    draw_wire_func = rl.DrawTriangleLines
    position2: Vector2 = field(default_factory=Vector2)
    position3: Vector2 = field(default_factory=Vector2)

    def draw(self):
        self._draw([self.position.x, self.position.y], [self.position2.x, self.position2.y], [self.position3.x, self.position3.y], self.color)

@dataclass
class EllipseActor2D(ShapeActor2D, BaseShape):
    draw_func = rl.DrawEllipse
    draw_wire_func = rl.DrawEllipseLines
    width: float = 1.
    height: float = 1.

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

class Line2D(LineActor2D):
    pass

class Rectangle(RectangleActor2D):
    pass

class Circle(CircleActor2D):
    pass

class Triangle(TriangleActor2D):
    pass

class Ellipse(EllipseActor2D):
    pass

class Sprite(SpriteActor2D):
    pass
