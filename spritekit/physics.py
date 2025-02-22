from .scene import Scene
from typing import override
import pyray as r
import atexit
from .math import Vector2

__all__ = ["PhysicsBody", "PhysicsScene"]

class PhysicsBody(object):
    __slots__ = ["_body"]

    def __init__(self, body: r.PhysicsBodyData, **kwargs):
        self._body = body
        for key, value in kwargs.items():
            if hasattr(self._body, key):
                setattr(self._body, key, value)
            else:
                raise AttributeError(f"{key} not found in PhysicsBody or _body")

    @property
    def body(self):
        return self._body
    
    def __getattr__(self, name: str):
        try:
            _body = object.__getattribute__(self, "_body")
            _val = _body.__getattribute__(name)
            if _val is not None:
                return _val
            else:
                object.__getattribute__(self, name)
        except ValueError:
            raise AttributeError(f"{name} not found in PhysicsBody or _body")
        except AttributeError:
            return object.__getattribute__(self, name)
    
    def __setattr__(self, name: str, value):
        if name in object.__getattribute__(self, '__slots__'):
            object.__setattr__(self, name, value)
        else:
            try:
                _body = object.__getattribute__(self, '_body')
                _val = _body.__getattribute__(name)
                if _val is not None:
                    _body.__setattr__(name, value)
                else:
                    object.__setattr__(self, name, value)
            except ValueError:
                raise AttributeError(f"{name} not found in PhysicsBody or _body")
    
    def add_force(self, force: Vector2):
        r.physics_add_force(self._body, r.Vector2(force.x, force.y))
    
    def add_torque(self, torque: float):
        r.physics_add_torque(self._body, torque)
    
    def shatter(self):
        r.physics_shatter(self._body)

    def shape_vertex(self, index: int):
        x = r.get_physics_shape_vertex(self._body, index)
        return Vector2(x.x, x.y)
    
    def set_rotation(self, rotation: float):
        r.set_physics_body_rotation(self._body, rotation)

    def __del__(self):
        r.destroy_physics_body(self._body)

class PhysicsCircle(PhysicsBody):
    def __init__(self,
                 position: Vector2,
                 raidus: float,
                 density: float,
                 **kwargs):
        PhysicsBody.__init__(self, r.create_physics_body_circle(position, raidus, density), **kwargs)

class PhysicsRectangle(PhysicsBody):
    def __init__(self,
                 position: Vector2,
                 width: float,
                 height: float,
                 density: float,
                 **kwargs):
        PhysicsBody.__init__(self, r.create_physics_body_rectangle(position, width, height, density), **kwargs)

class PhysicsPolygon(PhysicsBody):
    def __init__(self,
                 position: Vector2,
                 raidus: float,
                 sides: int,
                 density: float,
                 **kwargs):
        PhysicsBody.__init__(self, r.create_physics_body_polygon(position, raidus, sides, density), **kwargs)

class PhysicsScene(Scene):
    def __init__(self, gravity: Vector2 = Vector2([0, -9.81]), **kwargs):
        super().__init__(**kwargs)
        r.init_physics()
        r.set_physics_gravity(gravity.x, gravity.y)
        atexit.register(r.close_physics)

    @override
    def step(self, delta):
        r.update_physics()
        super().step(delta)

    @override
    def step_background(self, delta):
        if self.run_in_background:
            self.step(delta)
