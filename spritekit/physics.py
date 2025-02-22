from .scene import Scene
from typing import override
import pyray as r
import atexit
from .math import Vector2

class PhysicsBody:
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

    def __del__(self):
        r.destroy_physics_body(self._body)

class PhysicsScene(Scene):
    def __init__(self, gravity: Vector2 = Vector2([0, 9.81]), **kwargs):
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
