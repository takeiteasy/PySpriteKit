from slimrr import *
import pyray as r

def _get_center(vertices: list[Vector2]) -> Vector2:
    return Vector2([sum(v.x for v in vertices) / len(vertices),
                    sum(v.y for v in vertices) / len(vertices)])

def _normalize(vertices: list[Vector2]) -> list[Vector2]:
    minx = min(v.x for v in vertices)
    maxx = max(v.x for v in vertices)
    miny = min(v.y for v in vertices)
    maxy = max(v.y for v in vertices)
    return [Vector2([x - minx - (maxx - minx) / 2., y - miny - (maxy - miny) / 2.]) for x, y in vertices]

def _get_pixels_per_meter():
    monitor = r.get_current_monitor()
    monitor_size = Vector2([float(r.get_monitor_physical_width(monitor)),
                            float(r.get_monitor_physical_height(monitor))])
    fb_size = Vector2([float(r.get_render_width()),
                       float(r.get_render_height())])
    return fb_size / monitor_size

class RigidBody:
    def __init__(self, vertices: list[Vector2], mass: float = 1., is_static: bool = False):
        self.vertices = _normalize(vertices)
        self.position = _get_center(vertices)
        self.velocity = Vector2([0., 0.])
        self.force = Vector2([0., 0.])
        self.mass = mass
        self.is_static = is_static

    def apply_force(self, force: Vector2):
        self.force += force

    def update(self, dt: float):
        self.velocity += self.force * (dt / self.mass)
        self.position += self.velocity * dt
        self.force = Vector2([0., 0.])
    
    def support(self, direction: Vector2) -> Vector2:
        return max(self.vertices, key=lambda v: Vector2.dot(v, direction))
 
    def draw(self):
        vertices = [v + self.position for v in self.vertices]
        for v in vertices:
            r.draw_circle(int(v.x), int(v.y), 1, r.RED)
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            r.draw_line(int(vertices[i].x), int(vertices[i].y), int(vertices[j].x), int(vertices[j].y), r.RED)

class Polygon(RigidBody):
    def __init__(self, vertices: list[Vector2], mass: float = 1., is_static: bool = False):
        super().__init__(vertices=vertices, mass=mass, is_static=is_static)

class Point(RigidBody):
    def __init__(self, position: Vector2, mass: float = 1., is_static: bool = False):
        super().__init__(vertices=[position], mass=mass, is_static=is_static)
    
    def support(self, direction: Vector2) -> Vector2:
        return self.position
    
    def draw(self):
        r.draw_circle(int(self.position.x), int(self.position.y), 1, r.RED)

class Line(RigidBody):
    def __init__(self, start: Vector2, end: Vector2, mass: float = 1., is_static: bool = False):
        super().__init__(vertices=[start, end], mass=mass, is_static=is_static)
        self.position = (start + end) / 2.
    
    def support(self, direction: Vector2) -> Vector2:
        return self.start if self.start.dot(direction) > self.end.dot(direction) else self.end
    
    def draw(self):
        r.draw_line(int(self.vertices[0].x), int(self.vertices[0].y), int(self.vertices[1].x), int(self.vertices[1].y), r.RED)

class Rectangle(RigidBody):
    def __init__(self, position: Vector2, width: float, height: float, mass: float = 1., is_static: bool = False):
        self.width = width
        self.height = height
        super().__init__(vertices=[position,
                                   position + Vector2([width, 0]),
                                   position + Vector2([width, height]),
                                   position + Vector2([0, height])],
                         mass=mass,
                         is_static=is_static)
    
class Circle(RigidBody):
    def __init__(self, position: Vector2, radius: float, mass: float = 1., is_static: bool = False):
        super().__init__(vertices=[position], mass=mass, is_static=is_static)
        self.radius = radius
    
    def support(self, direction: Vector2) -> Vector2:
        return self.position + self.radius * direction.normalize()

    def draw(self):
        r.draw_circle(int(self.position.x), int(self.position.y), self.radius, r.RED)

class PhysicsWorld:
    def __init__(self):
        self.bodies = []
        self.gravity = Vector2([0., 9.81])
        self.pixels_per_meter = _get_pixels_per_meter()

    def step(self, dt: float):
        for body in self.bodies:
            if not body.is_static:
                body.apply_force(self.gravity * body.mass)
                body.update(dt * self.pixels_per_meter)
    
    def draw(self):
        for body in self.bodies:
            body.draw()

r.init_window(1280, 800, "i love physics i am a physics engine")

world = PhysicsWorld()

circle = Circle(Vector2([150., 350.]), 10.)
rectangle = Rectangle(Vector2([100., 300.]), 50., 50.)
pentagon = Polygon([Vector2([200., 300.]),
                    Vector2([235., 330.]),
                    Vector2([220., 370.]),
                    Vector2([180., 370.]),
                    Vector2([165., 330.])])
world.bodies.append(circle)
world.bodies.append(rectangle)
world.bodies.append(pentagon)

while not r.window_should_close():
    r.begin_drawing()
    r.clear_background(r.RAYWHITE)
    world.step(r.get_frame_time())
    world.draw()
    r.end_drawing()

r.close_window()