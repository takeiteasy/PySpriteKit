from slimrr import *
import pyray as r
import math

_K = 1./3.

def _get_pixels_per_meter() -> Vector2:
    monitor = r.get_current_monitor()
    monitor_size = Vector2([float(r.get_monitor_physical_width(monitor)),
                            float(r.get_monitor_physical_height(monitor))])
    fb_size = Vector2([float(r.get_render_width()),
                       float(r.get_render_height())])
    return fb_size / monitor_size

def _calculate_normals(vertices: list[Vector2]) -> list[Vector2]:
    l = len(vertices)
    return [vector2.normalise(vertices[i + 1 if (i + 1)  % l else 0] - vertices[i]) for i in range(l)]

def _calculate_shape_properties(vertices: list[Vector2]) -> tuple[Vector2, float, float]:
    center = Vector2([0., 0.])
    area = 0.
    inertia = 0.
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        d  = vector2.cross(p1, p2)
        area += d / 2.
        center += area * _K * (p1 + p2)
        inertia += (.25 * _K * d) * ((p1.x * p1.x + p2.x * p1.x + p2.x * p2.x) + (p1.y * p1.y + p2.y * p1.y + p2.y * p2.y))
    return center * (1. / area), area, inertia

class RigidBody:
    def __init__(self,
                 position: Vector2,
                 enabled: bool = True,
                 mass: float = 1.,
                 inertia: float = 1.,
                 static_friction: float = .4,
                 dynamic_friction: float = .2,
                 restitution: float = .0,
                 gravity_scale: float = 1.,
                 freeze_orientation: bool = False,
                 initial_rotation: float = 0.,
                 initial_velocity: Vector2 = Vector2([0., 0.]),
                 initial_angular_velocity: float = 0.,
                 initial_torque: float = 0.,
                 initial_force: Vector2 = Vector2([0., 0.]),
                 vertices: list[Vector2] = [],
                 normals: list[Vector2] = []):
        self.enabled = enabled
        self.position = position
        self.velocity = initial_velocity
        self.force = initial_force
        self.angular_velocity = initial_angular_velocity
        self.torque = initial_torque
        self.rotation = initial_rotation
        self.inertia = inertia
        self.inv_inertia = 1. / inertia if inertia != 0. else 0.
        self.mass = mass
        self.inv_mass = 1. / mass if mass != 0. else 0.
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.restitution = restitution
        self.gravity_scale = gravity_scale
        self.is_grounded = False
        self.freeze_orientation = freeze_orientation
        self.vertices = vertices
        self.normals = normals
    
    def apply_force(self, force: Vector2):
        self.force += force
    
    def apply_torque(self, torque: float):
        self.torque += torque

class PolygonTypeBody(RigidBody):
    def __init__(self, position: Vector2, vertices: list[Vector2], density: float = 1., **kwargs):
        self.center, area, inertia = _calculate_shape_properties(vertices)
        super().__init__(position=position,
                         mass=density * area,
                         inertia=density * inertia,
                         vertices=vertices,
                         normals=_calculate_normals(vertices),
                         **kwargs)
        self.density = density

class PolygonBody(PolygonTypeBody):
    def __init__(self, position: Vector2, radius: float, sides: int, density: float = 1., **kwargs):
        super().__init__(position=position,
                         vertices=[position + Vector2([radius * math.cos(2. * math.pi * i / sides), radius * math.sin(2. * math.pi * i / sides)]) for i in range(sides)],
                         density=density,
                         **kwargs)
        self.radius = radius
        self.sides = sides

class RectangleBody(PolygonTypeBody):
    def __init__(self, position: Vector2, width: float, height: float, density: float = 1., **kwargs):
        half_width = width / 2
        half_height = height / 2
        super().__init__(position=position,
                         vertices=[Vector2([position.x + half_width, position.y - half_height]),
                                   Vector2([position.x + half_width, position.y + half_height]),
                                   Vector2([position.x - half_width, position.y + half_height]),
                                   Vector2([position.x - half_width, position.y - half_height])],
                         density=density, 
                         **kwargs)
        self.width = width
        self.height = height

class CircleBody(RigidBody):
    def __init__(self, position: Vector2, radius: float, density: float = 1., **kwargs):
        mass = math.pi * radius ** 2 * density
        super().__init__(position=position,
                         mass=mass,
                         inertia=mass * radius ** 2,
                         **kwargs)
        self.radius = radius
        self.density = density

class PhysicsManifold:
    def __init__(self, bodyA: RigidBody, bodyB: RigidBody):
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.penetration = 0.
        self.normal = Vector2([0., 0.])
        self.contacts = []
        self.restitution = 0.
        self.dynamic_friction = 0.
        self.static_friction = 0.
        self.solve()
    
    def solve(self):
        pass

class PhysicsWorld:
    def __init__(self, time_step: float = 1. / 60.):
        self.bodies = []
        self.pixels_per_meter = _get_pixels_per_meter()
        self._accumulator = 0.
        self._last_time = r.get_frame_time()
        self._time_step = time_step
    
    def add_body(self, body: RigidBody):
        self.bodies.append(body)

    def _step(self):
        for body in self.bodies:
            body.is_grounded = False
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                bodyA = self.bodies[i]
                bodyB = self.bodies[j]
    
    def step(self):
        t = r.get_frame_time()
        self._accumulator += t - self._last_time
        while self._accumulator >= self._time_step:
            self._step()
            self._accumulator -= self._time_step
        self._last_time = t
    
    def draw(self):
        for body in self.bodies:
            if isinstance(body, CircleBody):
                r.draw_circle(int(body.position.x), int(body.position.y), body.radius, r.BLACK)
            else:
                for i in range(len(body.vertices)):
                    v1 = body.vertices[i]
                    v2 = body.vertices[(i + 1) % len(body.vertices)]
                    r.draw_line(int(v1.x), int(v1.y), int(v2.x), int(v2.y), r.BLACK)

r.init_window(800, 600, "Physics Test")

world = PhysicsWorld()

polygon = PolygonBody(position=Vector2([100., 100.]), radius=20., sides=3)
rectangle = RectangleBody(position=Vector2([150., 100.]), width=50., height=50.)
circle = CircleBody(position=Vector2([200., 100.]), radius=10.)
world.add_body(polygon)
world.add_body(rectangle)
world.add_body(circle)

while not r.window_should_close():
    r.begin_drawing()
    r.clear_background(r.RAYWHITE)
    world.step()
    world.draw()
    r.end_drawing()

r.close_window()