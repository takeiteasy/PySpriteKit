from slimrr import *
import pyray as r
from enum import Enum

def _get_center(vertices: list[Vector2]) -> Vector2:
    l = len(vertices)
    return Vector2([sum(v.x for v in vertices), sum(v.y for v in vertices)]) / l

def _normalize_vertices(vertices: list[Vector2]) -> list[Vector2]:
    minx = min(v.x for v in vertices)
    maxx = max(v.x for v in vertices)
    miny = min(v.y for v in vertices)
    maxy = max(v.y for v in vertices)
    return [Vector2([x - minx - (maxx - minx) / 2., y - miny - (maxy - miny) / 2.]) for x, y in vertices]

def _get_pixels_per_meter() -> Vector2:
    monitor = r.get_current_monitor()
    monitor_size = Vector2([float(r.get_monitor_physical_width(monitor)),
                            float(r.get_monitor_physical_height(monitor))])
    fb_size = Vector2([float(r.get_render_width()),
                       float(r.get_render_height())])
    return fb_size / monitor_size

def _triple_product(a: Vector2, b: Vector2, c: Vector2) -> Vector2:
    xy, _ = Vector2.from_vector3(vector3.cross(Vector3([*a, 0.]).cross(Vector3([*b, 0.])), Vector3([*c, 0.])))
    return xy

class RigidBody:
    def __init__(self, vertices: list[Vector2], mass: float = 1., is_static: bool = False):
        self._vertices = _normalize_vertices(vertices)
        self.position = _get_center(vertices)
        self.velocity = Vector2([0., 0.])
        self.force = Vector2([0., 0.])
        self.mass = mass
        self.is_static = is_static
    
    @property
    def vertices(self) -> list[Vector2]:
        return [v + self.position for v in self._vertices]

    def apply_force(self, force: Vector2):
        self.force += force

    def update(self, dt: float):
        self.velocity += self.force * (dt / self.mass)
        self.position += self.velocity * dt
        self.force = Vector2([0., 0.])
    
    def support(self, direction: Vector2) -> Vector2:
        return max(self.vertices, key=lambda v: v.dot(direction))
 
    def draw(self):
        vertices = self.vertices
        for v in vertices:
            r.draw_circle(int(v.x), int(v.y), 1, r.RED)
        l = len(vertices)
        for i in range(l):
            j = (i + 1) % l
            r.draw_line(int(vertices[i].x), int(vertices[i].y), int(vertices[j].x), int(vertices[j].y), r.RED)

class PolygonBody(RigidBody):
    def __init__(self, vertices: list[Vector2], mass: float = 1., is_static: bool = False):
        super().__init__(vertices=vertices, mass=mass, is_static=is_static)

class PointBody(RigidBody):
    def __init__(self, position: Vector2, mass: float = 1., is_static: bool = False):
        super().__init__(vertices=[position], mass=mass, is_static=is_static)
    
    def support(self, direction: Vector2) -> Vector2:
        return self.position
    
    def draw(self):
        r.draw_circle(int(self.position.x), int(self.position.y), 1, r.RED)

class LineBody(RigidBody):
    def __init__(self, start: Vector2, end: Vector2, mass: float = 1., is_static: bool = False):
        super().__init__(vertices=[start, end], mass=mass, is_static=is_static)
        self.position = (start + end) / 2.
    
    def support(self, direction: Vector2) -> Vector2:
        return self.start if self.start.dot(direction) > self.end.dot(direction) else self.end
    
    def draw(self):
        r.draw_line(int(self.vertices[0].x), int(self.vertices[0].y), int(self.vertices[1].x), int(self.vertices[1].y), r.RED)

class RectangleBody(RigidBody):
    def __init__(self, position: Vector2, width: float, height: float, mass: float = 1., is_static: bool = False):
        self.width = width
        self.height = height
        super().__init__(vertices=[position,
                                   position + Vector2([width, 0.]),
                                   position + Vector2([width, height]),
                                   position + Vector2([0., height])],
                         mass=mass,
                         is_static=is_static)
    
class CircleBody(RigidBody):
    def __init__(self, position: Vector2, radius: float, mass: float = 1., is_static: bool = False):
        super().__init__(vertices=[position], mass=mass, is_static=is_static)
        self.radius = radius
        self.position = position
    
    def support(self, direction: Vector2) -> Vector2:
        return self.position if direction.squared_length < 1e-10 else self.position + self.radius * direction.normalised

    def draw(self):
        r.draw_circle(int(self.position.x), int(self.position.y), self.radius, r.RED)

class SimplexEvolution(Enum):
    NO_INTERSECTION = 0
    FOUND_INTERSECTION = 1
    STILL_EVOLVING = 2

class PolygonWinding(Enum):
    CLOCKWISE = 0
    ANTICLOCKWISE = 1

class Edge:
    def __init__(self, distance: float, normal: Vector2, index: int):
        self.distance = distance
        self.normal = normal
        self.index = index

class Collision:
    def __init__(self, shapeA: RigidBody, shapeB: RigidBody, intersection: Vector2, normal: Vector2, overlap: float):
        self.intersection = intersection
        self.normal = normal
        self.overlap = overlap
        self.shapeA = shapeA
        self.shapeB = shapeB
    
    def __str__(self):
        return f"Collision(intersection={self.intersection}, normal={self.normal}, overlap={self.overlap})"

class GJK:
    MAX_ITERATIONS = 20
    MAX_INTERSECTIONS = 32

    def __init__(self, shapeA: RigidBody, shapeB: RigidBody):
        self.shapeA = shapeA
        self.shapeB = shapeB
        self.direction = Vector2([0., 0.])
        self.simplex = []

    def support(self, direction: Vector2) -> Vector2:
        return self.shapeA.support(direction) - self.shapeB.support(direction * -1.)
    
    def evolve(self) -> SimplexEvolution:
        match len(self.simplex):
            case 0:
                self.direction = self.shapeB.position - self.shapeA.position
                if self.direction.squared_length < 1e-10:
                    self.direction = Vector2([1., 0.])
            case 1:
                self.direction *= -1.
            case 2:
                cb = self.simplex[1] - self.simplex[0]
                c0 = self.simplex[0] * -1.
                self.direction = _triple_product(cb, c0, cb)
            case 3:
                a0 = self.simplex[2] * -1.
                ab = self.simplex[1] - self.simplex[2]
                ac = self.simplex[0] - self.simplex[2]
                ab_perp = _triple_product(ac, ab, ab)
                ac_perp = _triple_product(ab, ac, ac)
                if ab_perp.dot(a0) > 0.:
                    self.simplex.pop(0)
                elif ac_perp.dot(a0) > 0.:
                    self.simplex.pop(1)
                else:
                    return SimplexEvolution.FOUND_INTERSECTION
            case _:
                assert False
        vertex = self.support(self.direction)
        self.simplex.append(vertex)
        return SimplexEvolution.STILL_EVOLVING if self.direction.dot(vertex) > -1e-10 else SimplexEvolution.NO_INTERSECTION

    def test(self):
        for _ in range(GJK.MAX_ITERATIONS):
            match self.evolve():
                case SimplexEvolution.FOUND_INTERSECTION:
                    return True
                case SimplexEvolution.NO_INTERSECTION:
                    return False
                case SimplexEvolution.STILL_EVOLVING:
                    pass
        return False
    
    def closest_edge(self, winding: PolygonWinding) -> Edge:
        closest_distance = float('inf')
        closest_normal = Vector2([0., 0.])
        closest_index = -1
        for i in range(len(self.simplex)):
            j = (i + 1) % len(self.simplex)
            line = self.simplex[j] - self.simplex[i]
            normal = Vector2([line.y, -line.x]) if winding == PolygonWinding.CLOCKWISE else Vector2([-line.y, line.x])
            normal = normal.normalised
            dist = normal.dot(self.simplex[i])
            if dist < closest_distance:
                closest_distance = dist
                closest_normal = normal
                closest_index = j
        return Edge(closest_distance, closest_normal, closest_index)

    def intersection(self) -> Collision | None:
        if not self.test():
            return None
        
        e0 = (self.simplex[1].x - self.simplex[0].x) * (self.simplex[1].y + self.simplex[0].y)
        e1 = (self.simplex[2].x - self.simplex[1].x) * (self.simplex[2].y + self.simplex[1].y)
        e2 = (self.simplex[0].x - self.simplex[2].x) * (self.simplex[0].y + self.simplex[2].y)
        winding = PolygonWinding.CLOCKWISE if e0 + e1 + e2 >= 0. else PolygonWinding.ANTICLOCKWISE
        
        for _ in range(GJK.MAX_INTERSECTIONS):
            edge = self.closest_edge(winding)
            support = self.support(edge.normal)
            distance = edge.normal.dot(support)
            if abs(distance - edge.distance) <= 1e-10:
                return Collision(self.shapeA, self.shapeB, support, edge.normal, abs(edge.distance))
            else:
                self.simplex.insert(edge.index, support)
        return None

def gjk(shapeA: RigidBody, shapeB: RigidBody) -> list[Vector2]:
    gjk = GJK(shapeA, shapeB)
    return gjk.simplex if gjk.test() else []

def epa(shapeA: RigidBody, shapeB: RigidBody) -> Collision | None:
    gjk = GJK(shapeA, shapeB)
    return gjk.intersection()

class PhysicsWorld:
    def __init__(self):
        self.bodies = []
        self.gravity = Vector2([0., 9.81])
        self.pixels_per_meter = _get_pixels_per_meter()

    def resolve_collision(self, collision: Collision):
        # Skip if both bodies are static
        if collision.shapeA.is_static and collision.shapeB.is_static:
            return

        # Calculate separation vector - flip the normal direction
        separation = collision.normal * -collision.overlap

        # If both bodies can move, split the separation between them
        if not collision.shapeA.is_static and not collision.shapeB.is_static:
            collision.shapeA.position += separation * 0.5
            collision.shapeB.position -= separation * 0.5
        # If only one body can move, move it the full separation distance
        elif not collision.shapeA.is_static:
            collision.shapeA.position += separation
        elif not collision.shapeB.is_static:
            collision.shapeB.position -= separation

    def step(self, dt: float):
        # Apply forces and update positions
        for body in self.bodies:
            if not body.is_static:
                # body.apply_force(self.gravity * body.mass)
                body.update(dt * self.pixels_per_meter)
        
        # Check and resolve collisions
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):  # Changed to avoid duplicate checks
                collision = epa(self.bodies[i], self.bodies[j])
                if collision:
                    self.resolve_collision(collision)
    
    def draw(self):
        for body in self.bodies:
            body.draw()

r.init_window(1280, 800, "i love physics i am a physics engine")

world = PhysicsWorld()

circle = CircleBody(Vector2([150., 350.]), 10., is_static=False)
rectangle = RectangleBody(Vector2([100., 300.]), 50., 50., is_static=True)
pentagon = PolygonBody([Vector2([200., 300.]),
                        Vector2([235., 330.]),
                        Vector2([220., 370.]),
                        Vector2([180., 370.]),
                        Vector2([165., 330.])], is_static=True)

world.bodies.append(circle)
world.bodies.append(rectangle)
world.bodies.append(pentagon)

while not r.window_should_close():
    r.begin_drawing()
    r.clear_background(r.RAYWHITE)
    
    mxy = r.get_mouse_position()
    circle.position = Vector2([mxy.x, mxy.y])
    
    world.step(r.get_frame_time())
    world.draw()
 
    r.end_drawing()

r.close_window()