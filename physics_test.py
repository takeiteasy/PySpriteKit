from slimrr import *
import pyray as r
from enum import Enum
import math

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
        self.mass = float('inf') if is_static else mass
        self.inv_mass = 0. if is_static else 1. / mass
        self.is_static = is_static
        self.restitution = 0.5
        self.friction = 0.2
        self.damping = 0.98
        self.angular_damping = 0.98  # Separate damping for rotation
        
        # Angular properties
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.torque = 0.0
        self.inertia = self._calculate_inertia()
        self.inv_inertia = 0. if is_static else 1. / self.inertia
    
    def _calculate_inertia(self) -> float:
        # Base calculation - override in subclasses
        # For point masses, use small fixed inertia
        return 0.01 * self.mass
    
    @property
    def vertices(self) -> list[Vector2]:
        # Transform vertices by both position and rotation
        rotated = []
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        for v in self._vertices:
            # Rotate vertex
            rx = v.x * cos_a - v.y * sin_a
            ry = v.x * sin_a + v.y * cos_a
            # Translate to position
            rotated.append(Vector2([rx + self.position.x, ry + self.position.y]))
        return rotated

    def apply_force(self, force: Vector2, point: Vector2 = None):
        self.force += force
        if point is not None:
            # Calculate torque from force applied at point
            r = point - self.position
            self.torque += r.x * force.y - r.y * force.x

    def update(self, dt: float):
        if not self.is_static:
            # Linear motion
            self.velocity += self.force * (dt * self.inv_mass)
            self.velocity *= self.damping
            self.position += self.velocity * dt
            self.force = Vector2([0., 0.])
            
            # Angular motion
            self.angular_velocity += self.torque * (dt * self.inv_inertia)
            self.angular_velocity *= self.angular_damping
            self.angle += self.angular_velocity * dt
            self.torque = 0.0

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
    def __init__(self, vertices: list[Vector2], mass: float = 1., is_static: bool=False):
        super().__init__(vertices=vertices, mass=mass, is_static=is_static)

class PointBody(RigidBody):
    def __init__(self, position: Vector2, mass: float = 1.):
        super().__init__(vertices=[position], mass=mass, is_static=True)
    
    def support(self, direction: Vector2) -> Vector2:
        return self.position
    
    def draw(self):
        r.draw_circle(int(self.position.x), int(self.position.y), 1, r.RED)

class LineBody(RigidBody):
    def __init__(self, start: Vector2, end: Vector2, mass: float = 1.):
        super().__init__(vertices=[start, end], mass=mass, is_static=True)
        self.position = (start + end) / 2.
    
    def support(self, direction: Vector2) -> Vector2:
        return self.start if self.start.dot(direction) > self.end.dot(direction) else self.end
    
    def draw(self):
        r.draw_line(int(self.vertices[0].x), int(self.vertices[0].y), int(self.vertices[1].x), int(self.vertices[1].y), r.RED)

class RectangleBody(RigidBody):
    def __init__(self, position: Vector2, width: float, height: float, mass: float=1., is_static: bool=False):
        self.width = width
        self.height = height
        super().__init__(vertices=[position,
                                   position + Vector2([width, 0.]),
                                   position + Vector2([width, height]),
                                   position + Vector2([0., height])],
                         mass=mass,
                         is_static=is_static)
    
    def _calculate_inertia(self) -> float:
        # Moment of inertia for a rectangle: I = (1/12) * m * (w^2 + h^2)
        return (1.0/12.0) * self.mass * (self.width * self.width + self.height * self.height)

class CircleBody(RigidBody):
    def __init__(self, position: Vector2, radius: float, mass: float=1., is_static: bool=False):
        self.radius = radius
        super().__init__(vertices=[position], mass=mass, is_static=is_static)
        self.position = position
    
    def _calculate_inertia(self) -> float:
        # Moment of inertia for a solid circle: I = (1/2) * m * r^2
        return 0.5 * self.mass * self.radius * self.radius

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
        self.restitution = 0.3  # Reduced restitution for less bouncy collisions

class GJK:
    MAX_ITERATIONS = 20
    MAX_INTERSECTIONS = 32
    EPSILON = 1e-10

    def __init__(self, shapeA: RigidBody, shapeB: RigidBody):
        self.shapeA = shapeA
        self.shapeB = shapeB
        self.direction = Vector2([0., 0.])
        self.simplex = []

    def _support(self, direction: Vector2) -> Vector2:
        return self.shapeA.support(direction) - self.shapeB.support(direction * -1.)
    
    def _evolve(self) -> SimplexEvolution:
        match len(self.simplex):
            case 0:
                self.direction = self.shapeB.position - self.shapeA.position
                if self.direction.squared_length < GJK.EPSILON:
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
        vertex = self._support(self.direction)
        self.simplex.append(vertex)
        return SimplexEvolution.STILL_EVOLVING if self.direction.dot(vertex) > -GJK.EPSILON else SimplexEvolution.NO_INTERSECTION

    def test(self):
        for _ in range(GJK.MAX_ITERATIONS):
            match self._evolve():
                case SimplexEvolution.FOUND_INTERSECTION:
                    return True
                case SimplexEvolution.NO_INTERSECTION:
                    return False
                case SimplexEvolution.STILL_EVOLVING:
                    pass
        return False
    
    def _closest_edge(self, winding: PolygonWinding) -> Edge:
        closest_distance = float('inf')
        closest_normal = Vector2([0., 0.])
        closest_index = -1
        for i in range(len(self.simplex)):
            j = (i + 1) % len(self.simplex)
            line = self.simplex[j] - self.simplex[i]
            # Skip if line is too short (points are too close together)
            if line.squared_length < GJK.EPSILON:
                continue
            # Calculate normal and normalize it safely
            normal = Vector2([line.y, -line.x]) if winding == PolygonWinding.CLOCKWISE else Vector2([-line.y, line.x])
            length = (normal.x * normal.x + normal.y * normal.y) ** 0.5
            if length > GJK.EPSILON:  # Only normalize if length is not too close to zero
                normal = normal / length
                dist = normal.dot(self.simplex[i])
                if dist < closest_distance:
                    closest_distance = dist
                    closest_normal = normal
                    closest_index = j
        return Edge(closest_distance, closest_normal, closest_index)

    def intersection(self) -> Collision | None:
        if not self.test():
            return None
        # Get winding direction
        e0 = (self.simplex[1].x - self.simplex[0].x) * (self.simplex[1].y + self.simplex[0].y)
        e1 = (self.simplex[2].x - self.simplex[1].x) * (self.simplex[2].y + self.simplex[1].y)
        e2 = (self.simplex[0].x - self.simplex[2].x) * (self.simplex[0].y + self.simplex[2].y)
        winding = PolygonWinding.CLOCKWISE if e0 + e1 + e2 >= 0. else PolygonWinding.ANTICLOCKWISE
        # EPA loop
        min_distance = float('inf')
        min_normal = Vector2([0., 0.])
        min_support = None
        for _ in range(GJK.MAX_INTERSECTIONS):
            edge = self._closest_edge(winding)
            support = self._support(edge.normal)
            distance = edge.normal.dot(support)
            if distance < min_distance:
                min_distance = distance
                min_normal = edge.normal
                min_support = support
            if abs(distance - edge.distance) <= GJK.EPSILON:
                # Calculate direction from A to B
                direction = self.shapeB.position - self.shapeA.position
                # Ensure normal points from A to B
                if min_normal.dot(direction) < 0:
                    min_normal *= -1
                    min_distance *= -1
                return Collision(self.shapeA, self.shapeB, min_support, min_normal, abs(min_distance))
            self.simplex.insert(edge.index, support)
        # If we reach here, use the best result found
        if min_support is not None:
            direction = self.shapeB.position - self.shapeA.position
            if min_normal.dot(direction) < 0:
                min_normal *= -1
                min_distance *= -1
            return Collision(self.shapeA, self.shapeB, min_support, min_normal, abs(min_distance))
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
        self.gravity = Vector2([0., 981.0])
        self.pixels_per_meter = _get_pixels_per_meter()
        self.position_correction_factor = 0.8    # More aggressive correction
        self.penetration_allowance = 0.001      # Smaller penetration allowance
        self.velocity_threshold = 0.1
        self.angular_velocity_threshold = 0.01
        self.resting_threshold = 0.5
        self.collision_iterations = 4           # Multiple iterations for stability

    def _resolve_collision(self, collision: Collision):
        bodyA = collision.shapeA
        bodyB = collision.shapeB
        
        if bodyA.is_static and bodyB.is_static:
            return

        # Position correction first - more aggressive for floor collisions
        correction = max(collision.overlap - self.penetration_allowance, 0.0)
        correction *= self.position_correction_factor
        
        # More aggressive correction if one body is static (like the floor)
        if bodyA.is_static or bodyB.is_static:
            correction *= 1.5
            
        correction_vector = collision.normal * correction
        
        if not bodyA.is_static:
            bodyA.position -= correction_vector * bodyA.inv_mass
        if not bodyB.is_static:
            bodyB.position += correction_vector * bodyB.inv_mass

        # Calculate collision point relative to each body's center
        ra = collision.intersection - bodyA.position
        rb = collision.intersection - bodyB.position

        # Calculate relative velocity at collision point
        va = bodyA.velocity + Vector2([-ra.y, ra.x]) * bodyA.angular_velocity
        vb = bodyB.velocity + Vector2([-rb.y, rb.x]) * bodyB.angular_velocity
        rel_velocity = vb - va
        
        vel_along_normal = rel_velocity.dot(collision.normal)
        
        # Early out if separating
        if vel_along_normal > 0:
            return

        # Check for resting contact
        is_resting = abs(vel_along_normal) < self.resting_threshold
        
        # Calculate restitution - reduce for floor collisions
        restitution = 0.0 if is_resting else min(bodyA.restitution, bodyB.restitution)
        if bodyA.is_static or bodyB.is_static:
            restitution *= 0.8  # Reduce bouncing off static objects
        
        # Calculate impulse
        raCrossN = ra.x * collision.normal.y - ra.y * collision.normal.x
        rbCrossN = rb.x * collision.normal.y - rb.y * collision.normal.x
        inv_mass_sum = bodyA.inv_mass + bodyB.inv_mass + \
                      raCrossN * raCrossN * bodyA.inv_inertia + \
                      rbCrossN * rbCrossN * bodyB.inv_inertia

        if inv_mass_sum <= 0:
            return
            
        j = -(1.0 + restitution) * vel_along_normal / inv_mass_sum
        
        # Apply impulse
        impulse = collision.normal * j
        
        if not bodyA.is_static:
            bodyA.velocity -= impulse * bodyA.inv_mass
            bodyA.angular_velocity -= raCrossN * j * bodyA.inv_inertia
        
        if not bodyB.is_static:
            bodyB.velocity += impulse * bodyB.inv_mass
            bodyB.angular_velocity += rbCrossN * j * bodyB.inv_inertia

        # Friction
        tangent = rel_velocity - (collision.normal * vel_along_normal)
        if tangent.squared_length > 1e-6:
            tangent = tangent.normalised
            friction = min(bodyA.friction, bodyB.friction)
            
            # Increase friction for floor collisions
            if bodyA.is_static or bodyB.is_static:
                friction *= 1.2
            
            jt = -rel_velocity.dot(tangent)
            jt /= inv_mass_sum
            
            if not is_resting:
                if abs(jt) > j * friction:
                    jt = j * friction if jt > 0 else -j * friction
            else:
                jt *= 0.4  # More friction for resting contacts
            
            friction_impulse = tangent * jt
            
            if not bodyA.is_static:
                bodyA.velocity -= friction_impulse * bodyA.inv_mass
                bodyA.angular_velocity -= (ra.x * friction_impulse.y - ra.y * friction_impulse.x) * bodyA.inv_inertia
            
            if not bodyB.is_static:
                bodyB.velocity += friction_impulse * bodyB.inv_mass
                bodyB.angular_velocity += (rb.x * friction_impulse.y - rb.y * friction_impulse.x) * bodyB.inv_inertia

    def resolve(self):
        # Multiple iterations of collision resolution
        for _ in range(self.collision_iterations):
            # Sort collisions by overlap for more stable resolution
            collisions = []
            for i in range(len(self.bodies)):
                for j in range(i + 1, len(self.bodies)):
                    collision = epa(self.bodies[i], self.bodies[j])
                    if collision:
                        collisions.append(collision)
            
            # Sort by overlap (resolve biggest overlaps first)
            collisions.sort(key=lambda c: c.overlap, reverse=True)
            
            # Resolve all collisions
            for collision in collisions:
                self._resolve_collision(collision)

    def update(self, dt: float):
        dt = min(dt, 1.0/60.0)
        
        for body in self.bodies:
            if not body.is_static:
                # Apply gravity
                body.apply_force(self.gravity * body.mass)
                
                # Update velocities and position
                body.update(dt)
                
                # Stop very small movements, but handle vertical and horizontal separately
                if abs(body.velocity.y) < self.velocity_threshold:
                    body.velocity.y = 0
                if abs(body.velocity.x) < self.velocity_threshold:
                    body.velocity.x = 0
                
                # Stop very small rotations
                if abs(body.angular_velocity) < self.angular_velocity_threshold:
                    body.angular_velocity = 0.0

    def draw(self):
        for body in self.bodies:
            body.draw()
    
    def step(self, dt: float):
        self.update(dt)
        self.resolve()
        self.draw()

r.init_window(1280, 800, "i hate physics i am a fucked up physics engine")

world = PhysicsWorld()

# Create floor
floor = RectangleBody(Vector2([0., 750.]), 1280., 50., is_static=True)
floor.friction = 0.8     # Higher friction for floor
floor.restitution = 0.2  # Less bouncy floor

# Create falling objects
circle = CircleBody(Vector2([150., 50.]), 10., mass=1.0)
circle.restitution = 0.4
circle.friction = 0.4
circle.damping = 0.995
circle.angular_damping = 0.995
circle.angular_velocity = 10.0

# Create a rectangle that will fall and rotate
falling_rect = RectangleBody(Vector2([300., 50.]), 40., 20., mass=1.0)
falling_rect.restitution = 0.4
falling_rect.friction = 0.4
falling_rect.damping = 0.995
falling_rect.angular_damping = 0.995
falling_rect.angle = 0.5
falling_rect.angular_velocity = -5.0

# Static rectangle obstacle
rectangle = RectangleBody(Vector2([100., 300.]), 50., 50., is_static=True)
rectangle.restitution = 0.4
rectangle.friction = 0.4

# Static pentagon obstacle
pentagon = PolygonBody([Vector2([200., 300.]),
                       Vector2([235., 330.]),
                       Vector2([220., 370.]),
                       Vector2([180., 370.]),
                       Vector2([165., 330.])],
                      is_static=True)
pentagon.restitution = 0.4
pentagon.friction = 0.4

# Add all bodies to the world
world.bodies.extend([floor, circle, falling_rect, rectangle, pentagon])

while not r.window_should_close():
    r.begin_drawing()
    r.clear_background(r.RAYWHITE)
    
    world.step(r.get_frame_time())
    
    # Debug drawing - show rotation
    for body in world.bodies:
        if isinstance(body, CircleBody):
            # Draw a line from center to edge to show rotation
            end = Vector2([
                body.position.x + body.radius * math.cos(body.angle),
                body.position.y + body.radius * math.sin(body.angle)
            ])
            r.draw_line(int(body.position.x), int(body.position.y),
                       int(end.x), int(end.y), r.BLUE)
 
    r.end_drawing()

r.close_window()