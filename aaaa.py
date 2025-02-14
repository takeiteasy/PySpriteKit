import numpy as np

def support(shape1, shape2, direction):
    """Finds the support point in the Minkowski difference."""
    return shape1.get_farthest_point_in_direction(direction) - shape2.get_farthest_point_in_direction(-direction)

def cross_2d(a, b):
    """Compute the 2D cross product which returns a scalar."""
    return np.cross(np.append(a, 0), np.append(b, 0))[2]

def closest_point_to_origin(simplex):
    """Finds the closest point on the simplex to the origin."""
    if len(simplex) == 1:
        return simplex[0]
    elif len(simplex) == 2:
        a, b = simplex
        ab = b - a
        ao = -a
        if np.dot(ab, ao) <= 0:
            return a
        else:
            t = np.dot(ab, ao) / np.dot(ab, ab)
            if t >= 1:
                return b
            return a + t * ab
    elif len(simplex) == 3:
        a, b, c = simplex
        ab = b - a
        ac = c - a
        ao = -a
        
        # Use our custom 2D cross product function
        abc = cross_2d(ab, ac)  # Normal direction (scalar)
        
        # Check Voronoi regions using 2D cross products
        if abc * cross_2d(ac, ao) > 0:
            # Origin is closest to AC edge
            return closest_point_to_origin([a, c])
        elif abc * cross_2d(ab, ao) < 0:
            # Origin is closest to AB edge
            return closest_point_to_origin([a, b])
        else:
            # Origin is closest to point A
            return a
    return None

def contains_origin(simplex):
    """Checks if the simplex contains the origin."""
    if len(simplex) == 2:
        a, b = simplex
        ab = b - a
        ao = -a
        return float(np.dot(ab, ao)) > 0
    elif len(simplex) == 3:
        a, b, c = simplex
        ab = b - a
        ac = c - a
        # Use our custom 2D cross product function
        abc = cross_2d(ab, ac)
        return abc * cross_2d(ab, -a) > 0 and \
               abc * cross_2d(ac, -a) > 0 and \
               abc * cross_2d(b - c, -c) > 0
    return False

def normalize(v):
    """Safely normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:  # Avoid division by zero
        return v
    return v / norm

def gjk(shape1, shape2):
    """GJK algorithm to detect intersection between two convex shapes."""
    # Initial direction
    direction = np.array([1., 0.])
    # Initial simplex
    simplex = [support(shape1, shape2, direction)]
    direction = -simplex[0]
    
    while True:
        new_point = support(shape1, shape2, direction)
        if np.dot(new_point, direction) < 0:
            return False
        simplex.append(new_point)
        if contains_origin(simplex):
            return True
        direction = closest_point_to_origin(simplex)
        if np.linalg.norm(direction) < 1e-10:
            return True

class ShapeType:
    pass

class ConvexShape(ShapeType):
    def __init__(self, points):
        self.points = np.array(points)
    
    def get_farthest_point_in_direction(self, direction):
        return self.points[np.argmax(np.dot(self.points, direction))]

class Circle(ConvexShape):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def get_farthest_point_in_direction(self, direction):
        return self.center + self.radius * normalize(direction)

class Rectangle(ConvexShape):
    def __init__(self, center, width, height, rotation=0, scale=1.):
        self.center = np.array(center)
        self.width = width
        self.height = height
        self.rotation = rotation
        self.scale = scale
        self._corners = None 
        self._dirty = True 
    
    def __setattr__(self, key, value):
        if key in ["rotation", "scale", "width", "height", "center"]:
            self._dirty = True
        super().__setattr__(key, value)
    
    def get_transformed_corners(self):
        """
        Calculate the transformed corners of the rectangle after applying rotation and scaling.
        """
        # Define the local corners of the rectangle (before rotation and scaling)
        half_width = (self.width / 2) * self.scale
        half_height = (self.height / 2) * self.scale
        local_corners = [
            np.array([half_width, half_height]),
            np.array([-half_width, half_height]),
            np.array([half_width, -half_height]),
            np.array([-half_width, -half_height])
        ]

        # Create a rotation matrix
        cos_theta = np.cos(self.rotation)
        sin_theta = np.sin(self.rotation)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # Apply rotation and translation to each corner
        transformed_corners = [self.center + rotation_matrix @ corner for corner in local_corners]
        return transformed_corners

    def get_farthest_point_in_direction(self, direction):
        """
        Find the farthest point of the rectangle in the given direction after applying transformations.
        """
        # Get the transformed corners
        if not self._corners or self._dirty:
            self._corners = self.get_transformed_corners()
            self._dirty = False
        return self._corners[np.argmax(np.dot(self._corners, direction))]

class Triangle(ConvexShape):
    def __init__(self, a, b, c):
        self.points = np.array([a, b, c])
    
    def get_farthest_point_in_direction(self, direction):
        return self.points[np.argmax(np.dot(self.points, direction))]

class LineSegment(ConvexShape):
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
    
    def get_farthest_point_in_direction(self, direction):
        return self.start if np.dot(self.start, direction) > np.dot(self.end, direction) else self.end

class Capsule(ConvexShape):
    def __init__(self, start, end, radius):
        self.start = np.array(start)
        self.end = np.array(end)
        self.radius = radius
    
    def get_farthest_point_in_direction(self, direction):
        line_segment_point = self.start if np.dot(self.start, direction) > np.dot(self.end, direction) else self.end
        direction_norm = direction / np.linalg.norm(direction)
        return line_segment_point + self.radius * direction_norm

class RigidBody:
    def __init__(self, shape, mass, position, velocity=np.array([0., 0.]), angle=0., angular_velocity=0.):
        """
        Initialize a rigid body.
        
        :param shape: The shape of the rigid body (e.g., Circle, Rectangle, Polygon).
        :param mass: The mass of the rigid body.
        :param position: The initial position of the rigid body as a 2D vector.
        :param velocity: The initial velocity of the rigid body as a 2D vector (default is [0, 0]).
        :param angle: The initial angle of the rigid body in radians (default is 0).
        :param angular_velocity: The initial angular velocity of the rigid body (default is 0).
        """
        self.shape = shape
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.force = np.array([0., 0.])  # Net force acting on the body
        self.torque = 0  # Net torque acting on the body

    def apply_force(self, force, point=np.array([0., 0.])):
        """
        Apply a force to the rigid body at a specific point.
        
        :param force: The force to apply as a 2D vector.
        :param point: The point at which the force is applied, relative to the body's center (default is [0, 0]).
        """
        self.force += force
        self.torque += np.cross(point, force)  # Torque = r x F

    def integrate(self, dt):
        """
        Update the rigid body's position, velocity, angle, and angular velocity over a time step.
        
        :param dt: The time step for integration.
        """
        # Update velocity and position using Newton's second law (F = ma)
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Update angular velocity and angle using torque
        angular_acceleration = self.torque / self.mass
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

        # Reset forces and torque
        self.force = np.array([0., 0.])
        self.torque = 0

        # Update the shape's position and rotation
        if hasattr(self.shape, 'center'):
            self.shape.center = self.position
        if hasattr(self.shape, 'rotation'):
            self.shape.rotation = self.angle

    def get_shape(self):
        """
        Get the shape of the rigid body.
        """
        return self.shape

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.direction = self.direction / np.linalg.norm(self.direction)
    
    def intersect_convex_shape(self, polygon):
        """
        Check if the ray intersects a polygon.
        :param polygon: A list of vertices defining the polygon.
        :return: True if the ray intersects the polygon, False otherwise.
        """
        vertices = polygon.points
        intersections = 0
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            if self.intersect_segment(p1, p2):
                intersections += 1
        return intersections % 2 == 1
    
    def intersect_circle(self, circle):
        """
        Check if the ray intersects a circle.
        :param circle: A Circle object with center and radius.
        :return: True if the ray intersects the circle, False otherwise.
        """
        # Vector from ray origin to circle center
        oc = circle.center - self.origin
        # Projection of oc onto the ray direction
        projection = np.dot(oc, self.direction)
        # Closest point on the ray to the circle center
        closest_point = self.origin + projection * self.direction
        # Distance from closest point to circle center
        distance = np.linalg.norm(closest_point - circle.center)
        return distance <= circle.radius
    
    def intersect_rectangle(self, rectangle):
        """
        Check if the ray intersects a rectangle.
        :param rectangle: A Rectangle object with center, width, height, and rotation.
        :return: True if the ray intersects the rectangle, False otherwise.
        """
        # Get the transformed corners of the rectangle
        corners = rectangle.get_transformed_corners()
        # Define the edges of the rectangle
        edges = [
            (corners[0], corners[1]),
            (corners[1], corners[3]),
            (corners[3], corners[2]),
            (corners[2], corners[0])
        ]
        # Check intersection with each edge
        for edge in edges:
            if self.intersect_segment(edge[0], edge[1]):
                return True
        return False
    
    def intersect_line_segment(self, p1, p2):
        """
        Check if the ray intersects a line segment.
        :param p1: First endpoint of the segment.
        :param p2: Second endpoint of the segment.
        :return: True if the ray intersects the segment, False otherwise.
        """
        # Ray: origin + t * direction
        # Segment: p1 + u * (p2 - p1)
        segment_dir = p2 - p1
        denominator = np.cross(self.direction, segment_dir)
        if abs(denominator) < 1e-10:
            return False  # Parallel
        t = np.cross(p1 - self.origin, segment_dir) / denominator
        u = np.cross(p1 - self.origin, self.direction) / denominator
        return t >= 0 and 0 <= u <= 1
    
    def intersect_triangle(self, triangle):
        """
        Check if the ray intersects a triangle using the Möller-Trumbore algorithm.
        :param triangle: A Triangle object with vertices a, b, c.
        :return: True if the ray intersects the triangle, False otherwise.
        """
        a, b, c = triangle.points
        edge1 = b - a
        edge2 = c - a
        h = np.cross(self.direction, edge2)
        det = np.dot(edge1, h)
        if abs(det) < 1e-10:
            return False  # Ray is parallel to the triangle
        inv_det = 1.0 / det
        s = self.origin - a
        u = inv_det * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return False
        q = np.cross(s, edge1)
        v = inv_det * np.dot(self.direction, q)
        if v < 0.0 or u + v > 1.0:
            return False
        t = inv_det * np.dot(edge2, q)
        return t >= 0.0  # Intersection is in front of the ray origin
    
    def intersect_capsule(self, capsule):
        """
        Check if the ray intersects a capsule.
        :param capsule: A Capsule object with start, end, and radius.
        :return: True if the ray intersects the capsule, False otherwise.
        """
        # Vector from ray origin to capsule start
        start_to_origin = self.origin - capsule.start
        # Vector from capsule start to end
        segment_dir = capsule.end - capsule.start
        segment_length = np.linalg.norm(segment_dir)
        segment_dir = segment_dir / segment_length  # Normalize

        # Project start_to_origin onto the segment direction
        projection = np.dot(start_to_origin, segment_dir)
        # Clamp the projection to the segment
        projection = np.clip(projection, 0, segment_length)
        # Closest point on the segment to the ray origin
        closest_point = capsule.start + projection * segment_dir

        # Vector from closest point to ray origin
        closest_to_origin = self.origin - closest_point
        # Distance from closest point to ray origin
        distance = np.linalg.norm(closest_to_origin)

        # Check if the distance is within the capsule's radius
        if distance > capsule.radius:
            return False

        # Check if the ray intersects the infinite cylinder around the segment
        # Ray direction cross segment direction
        ray_cross_segment = np.cross(self.direction, segment_dir)
        # Distance between the ray and the segment
        distance_squared = np.dot(closest_to_origin, ray_cross_segment) ** 2 / np.linalg.norm(ray_cross_segment) ** 2
        return distance_squared <= capsule.radius ** 2

    def intersect(self, shape):
        if isinstance(shape, ConvexShape):
            return self.intersect_convex_shape(shape)
        elif isinstance(shape, Circle):
            return self.intersect_circle(shape)
        elif isinstance(shape, Rectangle):
            return self.intersect_rectangle(shape)
        elif isinstance(shape, Triangle):
            return self.intersect_triangle(shape)
        elif isinstance(shape, LineSegment):
            return self.intersect_line_segment(shape.start, shape.end)
        elif isinstance(shape, Capsule):
            return self.intersect_capsule(shape)
        else:
            raise ValueError(f"Unsupported shape type: {type(shape)}")

def epa(simplex, shape1, shape2, tolerance=1e-6, max_iterations=100):
    """
    Expanding Polytope Algorithm (EPA) to find the penetration depth and contact normal.
    
    :param simplex: The simplex from GJK that contains the origin.
    :param shape1: The first shape.
    :param shape2: The second shape.
    :param tolerance: The tolerance for convergence (default is 1e-6).
    :param max_iterations: The maximum number of iterations (default is 100).
    :return: A tuple (penetration_depth, contact_normal, contact_point).
    """
    polytope = simplex  # Initialize the polytope with the simplex from GJK
    for _ in range(max_iterations):
        # Find the closest face to the origin
        face_index, normal, distance = find_closest_face(polytope)
        # Get the support point in the direction of the normal
        support_point = support(shape1, shape2, normal)
        # Check if the support point is close enough to the face
        if np.dot(support_point, normal) - distance < tolerance:
            # Return the penetration depth and contact normal
            return distance, normal, support_point
        # Add the support point to the polytope
        polytope.insert(face_index + 1, support_point)
    return None  # EPA did not converge

def find_closest_face(polytope):
    """
    Find the closest face of the polytope to the origin.
    
    :param polytope: The polytope (list of vertices).
    :return: A tuple (face_index, normal, distance).
    """
    closest_distance = float('inf')
    closest_face_index = 0
    closest_normal = np.array([0, 0])
    for i in range(len(polytope)):
        a = polytope[i]
        b = polytope[(i + 1) % len(polytope)]
        ab = b - a
        ao = -a
        normal = np.array([-ab[1], ab[0]])  # Perpendicular to AB
        normal = normal / np.linalg.norm(normal)  # Normalize
        distance = np.dot(normal, a)
        if distance < closest_distance:
            closest_distance = distance
            closest_face_index = i
            closest_normal = normal
    return closest_face_index, closest_normal, closest_distance

# Helper function to get the simplex from GJK
def get_simplex_from_gjk(shape1, shape2):
    """
    Get the simplex from GJK (for demonstration purposes).
    """
    # This is a placeholder. In practice, you would store the simplex during GJK execution.
    return [support(shape1, shape2, np.array([1, 0])),
            support(shape1, shape2, np.array([-1, 0])),
            support(shape1, shape2, np.array([0, 1]))]

class Simulator:
    def __init__(self):
        self.rigid_bodies = []
        self.gravity = np.array([0, -9.8])

    def add_rigid_body(self, rigid_body):
        self.rigid_bodies.append(rigid_body)

    def set_gravity(self, gravity):
        self.gravity = np.array(gravity)

    def update(self, dt):
        # Apply gravity to all rigid bodies
        for body in self.rigid_bodies:
            body.apply_force(self.gravity * body.mass)

        # Integrate all rigid bodies
        for body in self.rigid_bodies:
            body.integrate(dt)

        # Check for collisions between all pairs of rigid bodies
        self.handle_collisions()

    def handle_collisions(self):
        for i in range(len(self.rigid_bodies)):
            for j in range(i + 1, len(self.rigid_bodies)):
                body1 = self.rigid_bodies[i]
                body2 = self.rigid_bodies[j]
                if gjk(body1.get_shape(), body2.get_shape()):
                    print(f"Collision detected between body {i} and body {j}")
                    # Use EPA to resolve the collision
                    simplex = get_simplex_from_gjk(body1.get_shape(), body2.get_shape())  # Simplex from GJK
                    result = epa(simplex, body1.get_shape(), body2.get_shape())
                    if result:
                        penetration_depth, contact_normal, contact_point = result
                        self.resolve_collision(body1, body2, contact_normal, penetration_depth)

    def resolve_collision(self, body1, body2, contact_normal, penetration_depth):
        """
        Resolve a collision using EPA results.
        
        :param body1: The first rigid body.
        :param body2: The second rigid body.
        :param contact_normal: The contact normal (direction of collision).
        :param penetration_depth: The penetration depth.
        """
        # Relative velocity
        relative_velocity = body2.velocity - body1.velocity
        velocity_along_normal = np.dot(relative_velocity, contact_normal)

        # Do not resolve if objects are separating
        if velocity_along_normal > 0:
            return

        # Calculate restitution (bounciness)
        e = 0.5  # Coefficient of restitution

        # Calculate impulse scalar
        j = -(1 + e) * velocity_along_normal
        j /= (1 / body1.mass) + (1 / body2.mass)

        # Apply impulse
        impulse = j * contact_normal
        body1.velocity -= impulse / body1.mass
        body2.velocity += impulse / body2.mass

        # Positional correction to prevent sinking
        correction = penetration_depth / (1 / body1.mass + 1 / body2.mass) * contact_normal
        body1.position -= correction / body1.mass
        body2.position += correction / body2.mass

# Create two circles
circle1 = Circle(center=[0., 0.], radius=1)
circle2 = Circle(center=[3., 0.], radius=1)
rigid_body1 = RigidBody(shape=circle1, mass=1, position=[0., 0.], velocity=[1., 0.])
rigid_body2 = RigidBody(shape=circle2, mass=1, position=[3., 0.])

# Create a simulator
simulator = Simulator()
simulator.add_rigid_body(rigid_body1)
simulator.add_rigid_body(rigid_body2)

# Simulate for multiple time steps
dt = 0.1
for step in range(20):
    print(f"Step {step + 1}")
    simulator.update(dt)
    print("Body 1 position:", rigid_body1.position)
    print("Body 2 position:", rigid_body2.position)
    print()