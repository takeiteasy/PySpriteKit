import numpy as np
import inspect
from functools import wraps
from numbers import Number
from multipledispatch import dispatch

def all_parameters_as_numpy_arrays(fn):
    """Converts all of a function's arguments to numpy arrays.

    Used as a decorator to reduce duplicate code.
    """
    # wraps allows us to pass the docstring back
    # or the decorator will hide the function from our doc generator
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if v is not None:
                args[i] = np.asarray(v)
        for k,v in kwargs.items():
            if v is not None:
                kwargs[k] = np.asarray(v)
        return fn(*args, **kwargs)
    return wrapper

def parameters_as_numpy_arrays(*args_to_convert):
    """Converts specific arguments to numpy arrays.

    Used as a decorator to reduce duplicate code.

    Arguments are specified by their argument name.
    For example
    ::

        @parameters_as_numpy_arrays('a', 'b', 'optional')
        def myfunc(a, b, *args, **kwargs):
            pass

        myfunc(1, [2,2], optional=[3,3,3])
    """
    def decorator(fn):
        # wraps allows us to pass the docstring back
        # or the decorator will hide the function from our doc generator

        try:
            getfullargspec = inspect.getfullargspec
        except AttributeError:
            getfullargspec = inspect.getargspec

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # get the arguments of the function we're decorating
            fn_args = getfullargspec(fn)

            # convert any values that are specified
            # if the argument isn't in our list, just pass it through

            # convert the *args list
            # we zip the args with the argument names we received from
            # the inspect function
            args = list(args)
            for i, (k, v) in enumerate(zip(fn_args.args, args)):
                if k in args_to_convert and v is not None:
                    args[i] = np.array(v)

            # convert the **kwargs dict
            for k,v in kwargs.items():
                if k in args_to_convert and v is not None:
                    kwargs[k] = np.array(v)

            # pass the converted values to our function
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def solve_quadratic_equation(a, b, c):
    """Quadratic equation solver.
    Solve function of form f(x) = ax^2 + bx + c

    :param float a: Quadratic part of equation.
    :param float b: Linear part of equation.
    :param float c: Static part of equation.
    :rtype: list
    :return: List contains either two elements for two solutions, one element for one solution, or is empty if
        no solution for the quadratic equation exists.
    """
    delta = b * b - 4 * a * c
    if delta > 0:
        # Two solutions
        # See https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
        # Why not use simple form:
        # s1 = (-b + math.sqrt(delta)) / (2 * a)
        # s2 = (-b - math.sqrt(delta)) / (2 * a)
        q = -0.5 * (b + np.math.sqrt(delta)) if b > 0 else -0.5 * (b - np.math.sqrt(delta))
        s1 = q / a
        s2 = c / q
        return [s1, s2]
    elif delta == 0:
        # One solution
        return [-b / (2 * a)]
    else:
        # No solution exists
        return list()

class NpProxy(object):
    def __init__(self, index):
        self._index = index

    def __get__(self, obj, cls):
        return obj[self._index]

    def __set__(self, obj, value):
        obj[self._index] = value

class BaseObject(np.ndarray):
    _shape = None

    def __new__(cls, obj):
        # ensure the object matches the required shape
        obj.shape = cls._shape
        return obj

    def _unsupported_type(self, method, other):
        raise ValueError('Cannot {} a {} to a {}'.format(method, type(other).__name__, type(self).__name__))

    ########################
    # Redirect assignment operators
    def __iadd__(self, other):
        self[:] = self.__add__(other)
        return self

    def __isub__(self, other):
        self[:] = self.__sub__(other)
        return self

    def __imul__(self, other):
        self[:] = self.__mul__(other)
        return self

    def __idiv__(self, other):
        self[:] = self.__div__(other)
        return self


@all_parameters_as_numpy_arrays
def _normalise(vec):    # TODO: mark as deprecated
    """normalizes an Nd list of vectors or a single vector
    to unit length.

    The vector is **not** changed in place.

    For zero-length vectors, the result will be np.nan.

    :param numpy.array vec: an Nd array with the final dimension
        being vectors
        ::

            numpy.array([ x, y, z ])

        Or an NxM array::

            numpy.array([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).

    :rtype: numpy.array
    :return: The normalized vector/s
    """
    # calculate the length
    # this is a duplicate of length(vec) because we
    # always want an array, even a 0-d array.
    return (vec.T  / np.sqrt(np.sum(vec**2,axis=-1))).T



@all_parameters_as_numpy_arrays
def _squared_length(vec):
    """Calculates the squared length of a vector.

    Useful when trying to avoid the performance
    penalty of a square root operation.

    :param numpy.array vec: An Nd numpy.array.
    :rtype: np.array
    :return: The squared length of vec, if a 1D vector is input, the
    result will be a scalar. If a Matrix is input, the result
    will be a row vector, with each element representing the
    squared length along the matrix's corresponding row.
    """
    lengths = np.sum(vec ** 2., axis=-1)

    return lengths

@all_parameters_as_numpy_arrays
def _length(vec):
    """Returns the length of an Nd list of vectors
    or a single vector.

    :param numpy.array vec: an Nd array with the final dimension
        being size 3 (a vector).

        Single vector::

            numpy.array([ x, y, z ])

        Nd array::

            numpy.array([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).

    :rtype: np.array
    :return: The length of vec, if a 1D vector is input, the
    result will be a scalar. If a Matrix is input, the result
    will be a row vector, with each element representing the
    length along the matrix's corresponding row.
    """
    return np.sqrt(np.sum(vec**2,axis=-1))


@parameters_as_numpy_arrays('vec')
def _set_length(vec, len):
    """Renormalizes an Nd list of vectors or a single vector to 'length'.

    The vector is **not** changed in place.

    :param numpy.array vec: an Nd array with the final dimension
        being size 3 (a vector).

        Single vector::
            numpy.array([ x, y, z ])

        Nd array::
            numpy.array([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).

    :rtype: numpy.array
    :return: A renormalized copy of vec, normalized according to its
    the last axis.
    If a vector is input, the result is a vector. If a Matrix is input, 
    the result will be a Matrix, with each row renormalized to a length of len.
    """
    # calculate the length
    # this is a duplicate of length(vec) because we
    # always want an array, even a 0-d array.

    return (vec.T  / np.sqrt(np.sum(vec**2,axis=-1)) * len).T


@all_parameters_as_numpy_arrays
def _dot(v1, v2):
    """Calculates the dot product of two vectors.

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3 (a vector)
    :rtype: numpy.array
    :return: The resulting dot product. If a 1d array was passed, 
    it will be a scalar.
    Otherwise the result will be an array of scalars storing the
    dot product of corresponding rows.
    """
    return np.sum(v1 * v2, axis=-1)

@parameters_as_numpy_arrays('v1', 'v2')
def _interpolate(v1, v2, delta):
    """Interpolates between 2 arrays of vectors (shape = N,3)
    by the specified delta (0.0 <= delta <= 1.0).

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3. (a vector)
    :param float delta: The interpolation percentage to apply,
        where 0.0 <= delta <= 1.0.
        When delta is 0.0, the result will be v1.
        When delta is 1.0, the result will be v2.
        Values in between will be an interpolation.
    :rtype: numpy.array
    :return: The result of intperpolation between v1 and v2
    """
    # scale the difference based on the time
    # we must do it this 'unreadable' way to avoid
    # loss of precision.
    # the 'readable' method (f_now = f_0 + (f1 - f0) * delta)
    # causes floating point errors due to the small values used
    # in md2 files and the values become corrupted.
    # this horrible code curtousey of this comment:
    # http://stackoverflow.com/questions/5448322/temporal-interpolation-in-numpy-matplotlib
    return v1 + ((v2 - v1) * delta)
    #return v1 * (1.0 - delta ) + v2 * delta
    t = delta
    t0 = 0.0
    t1 = 1.0
    delta_t = t1 - t0
    return (t1 - t) / delta_t * v1 + (t - t0) / delta_t * v2

def _cross(v1, v2):
    """Calculates the cross-product of two vectors.

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3. (a vector)
    :rtype: np.array
    :return: The cross product of v1 and v2.
    """
    return np.cross(v1, v2)

def _generate_normals(v1, v2, v3, normalize_result=True):
    r"""Generates a normal vector for 3 vertices.

    The result is a normalized vector.

    It is assumed the ordering is counter-clockwise starting
    at v1, v2 then v3::

        v1      v3
          \    /
            v2

    The vertices are Nd arrays and may be 1d or Nd.
    As long as the final axis is of size 3.

    For 1d arrays::
        >>> v1 = numpy.array( [ 1.0, 0.0, 0.0 ] )
        >>> v2 = numpy.array( [ 0.0, 0.0, 0.0 ] )
        >>> v3 = numpy.array( [ 0.0, 1.0, 0.0 ] )
        >>> vector.generate_normals( v1, v2, v3 )
        array([ 0.,  0., -1.])

    For Nd arrays::
        >>> v1 = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 1.0, 0.0, 0.0 ] ] )
        >>> v2 = numpy.array( [ [ 0.0, 0.0, 0.0 ], [ 0.0, 0.0, 0.0 ] ] )
        >>> v3 = numpy.array( [ [ 0.0, 1.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ] )
        >>> vector.generate_normals( v1, v2, v3 )
        array([[ 0.,  0., -1.],
               [ 0.,  0., -1.]])

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v3: an Nd array with the final dimension
        being size 3. (a vector)
    :param boolean normalize_result: Specifies if the result should
        be normalized before being returned.
    """
    # make vectors relative to v2
    # we assume opengl counter-clockwise ordering
    a = v1 - v2
    b = v3 - v2
    n = _cross(b, a)
    if normalize_result:
        n = _normalise(n)
    return n

class BaseVector(BaseObject):
    @classmethod
    def from_matrix44_translation(cls, matrix, dtype=None):
        return cls(cls._module.create_from_matrix44_translation(matrix, dtype))

    def normalize(self):
        self[:] = self.normalized

    @property
    def normalized(self):
        return type(self)(self._module.normalize(self))

    def normalise(self):    # TODO: mark as deprecated
        self[:] = self.normalized

    @property
    def normalised(self):    # TODO: mark as deprecated
        return type(self)(self._module.normalize(self))

    @property
    def squared_length(self):
        return self._module.squared_length(self)

    @property
    def length(self):
        return self._module.length(self)

    @length.setter
    def length(self, length):
        self[:] = _set_length(self, length)

    def dot(self, other):
        return _dot(self, type(self)(other))

    def cross(self, other):
        return type(self)(_cross(self[:3], other[:3]))

    def interpolate(self, other, delta):
        return type(self)(_interpolate(self, type(self)(other), delta))

    def normal(self, v2, v3, normalize_result=True):
        return type(self)(_generate_normals(self, type(self)(v2), type(self)(v3), normalize_result))

class BaseVector2(BaseVector):
    _shape = (2,)

class BaseVector3(BaseVector):
    _shape = (3,)
    #: The X value of this Vector.
    x = NpProxy(0)
    #: The Y value of this Vector.
    y = NpProxy(1)
    #: The Z value of this Vector.
    z = NpProxy(2)
    #: The X,Y values of this Vector as a numpy.ndarray.
    xy = NpProxy([0,1])
    #: The X,Y,Z values of this Vector as a numpy.ndarray.
    xyz = NpProxy([0,1,2])
    #: The X,Z values of this Vector as a numpy.ndarray.
    xz = NpProxy([0,2])

class BaseVector4(BaseVector):
    _shape = (4,)
    #: The X value of this Vector.
    x = NpProxy(0)
    #: The Y value of this Vector.
    y = NpProxy(1)
    #: The Z value of this Vector.
    z = NpProxy(2)
    #: The W value of this Vector.
    w = NpProxy(3)
    #: The X,Y values of this Vector as a numpy.ndarray.
    xy = NpProxy([0,1])
    #: The X,Y,Z values of this Vector as a numpy.ndarray.
    xyz = NpProxy([0,1,2])
    #: The X,Y,Z,W values of this Vector as a numpy.ndarray.
    xyzw = NpProxy(slice(0,4))
    #: The X,Z values of this Vector as a numpy.ndarray.
    xz = NpProxy([0,2])
    #: The X,W values of this Vector as a numpy.ndarray.
    xw = NpProxy([0,3])
    #: The X,Y,W values of this Vector as a numpy.ndarray.
    xyw = NpProxy([0,1,3])
    #: The X,Z,W values of this Vector as a numpy.ndarray.
    xzw = NpProxy([0,2,3])

@parameters_as_numpy_arrays('vector')
def _vector2_from_vector3(vector, dtype=None):
    """Returns a vector2 and the W component as a tuple.
    """
    dtype = dtype or vector.dtype
    return np.array([vector[0], vector[1]], dtype=dtype), vector[2]

@parameters_as_numpy_arrays('vector')
def _vector2_from_vector4(vector, dtype=None):
    """Returns a vector2 and the W component as a tuple.
    """
    dtype = dtype or vector.dtype
    return np.array([vector[0], vector[1]], dtype=dtype), vector[2], vector[3]


class Vector2(BaseVector2):
    @classmethod
    def from_vector3(cls, vector, dtype=None):
        """Create a Vector2 from a Vector3.

        Returns the Vector2 and the Z component as a tuple.
        """
        vec, z = _vector2_from_vector3(vector, dtype)
        return cls(vec), z

    @classmethod
    def from_vector4(cls, vector, dtype=None):
        """Create a Vector2 from a Vector4.

        Returns the Vector2, Z and the W component as a tuple.
        """
        vec, z, w = _vector2_from_vector4(vector, dtype)
        return cls(vec), z, w

    def __new__(cls, value=None, w=0.0, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Vector2, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(BaseObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(BaseObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(BaseObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(BaseObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(BaseObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    @dispatch((BaseObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((BaseObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((BaseObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((BaseObject, Number, np.number))
    def __eq__(self, other):
        self._unsupported_type('EQ', other)

    ########################
    # Vectors
    @dispatch((BaseVector2, np.ndarray, list))
    def __add__(self, other):
        return Vector2(super(Vector2, self).__add__(other))

    @dispatch((BaseVector2, np.ndarray, list))
    def __sub__(self, other):
        return Vector2(super(Vector2, self).__sub__(other))

    @dispatch((BaseVector2, np.ndarray, list))
    def __mul__(self, other):
        return Vector2(super(Vector2, self).__mul__(other))

    @dispatch((BaseVector2, np.ndarray, list))
    def __truediv__(self, other):
        return Vector2(super(Vector2, self).__truediv__(other))

    @dispatch((BaseVector2, np.ndarray, list))
    def __div__(self, other):
        return Vector2(super(Vector2, self).__div__(other))

    @dispatch((BaseVector2, np.ndarray, list))
    def __xor__(self, other):
        return self.cross(other)

    @dispatch((BaseVector2, np.ndarray, list))
    def __or__(self, other):
        return self.dot(other)

    @dispatch((BaseVector2, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Vector2, self).__ne__(other)))

    @dispatch((BaseVector2, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Vector2, self).__eq__(other)))

    ########################
    # Number
    @dispatch((Number,np.number))
    def __add__(self, other):
        return Vector2(super(Vector2, self).__add__(other))

    @dispatch((Number,np.number))
    def __sub__(self, other):
        return Vector2(super(Vector2, self).__sub__(other))

    @dispatch((Number,np.number))
    def __mul__(self, other):
        return Vector2(super(Vector2, self).__mul__(other))

    @dispatch((Number,np.number))
    def __truediv__(self, other):
        return Vector2(super(Vector2, self).__truediv__(other))

    @dispatch((Number,np.number))
    def __div__(self, other):
        return Vector2(super(Vector2, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def inverse(self):
        """Returns the opposite of this vector.
        """
        return Vector2(-self)

    @property
    def vector2(self):
        return self

@parameters_as_numpy_arrays('vector')
def _vector3_from_vector4(vector, dtype=None):
    """Returns a vector3 and the W component as a tuple.
    """
    dtype = dtype or vector.dtype
    return (np.array([vector[0], vector[1], vector[2]], dtype=dtype), vector[3])

@parameters_as_numpy_arrays('mat')
def _vector3_from_matrix44_translation(mat, dtype=None):
    return np.array(mat[3, :3], dtype=dtype)

class Vector3(BaseVector3):
    @classmethod
    def from_vector4(cls, vector, dtype=None):
        """Create a Vector3 from a Vector4.

        Returns the Vector3 and the W component as a tuple.
        """
        vec, w = _vector3_from_vector4(vector, dtype)
        return (cls(vec), w)

    def __new__(cls, value=None, w=0.0, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Vector3, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(BaseObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(BaseObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(BaseObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(BaseObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(BaseObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    @dispatch((BaseObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((BaseObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((BaseObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((BaseObject, Number, np.number))
    def __eq__(self, other):
        self._unsupported_type('EQ', other)

    ########################
    # Vectors
    @dispatch((BaseVector3, np.ndarray, list))
    def __add__(self, other):
        return Vector3(super(Vector3, self).__add__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __sub__(self, other):
        return Vector3(super(Vector3, self).__sub__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __mul__(self, other):
        return Vector3(super(Vector3, self).__mul__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __truediv__(self, other):
        return Vector3(super(Vector3, self).__truediv__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __div__(self, other):
        return Vector3(super(Vector3, self).__div__(other))

    @dispatch((BaseVector3, np.ndarray, list))
    def __xor__(self, other):
        return self.cross(other)

    @dispatch((BaseVector3, np.ndarray, list))
    def __or__(self, other):
        return self.dot(other)

    @dispatch((BaseVector3, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Vector3, self).__ne__(other)))

    @dispatch((BaseVector3, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Vector3, self).__eq__(other)))

    ########################
    # Number
    @dispatch((Number,np.number))
    def __add__(self, other):
        return Vector3(super(Vector3, self).__add__(other))

    @dispatch((Number,np.number))
    def __sub__(self, other):
        return Vector3(super(Vector3, self).__sub__(other))

    @dispatch((Number,np.number))
    def __mul__(self, other):
        return Vector3(super(Vector3, self).__mul__(other))

    @dispatch((Number,np.number))
    def __truediv__(self, other):
        return Vector3(super(Vector3, self).__truediv__(other))

    @dispatch((Number,np.number))
    def __div__(self, other):
        return Vector3(super(Vector3, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def inverse(self):
        """Returns the opposite of this vector.
        """
        return Vector3(-self)

    @property
    def vector3(self):
        return self

@parameters_as_numpy_arrays('vector')
def _vector4_from_vector3(vector, w=0., dtype=None):
    dtype = dtype or vector.dtype
    return np.array([vector[0], vector[1], vector[2], w], dtype=dtype)

@parameters_as_numpy_arrays('mat')
def _vector4_from_matrix44_translation(mat, dtype=None):
    return np.array(mat[3, :4], dtype=dtype)

class Vector4(BaseVector4):
    @classmethod
    def from_vector3(cls, vector, w=0.0, dtype=None):
        """Create a Vector4 from a Vector3.

        By default, the W value is 0.0.
        """
        return cls(_vector4_from_vector3(vector, w, dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Vector4, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(BaseObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(BaseObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(BaseObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(BaseObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(BaseObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    @dispatch((BaseObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((BaseObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((BaseObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((BaseObject, Number, np.number))
    def __eq__(self, other):
        self._unsupported_type('EQ', other)

    ########################
    # Vectors
    @dispatch((BaseVector4, np.ndarray, list))
    def __add__(self, other):
        return Vector4(super(Vector4, self).__add__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __sub__(self, other):
        return Vector4(super(Vector4, self).__sub__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __mul__(self, other):
        return Vector4(super(Vector4, self).__mul__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __truediv__(self, other):
        return Vector4(super(Vector4, self).__truediv__(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __div__(self, other):
        return Vector4(super(Vector4, self).__div__(other))

    @dispatch(BaseVector)
    def __xor__(self, other):
       return self.cross(Vector4(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __or__(self, other):
        return self.dot(Vector4(other))

    @dispatch((BaseVector4, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Vector4, self).__ne__(other)))

    @dispatch((BaseVector4, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Vector4, self).__eq__(other)))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Vector4(super(Vector4, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Vector4(super(Vector4, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Vector4(super(Vector4, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Vector4(super(Vector4, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Vector4(super(Vector4, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def inverse(self):
        """Returns the opposite of this vector.
        """
        return Vector4(-self)

    @property
    def vector3(self):
        """Returns a Vector3 and the W component as a tuple.
        """
        return (Vector3(self[:3]), self[3])