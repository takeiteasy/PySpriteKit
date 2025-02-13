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

class BaseVector(BaseObject):
    pass

class BaseVector2(BaseVector):
    _shape = (2,)
    #: The X value of this Vector.
    x = NpProxy(0)
    #: The Y value of this Vector.
    y = NpProxy(1)
    #: The X,Y values of this Vector as a numpy.ndarray.
    xy = NpProxy([0,1])
    #: The X,Y,Z values of this Vector as a numpy.ndarray.
    yx = NpProxy([1,0])

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

class BaseQuaternion(BaseObject):
    _shape = (4,)
    #: The X value of this Quaternion.
    x = NpProxy(0)
    #: The Y value of this Quaternion.
    y = NpProxy(1)
    #: The Z value of this Quaternion.
    z = NpProxy(2)
    #: The W value of this Quaternion.
    w = NpProxy(3)
    #: The X,Y value of this Quaternion as a numpy.ndarray.
    xy = NpProxy([0,1])
    #: The X,Y,Z value of this Quaternion as a numpy.ndarray.
    xyz = NpProxy([0,1,2])
    #: The X,Y,Z,W value of this Quaternion as a numpy.ndarray.
    xyzw = NpProxy([0,1,2,3])
    #: The X,Z value of this Quaternion as a numpy.ndarray.
    xz = NpProxy([0,2])
    #: The X,Z,W value of this Quaternion as a numpy.ndarray.
    xzw = NpProxy([0,2,3])
    #: The X,Y,W value of this Quaternion as a numpy.ndarray.
    xyw = NpProxy([0,1,3])
    #: The X,W value of this Quaternion as a numpy.ndarray.
    xw = NpProxy([0,3])

class BaseMatrix(BaseObject):
    pass

class BaseMatrix3(BaseMatrix):
    _shape = (3,3,)
    # m<c> style access
    #: The first row of this Matrix as a numpy.ndarray.
    m1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray.
    m2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray.
    m3 = NpProxy(2)
    # m<r><c> access
    #: The [0,0] value of this Matrix.
    m11 = NpProxy((0,0))
    #: The [0,1] value of this Matrix.
    m12 = NpProxy((0,1))
    #: The [0,2] value of this Matrix.
    m13 = NpProxy((0,2))
    #: The [1,0] value of this Matrix.
    m21 = NpProxy((1,0))
    #: The [1,1] value of this Matrix.
    m22 = NpProxy((1,1))
    #: The [1,2] value of this Matrix.
    m23 = NpProxy((1,2))
    #: The [2,0] value of this Matrix.
    m31 = NpProxy((2,0))
    #: The [2,1] value of this Matrix.
    m32 = NpProxy((2,1))
    #: The [2,2] value of this Matrix.
    m33 = NpProxy((2,2))
    # rows
    #: The first row of this Matrix as a numpy.ndarray. This is the same as m1.
    r1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray. This is the same as m2.
    r2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray. This is the same as m3.
    r3 = NpProxy(2)
    # columns
    #: The first column of this Matrix as a numpy.ndarray.
    c1 = NpProxy((slice(0,3),0))
    #: The second column of this Matrix as a numpy.ndarray.
    c2 = NpProxy((slice(0,3),1))
    #: The third column of this Matrix as a numpy.ndarray.
    c3 = NpProxy((slice(0,3),2))

class BaseMatrix4(BaseMatrix):
    _shape = (4,4,)
    # m<c> style access
    #: The first row of this Matrix as a numpy.ndarray.
    m1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray.
    m2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray.
    m3 = NpProxy(2)
    #: The fourth row of this Matrix as a numpy.ndarray.
    m4 = NpProxy(3)
    # m<r><c> access
    #: The [0,0] value of this Matrix.
    m11 = NpProxy((0,0))
    #: The [0,1] value of this Matrix.
    m12 = NpProxy((0,1))
    #: The [0,2] value of this Matrix.
    m13 = NpProxy((0,2))
    #: The [0,3] value of this Matrix.
    m14 = NpProxy((0,3))
    #: The [1,0] value of this Matrix.
    m21 = NpProxy((1,0))
    #: The [1,1] value of this Matrix.
    m22 = NpProxy((1,1))
    #: The [1,2] value of this Matrix.
    m23 = NpProxy((1,2))
    #: The [1,3] value of this Matrix.
    m24 = NpProxy((1,3))
    #: The [2,0] value of this Matrix.
    m31 = NpProxy((2,0))
    #: The [2,1] value of this Matrix.
    m32 = NpProxy((2,1))
    #: The [2,2] value of this Matrix.
    m33 = NpProxy((2,2))
    #: The [2,3] value of this Matrix.
    m34 = NpProxy((2,3))
    #: The [3,0] value of this Matrix.
    m41 = NpProxy((3,0))
    #: The [3,1] value of this Matrix.
    m42 = NpProxy((3,1))
    #: The [3,2] value of this Matrix.
    m43 = NpProxy((3,2))
    #: The [3,3] value of this Matrix.
    m44 = NpProxy((3,3))
    # rows
    #: The first row of this Matrix as a numpy.ndarray. This is the same as m1.
    r1 = NpProxy(0)
    #: The second row of this Matrix as a numpy.ndarray. This is the same as m2.
    r2 = NpProxy(1)
    #: The third row of this Matrix as a numpy.ndarray. This is the same as m3.
    r3 = NpProxy(2)
    #: The fourth row of this Matrix as a numpy.ndarray. This is the same as m4.
    r4 = NpProxy(3)
    # columns
    #: The first column of this Matrix as a numpy.ndarray.
    c1 = NpProxy((slice(0,4),0))
    #: The second column of this Matrix as a numpy.ndarray.
    c2 = NpProxy((slice(0,4),1))
    #: The third column of this Matrix as a numpy.ndarray.
    c3 = NpProxy((slice(0,4),2))
    #: The fourth column of this Matrix as a numpy.ndarray.
    c4 = NpProxy((slice(0,4),3))

from .vector import Vector2, Vector3, Vector4
from .quaternion import Quaternion
from .matrix import Matrix3, Matrix4

__all__ = ['Vector2', 'Vector3', 'Vector4', 'Quaternion', 'Matrix3', 'Matrix4']