# spritekit/math.py
#
# Copyright (C) 2025 George Watson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Based off of Pyrr:
# https://github.com/adamlwgriffiths/Pyrr/
# In the original BSD license, both occurrences of the phrase "COPYRIGHT HOLDERS AND CONTRIBUTORS" in the disclaimer read "REGENTS AND CONTRIBUTORS".
#
# Copyright (c) 2015, Adam Griffiths
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies, 
# either expressed or implied, of the FreeBSD Project.

import numpy as np
import inspect
from functools import wraps
from numbers import Number
from multipledispatch import dispatch

__all__ = ["Vector2", "Vector3", "Vector4", "Matrix"]

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

class NpObject(np.ndarray):
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

class BaseVector(NpObject):
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

class BaseMatrix(NpObject):
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

@all_parameters_as_numpy_arrays
def vector_normalize(vec):    # TODO: mark as deprecated
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
def vector_squared_length(vec):
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
def vector_length(vec):
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
def vector_set_length(vec, len):
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
def vector_dot(v1, v2):
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
def vector_interpolate(v1, v2, delta):
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

def vector_cross(v1, v2):
    """Calculates the cross-product of two vectors.

    :param numpy.array v1: an Nd array with the final dimension
        being size 3. (a vector)
    :param numpy.array v2: an Nd array with the final dimension
        being size 3. (a vector)
    :rtype: np.array
    :return: The cross product of v1 and v2.
    """
    return np.cross(v1, v2)

def vector_generate_normals(v1, v2, v3, normalize_result=True):
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
    n = vector_cross(b, a)
    if normalize_result:
        n = vector_normalize(n)
    return n

class VectorCore:
    def normalize(self):
        self[:] = self.normalized

    @property
    def normalized(self):
        return vector_normalize(self)

    def normalise(self):    # TODO: mark as deprecated
        self[:] = self.normalized

    @property
    def normalised(self):    # TODO: mark as deprecated
        return vector_normalize(self)

    @property
    def squared_length(self):
        return vector_squared_length(self)

    @property
    def length(self):
        return vector_length(self)

    @length.setter
    def length(self, length):
        self[:] = vector_set_length(self, length)

    def dot(self, other):
        return vector_dot(self, type(self)(other))

    def cross(self, other):
        return vector_cross(self[:3], other[:3])

    def interpolate(self, other, delta):
        return vector_interpolate(self, other, delta)

    def normal(self, v2, v3, normalize_result=True):
        return vector_generate_normals(self, v2, v3, normalize_result)

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

class Vector2(BaseVector2, VectorCore):
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
    @dispatch(NpObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(NpObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(NpObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(NpObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(NpObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    @dispatch((NpObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((NpObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((NpObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((NpObject, Number, np.number))
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
def create_from_matrix44_translation(mat, dtype=None):
    return np.array(mat[3, :3], dtype=dtype)

class Vector3(BaseVector3, VectorCore):
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
            # matrix44
            if obj.shape in ((4,4,)):
                obj = create_from_matrix44_translation(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Vector3, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(NpObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(NpObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(NpObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(NpObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(NpObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    @dispatch((NpObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((NpObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((NpObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((NpObject, Number, np.number))
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

@parameters_as_numpy_arrays('vector')
def _vector4_from_vector3(vector, w=0., dtype=None):
    dtype = dtype or vector.dtype
    return np.array([vector[0], vector[1], vector[2], w], dtype=dtype)

@parameters_as_numpy_arrays('mat')
def create_from_matrix44_translation(mat, dtype=None):
    return np.array(mat[3, :4], dtype=dtype)

class Vector4(BaseVector4, VectorCore):
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
            # matrix44
            if obj.shape in ((4,4,)):
                obj = create_from_matrix44_translation(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Vector4, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(NpObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(NpObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(NpObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(NpObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(NpObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    @dispatch((NpObject, Number, np.number))
    def __xor__(self, other):
        self._unsupported_type('XOR', other)

    @dispatch((NpObject, Number, np.number))
    def __or__(self, other):
        self._unsupported_type('OR', other)

    @dispatch((NpObject, Number, np.number))
    def __ne__(self, other):
        self._unsupported_type('NE', other)

    @dispatch((NpObject, Number, np.number))
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

@parameters_as_numpy_arrays('vector')
def matrix33_create_from_eulers(eulers, dtype=None):
    """Creates a matrix from the specified Euler rotations.

    :param numpy.array eulers: A set of euler rotations in the format
        specified by the euler modules.
    :rtype: numpy.array
    :return: A matrix with shape (3,3) with the euler's rotation.
    """
    dtype = dtype or eulers.dtype

    pitch, roll, yaw = eulers

    sP = np.sin(pitch)
    cP = np.cos(pitch)
    sR = np.sin(roll)
    cR = np.cos(roll)
    sY = np.sin(yaw)
    cY = np.cos(yaw)

    return np.array(
        [
            # m1
            [
                cY * cP,
                -cY * sP * cR + sY * sR,
                cY * sP * sR + sY * cR,
            ],
            # m2
            [
                sP,
                cP * cR,
                -cP * sR,
            ],
            # m3
            [
                -sY * cP,
                sY * sP * cR + cY * sR,
                -sY * sP * sR + cY * cR,
            ]
        ],
        dtype=dtype
    )



def create_from_scale(scale, dtype=None):
    """Creates an identity matrix with the scale set.

    :param numpy.array scale: The scale to apply as a vector (shape 3).
    :rtype: numpy.array
    :return: A matrix with shape (3,3) with the scale
        set to the specified vector.
    """
    # apply the scale to the values diagonally
    # down the matrix
    m = np.diagflat(scale)
    if dtype:
        m = m.astype(dtype)
    return m

def matrix33_create_from_x_rotation(theta, dtype=None):
    """Creates a matrix with the specified rotation about the X axis.

    :param float theta: The rotation, in radians, about the X-axis.
    :rtype: numpy.array
    :return: A matrix with the shape (3,3) with the specified rotation about
        the X-axis.

    .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """
    cosT = np.cos(theta)
    sinT = np.sin(theta)

    return np.array(
        [
            [ 1.0, 0.0, 0.0 ],
            [ 0.0, cosT,-sinT ],
            [ 0.0, sinT, cosT ]
        ],
        dtype=dtype
    )

def matrix33_create_from_y_rotation(theta, dtype=None):
    """Creates a matrix with the specified rotation about the Y axis.

    :param float theta: The rotation, in radians, about the Y-axis.
    :rtype: numpy.array
    :return: A matrix with the shape (3,3) with the specified rotation about
        the Y-axis.

    .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """
    cosT = np.cos(theta)
    sinT = np.sin(theta)

    return np.array(
        [
            [ cosT, 0.0,sinT ],
            [ 0.0, 1.0, 0.0 ],
            [-sinT, 0.0, cosT ]
        ],
        dtype=dtype
    )

def matrix33_create_from_z_rotation(theta, dtype=None):
    """Creates a matrix with the specified rotation about the Z axis.

    :param float theta: The rotation, in radians, about the Z-axis.
    :rtype: numpy.array
    :return: A matrix with the shape (3,3) with the specified rotation about
        the Z-axis.

    .. seealso:: http://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """
    cosT = np.cos(theta)
    sinT = np.sin(theta)

    return np.array(
        [
            [ cosT,-sinT, 0.0 ],
            [ sinT, cosT, 0.0 ],
            [ 0.0, 0.0, 1.0 ]
        ],
        dtype=dtype
    )

class MatrixCore:
    @classmethod
    def identity(cls, dtype=None):
        """Creates an identity Matrix.
        """
        assert cls._shape[0] == cls._shape[1]
        return np.identity(cls._shape[0], dtype=dtype)

    @classmethod
    def from_eulers(cls, eulers, dtype=None):
        """Creates a Matrix from the specified Euler angles.
        """
        return matrix33_create_from_eulers(eulers, dtype=dtype)

    @classmethod
    def from_scale(cls, scale, dtype=None):
        return create_from_scale(scale, dtype=dtype)

    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the X-axis.
        """
        return matrix33_create_from_x_rotation(theta, dtype=dtype)

    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        return matrix33_create_from_y_rotation(theta, dtype=dtype)

    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the Z-axis.
        """
        return matrix33_create_from_z_rotation(theta, dtype=dtype)

    @property
    def inverse(self):
        """Returns the inverse of this matrix.
        """
        return np.linalg.inv(self)

def m324(m33, dtype=None):
    dtype = dtype or m33.dtype
    m = np.identity(4, dtype=dtype)
    m[0:3, 0:3] = m33
    return m

def matrix44_create_perspective_projection_from_bounds(
    left,
    right,
    bottom,
    top,
    near,
    far,
    dtype=None
):
    """Creates a perspective projection matrix using the specified near
    plane dimensions.

    :param float left: The left of the near plane relative to the plane's centre.
    :param float right: The right of the near plane relative to the plane's centre.
    :param float top: The top of the near plane relative to the plane's centre.
    :param float bottom: The bottom of the near plane relative to the plane's centre.
    :param float near: The distance of the near plane from the camera's origin.
        It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
        at close range.
    :param float far: The distance of the far plane from the camera's origin.
    :rtype: numpy.array
    :return: A projection matrix representing the specified perspective.

    .. seealso:: http://www.gamedev.net/topic/264248-building-a-projection-matrix-without-api/
    .. seealso:: http://www.glprogramming.com/red/chapter03.html
    """

    """
    E 0 A 0
    0 F B 0
    0 0 C D
    0 0-1 0

    A = (right+left)/(right-left)
    B = (top+bottom)/(top-bottom)
    C = -(far+near)/(far-near)
    D = -2*far*near/(far-near)
    E = 2*near/(right-left)
    F = 2*near/(top-bottom)
    """
    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -2. * far * near / (far - near)
    E = 2. * near / (right - left)
    F = 2. * near / (top - bottom)

    return np.array((
        (  E, 0., 0., 0.),
        ( 0.,  F, 0., 0.),
        (  A,  B,  C,-1.),
        ( 0., 0.,  D, 0.),
    ), dtype=dtype)

def matrix44_create_perspective_projection(fovy, aspect, near, far, dtype=None):
    """Creates perspective projection matrix.

    .. seealso:: http://www.opengl.org/sdk/docs/man2/xhtml/gluPerspective.xml
    .. seealso:: http://www.geeks3d.com/20090729/howto-perspective-projection-matrix-in-opengl/

    :param float fovy: field of view in y direction in degrees
    :param float aspect: aspect ratio of the view (width / height)
    :param float near: distance from the viewer to the near clipping plane (only positive)
    :param float far: distance from the viewer to the far clipping plane (only positive)
    :rtype: numpy.array
    :return: A projection matrix representing the specified perpective.
    """
    ymax = near * np.tan(fovy * np.pi / 360.0)
    xmax = ymax * aspect
    return matrix44_create_perspective_projection_from_bounds(-xmax, xmax, -ymax, ymax, near, far, dtype=dtype)

def matrix44_create_perspective_projection_from_bounds(
    left,
    right,
    bottom,
    top,
    near,
    far,
    dtype=None
):
    """Creates a perspective projection matrix using the specified near
    plane dimensions.

    :param float left: The left of the near plane relative to the plane's centre.
    :param float right: The right of the near plane relative to the plane's centre.
    :param float top: The top of the near plane relative to the plane's centre.
    :param float bottom: The bottom of the near plane relative to the plane's centre.
    :param float near: The distance of the near plane from the camera's origin.
        It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
        at close range.
    :param float far: The distance of the far plane from the camera's origin.
    :rtype: numpy.array
    :return: A projection matrix representing the specified perspective.

    .. seealso:: http://www.gamedev.net/topic/264248-building-a-projection-matrix-without-api/
    .. seealso:: http://www.glprogramming.com/red/chapter03.html
    """

    """
    E 0 A 0
    0 F B 0
    0 0 C D
    0 0-1 0

    A = (right+left)/(right-left)
    B = (top+bottom)/(top-bottom)
    C = -(far+near)/(far-near)
    D = -2*far*near/(far-near)
    E = 2*near/(right-left)
    F = 2*near/(top-bottom)
    """
    A = (right + left) / (right - left)
    B = (top + bottom) / (top - bottom)
    C = -(far + near) / (far - near)
    D = -2. * far * near / (far - near)
    E = 2. * near / (right - left)
    F = 2. * near / (top - bottom)

    return np.array((
        (  E, 0., 0., 0.),
        ( 0.,  F, 0., 0.),
        (  A,  B,  C,-1.),
        ( 0., 0.,  D, 0.),
    ), dtype=dtype)

def matrix44_create_orthogonal_projection(
    left,
    right,
    bottom,
    top,
    near,
    far,
    dtype=None
):
    """Creates an orthogonal projection matrix.

    :param float left: The left of the near plane relative to the plane's centre.
    :param float right: The right of the near plane relative to the plane's centre.
    :param float top: The top of the near plane relative to the plane's centre.
    :param float bottom: The bottom of the near plane relative to the plane's centre.
    :param float near: The distance of the near plane from the camera's origin.
        It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
        at close range.
    :param float far: The distance of the far plane from the camera's origin.
    :rtype: numpy.array
    :return: A projection matrix representing the specified orthogonal perspective.

    .. seealso:: http://msdn.microsoft.com/en-us/library/dd373965(v=vs.85).aspx
    """

    """
    A 0 0 Tx
    0 B 0 Ty
    0 0 C Tz
    0 0 0 1

    A = 2 / (right - left)
    B = 2 / (top - bottom)
    C = -2 / (far - near)

    Tx = (right + left) / (right - left)
    Ty = (top + bottom) / (top - bottom)
    Tz = (far + near) / (far - near)
    """
    rml = right - left
    tmb = top - bottom
    fmn = far - near

    A = 2. / rml
    B = 2. / tmb
    C = -2. / fmn
    Tx = -(right + left) / rml
    Ty = -(top + bottom) / tmb
    Tz = -(far + near) / fmn

    return np.array((
        ( A, 0., 0., 0.),
        (0.,  B, 0., 0.),
        (0., 0.,  C, 0.),
        (Tx, Ty, Tz, 1.),
    ), dtype=dtype)

def matrix44_create_orthogonal_projection(
    left, right, bottom, top, near, far, dtype=None):    # TDOO: mark as deprecated
    """Creates an orthogonal projection matrix.

    :param float left: The left of the near plane relative to the plane's centre.
    :param float right: The right of the near plane relative to the plane's centre.
    :param float top: The top of the near plane relative to the plane's centre.
    :param float bottom: The bottom of the near plane relative to the plane's centre.
    :param float near: The distance of the near plane from the camera's origin.
        It is recommended that the near plane is set to 1.0 or above to avoid rendering issues
        at close range.
    :param float far: The distance of the far plane from the camera's origin.
    :rtype: numpy.array
    :return: A projection matrix representing the specified orthogonal perspective.

    .. seealso:: http://msdn.microsoft.com/en-us/library/dd373965(v=vs.85).aspx
    """

    """
    A 0 0 Tx
    0 B 0 Ty
    0 0 C Tz
    0 0 0 1

    A = 2 / (right - left)
    B = 2 / (top - bottom)
    C = -2 / (far - near)

    Tx = (right + left) / (right - left)
    Ty = (top + bottom) / (top - bottom)
    Tz = (far + near) / (far - near)
    """
    return matrix44_create_orthogonal_projection(
        left, right, bottom, top, near, far, dtype
    )

def matrix44_create_look_at(eye, target, up, dtype=None):
    """Creates a look at matrix according to OpenGL standards.

    :param numpy.array eye: Position of the camera in world coordinates.
    :param numpy.array target: The position in world coordinates that the
        camera is looking at.
    :param numpy.array up: The up vector of the camera.
    :rtype: numpy.array
    :return: A look at matrix that can be used as a viewMatrix
    """

    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)

    forward = vector_normalize(target - eye)
    side = vector_normalize(np.cross(forward, up))
    up = vector_normalize(np.cross(side, forward))

    return np.array((
            (side[0], up[0], -forward[0], 0.),
            (side[1], up[1], -forward[1], 0.),
            (side[2], up[2], -forward[2], 0.),
            (-np.dot(side, eye), -np.dot(up, eye), np.dot(forward, eye), 1.0)
        ), dtype=dtype)

@parameters_as_numpy_arrays('vec')
def matrix44_create_from_translation(vec, dtype=None):
    """Creates an identity matrix with the translation set.

    :param numpy.array vec: The translation vector (shape 3 or 4).
    :rtype: numpy.array
    :return: A matrix with shape (4,4) that represents a matrix
        with the translation set to the specified vector.
    """
    dtype = dtype or vec.dtype
    mat = np.identity(4, dtype=dtype)
    mat[3, 0:3] = vec[:3]
    return mat

@all_parameters_as_numpy_arrays
def matrix44_apply_to_vector(mat, vec):
    """Apply a matrix to a vector.

    The matrix's rotation and translation are applied to the vector.
    Supports multiple matrices and vectors.

    :param numpy.array mat: The rotation / translation matrix.
        Can be a list of matrices.
    :param numpy.array vec: The vector to modify.
        Can be a numpy.array of vectors. ie. numpy.array([[x1,...], [x2,...], ...])
    :rtype: numpy.array
    :return: The vectors rotated by the specified matrix.
    """
    size = vec.shape[len(vec.shape) - 1]
    if size == 3:
        # convert to a vec4
        if len(vec.shape) == 1:
            vec4 = np.array([vec[0], vec[1], vec[2], 1.], dtype=vec.dtype)
            vec4 = np.dot(vec4, mat)
            if np.abs(vec4[3]) < 1e-8:
                vec4[:] = [np.inf, np.inf, np.inf, np.inf]
            else:
                vec4 /= vec4[3]
            return vec4[:3]
        else:
            vec4 = np.array([[v[0], v[1], v[2], 1.] for v in vec], dtype=vec.dtype)
            vec4 = np.dot(vec4, mat)
            for i in range(vec4.shape[0]):
                if np.abs(vec4[i,3])<1e-8:
                    vec4[i,:] = [np.inf, np.inf, np.inf, np.inf]
                else:
                    vec4[i,:] /= vec4[i,3]
            return vec4[:,:3]
    elif size == 4:
        return np.dot(vec, mat)
    else:
        raise ValueError("Vector size unsupported")

class Matrix(NpObject, MatrixCore):
    @classmethod
    def from_eulers(cls, eulers, dtype=None):
        """Creates a Matrix from the specified Euler angles.
        """
        return m324(matrix33_create_from_eulers(eulers, dtype=dtype), dtype=dtype)
    
    @classmethod
    def from_scale(cls, scale, dtype=None):
        """Creates an identity matrix with the scale set.
        """
        return create_from_scale([scale[0], scale[1], scale[2], 1.0], dtype=dtype)

    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the X-axis.
        """
        return m324(matrix33_create_from_x_rotation(theta, dtype=dtype), dtype=dtype)
    
    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the Y-axis.
        """
        return m324(matrix33_create_from_y_rotation(theta, dtype=dtype), dtype=dtype)
    
    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """Creates a Matrix with a rotation around the Z-axis.
        """
        return m324(matrix33_create_from_z_rotation(theta, dtype=dtype), dtype=dtype)

    @classmethod
    def perspective_projection(cls, fovy, aspect, near, far, dtype=None):
        """Creates a Matrix for use as a perspective projection matrix.
        """
        return cls(matrix44_create_perspective_projection(fovy, aspect, near, far, dtype))

    @classmethod
    def perspective_projection_bounds(cls, left, right, top, bottom, near, far, dtype=None):
        """Creates a Matrix for use as a perspective projection matrix.
        """
        return cls(matrix44_create_perspective_projection_from_bounds(left, right, top, bottom, near, far, dtype))

    @classmethod
    def orthogonal_projection(cls, left, right, top, bottom, near, far, dtype=None):
        """Creates a Matrix for use as an orthogonal projection matrix.
        """
        return cls(matrix44_create_orthogonal_projection(left, right, top, bottom, near, far, dtype))

    @classmethod
    def look_at(cls, eye, target, up, dtype=None):
        """Creates a Matrix for use as a lookAt matrix.
        """
        return cls(matrix44_create_look_at(eye, target, up, dtype))

    @classmethod
    def from_translation(cls, translation, dtype=None):
        """Creates a Matrix from the specified translation.
        """
        return cls(matrix44_create_from_translation(translation, dtype=dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Matrix, cls).__new__(cls, obj)

    ########################
    # Basic Operators
    @dispatch(NpObject)
    def __add__(self, other):
        self._unsupported_type('add', other)

    @dispatch(NpObject)
    def __sub__(self, other):
        self._unsupported_type('subtract', other)

    @dispatch(NpObject)
    def __mul__(self, other):
        self._unsupported_type('multiply', other)

    @dispatch(NpObject)
    def __truediv__(self, other):
        self._unsupported_type('divide', other)

    @dispatch(NpObject)
    def __div__(self, other):
        self._unsupported_type('divide', other)

    def __invert__(self):
        return self.inverse

    ########################
    # Matrices
    @dispatch((BaseMatrix, np.ndarray, list))
    def __add__(self, other):
        return Matrix(super(Matrix, self).__add__(Matrix(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __sub__(self, other):
        return Matrix(super(Matrix, self).__sub__(Matrix(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __mul__(self, other):
        return Matrix(np.dot(self, other))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Matrix, self).__ne__(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Matrix, self).__eq__(other)))

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(matrix44_apply_to_vector(self, other))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Matrix(super(Matrix, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Matrix(super(Matrix, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Matrix(super(Matrix, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Matrix(super(Matrix, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Matrix(super(Matrix, self).__div__(other))