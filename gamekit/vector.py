
from numbers import Number
from multipledispatch import dispatch
from .math import (all_parameters_as_numpy_arrays, parameters_as_numpy_arrays, 
                   BaseObject, BaseVector, BaseVector2, BaseVector3, BaseVector4)
import numpy as np

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