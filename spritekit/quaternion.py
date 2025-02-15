# spritekit/quaternion.py
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

from .vector import (vector_normalize, vector_squared_length, vector_length, vector_dot,
                     BaseVector, Vector3)
from .math import (all_parameters_as_numpy_arrays, parameters_as_numpy_arrays, 
                   BaseObject, BaseQuaternion, BaseMatrix3, BaseMatrix4, BaseMatrix)
import numpy as np
from multipledispatch import dispatch

def create(x=0., y=0., z=0., w=1., dtype=None):
    return np.array([x, y, z, w], dtype=dtype)

def create_from_x_rotation(theta, dtype=None):
    thetaOver2 = theta * 0.5

    return np.array(
        [
            np.sin(thetaOver2),
            0.0,
            0.0,
            np.cos(thetaOver2)
        ],
        dtype=dtype
    )

def create_from_y_rotation(theta, dtype=None):
    thetaOver2 = theta * 0.5

    return np.array(
        [
            0.0,
            np.sin(thetaOver2),
            0.0,
            np.cos(thetaOver2)
        ],
        dtype=dtype
    )

def create_from_z_rotation(theta, dtype=None):
    thetaOver2 = theta * 0.5

    return np.array(
        [
            0.0,
            0.0,
            np.sin(thetaOver2),
            np.cos(thetaOver2)
        ],
        dtype=dtype
    )

@parameters_as_numpy_arrays('axis')
def create_from_axis_rotation(axis, theta, dtype=None):
    dtype = dtype or axis.dtype
    # make sure the vector is normalized
    if not np.isclose(np.linalg.norm(axis), 1.):
        axis = vector_normalize(axis)

    thetaOver2 = theta * 0.5
    sinThetaOver2 = np.sin(thetaOver2)

    return np.array(
        [
            sinThetaOver2 * axis[0],
            sinThetaOver2 * axis[1],
            sinThetaOver2 * axis[2],
            np.cos(thetaOver2)
        ],
        dtype=dtype
    )

@parameters_as_numpy_arrays('axis')
def create_from_axis(axis, dtype=None):
    dtype = dtype or axis.dtype
    theta = np.linalg.norm(axis)
    return create_from_axis_rotation(axis, theta, dtype)

@parameters_as_numpy_arrays('mat')
def create_from_matrix(mat, dtype=None):
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    # optimised "alternative version" does not produce correct results
    # see issue #42
    dtype = dtype or mat.dtype

    trace = mat[0][0] + mat[1][1] + mat[2][2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qx = (mat[2][1] - mat[1][2]) * s
        qy = (mat[0][2] - mat[2][0]) * s
        qz = (mat[1][0] - mat[0][1]) * s
        qw = 0.25 / s
    elif mat[0][0] > mat[1][1] and mat[0][0] > mat[2][2]:
        s = 2.0 * np.sqrt(1.0 + mat[0][0] - mat[1][1] - mat[2][2])
        qx = 0.25 * s
        qy = (mat[0][1] + mat[1][0]) / s
        qz = (mat[0][2] + mat[2][0]) / s
        qw = (mat[2][1] - mat[1][2]) / s
    elif mat[1][1] > mat[2][2]:
        s = 2.0 * np.sqrt(1.0 + mat[1][1] - mat[0][0] - mat[2][2])
        qx = (mat[0][1] + mat[1][0]) / s
        qy = 0.25 * s
        qz = (mat[1][2] + mat[2][1]) / s
        qw = (mat[0][2] - mat[2][0]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2][2] - mat[0][0] - mat[1][1])
        qx = (mat[0][2] + mat[2][0]) / s
        qy = (mat[1][2] + mat[2][1]) / s
        qz = 0.25 * s
        qw = (mat[1][0] - mat[0][1]) / s

    quat = np.array([qx, qy, qz, qw], dtype=dtype)
    return quat

@parameters_as_numpy_arrays('vector')
def create_from_eulers(eulers, dtype=None):
    """Creates a quaternion from a set of Euler angles.

    Eulers are an array of length 3 in the following order:
        [roll, pitch, yaw]
    """
    dtype = dtype or eulers.dtype

    roll, pitch, yaw = eulers

    halfRoll = roll * 0.5
    sR = np.sin(halfRoll)
    cR = np.cos(halfRoll)

    halfPitch = pitch * 0.5
    sP = np.sin(halfPitch)
    cP = np.cos(halfPitch)

    halfYaw = yaw * 0.5
    sY = np.sin(halfYaw)
    cY = np.cos(halfYaw)

    return np.array(
        [
            (sR * cP * cY) + (cR * sP * sY),
            (cR * sP * cY) - (sR * cP * sY),
            (cR * cP * sY) + (sR * sP * cY),
            (cR * cP * cY) - (sR * sP * sY),
        ],
        dtype=dtype
    )

@parameters_as_numpy_arrays('vector')
def create_from_inverse_of_eulers(eulers, dtype=None):
    """Creates a quaternion from the inverse of a set of Euler angles.

    Eulers are an array of length 3 in the following order:
        [roll, pitch, yaw]
    """
    dtype = dtype or eulers.dtype

    roll, pitch, yaw = eulers

    halfRoll = roll * 0.5
    sinRoll = np.sin(halfRoll)
    cosRoll = np.cos(halfRoll)

    halfPitch = pitch * 0.5
    sinPitch = np.sin(halfPitch)
    cosPitch = np.cos(halfPitch)

    halfYaw = yaw * 0.5
    sinYaw = np.sin(halfYaw)
    cosYaw = np.cos(halfYaw)

    return np.array(
        [
            # x = cy * sp * cr + sy * cp * sr
            (cosYaw * sinPitch * cosRoll) + (sinYaw * cosPitch * sinRoll),
            # y = -cy * sp * sr + sy * cp * cr
            (-cosYaw * sinPitch * sinRoll) + (sinYaw * cosPitch * cosRoll),
            # z = -sy * sp * cr + cy * cp * sr
            (-sinYaw * sinPitch * cosRoll) + (cosYaw * cosPitch * sinRoll),
            # w = cy * cp * cr + sy * sp * sr
            (cosYaw * cosPitch * cosRoll) + (sinYaw * sinPitch * sinRoll)
        ],
        dtype=dtype
    )

@all_parameters_as_numpy_arrays
def cross(quat1, quat2):
    """Returns the cross-product of the two quaternions.

    Quaternions are **not** communicative. Therefore, order is important.

    This is NOT the same as a vector cross-product.
    Quaternion cross-product is the equivalent of matrix multiplication.
    """
    q1x, q1y, q1z, q1w = quat1
    q2x, q2y, q2z, q2w = quat2

    return np.array(
        [
             q1x * q2w + q1y * q2z - q1z * q2y + q1w * q2x,
            -q1x * q2z + q1y * q2w + q1z * q2x + q1w * q2y,
             q1x * q2y - q1y * q2x + q1z * q2w + q1w * q2z,
            -q1x * q2x - q1y * q2y - q1z * q2z + q1w * q2w,
        ],
        dtype=quat1.dtype
    )

def lerp(quat1, quat2, t):
    """Interpolates between quat1 and quat2 by t.
    The parameter t is clamped to the range [0, 1]
    """

    quat1 = np.asarray(quat1)
    quat2 = np.asarray(quat2)

    t = np.clip(t, 0, 1)
    return normalize(quat1 * (1 - t) + quat2 * t)

def slerp(quat1, quat2, t):
    """Spherically interpolates between quat1 and quat2 by t.
    The parameter t is clamped to the range [0, 1]
    """

    quat1 = np.asarray(quat1)
    quat2 = np.asarray(quat2)

    t = np.clip(t, 0, 1)
    dot = vector_dot(quat1, quat2)

    if dot < 0.0:
        dot = -dot
        quat3 = -quat2

    else:
        quat3 = quat2

    if dot < 0.95:
        angle = np.arccos(dot)
        res = (quat1 * np.sin(angle * (1 - t)) + quat3 * np.sin(angle * t)) / np.sin(angle)

    else:
        res = lerp(quat1, quat2, t)

    return res

def is_zero_length(quat):
    """Checks if a quaternion is zero length.

    :param numpy.array quat: The quaternion to check.
    :rtype: boolean.
    :return: True if the quaternion is zero length, otherwise False.
    """
    return quat[0] == quat[1] == quat[2] == quat[3] == 0.0

def is_non_zero_length(quat):
    """Checks if a quaternion is not zero length.

    This is the opposite to 'is_zero_length'.
    This is provided for readability.

    :param numpy.array quat: The quaternion to check.
    :rtype: boolean
    :return: False if the quaternion is zero length, otherwise True.

    .. seealso:: is_zero_length
    """
    return not is_zero_length(quat)

def squared_length(quat):
    """Calculates the squared length of a quaternion.

    Useful for avoiding the performanc penalty of
    the square root function.

    :param numpy.array quat: The quaternion to measure.
    :rtype: float, numpy.array
    :return: If a 1d array was passed, it will be a scalar.
        Otherwise the result will be an array of scalars with shape
        vec.ndim with the last dimension being size 1.
    """
    return vector_squared_length(quat)

def length(quat):
    """Calculates the length of a quaternion.

    :param numpy.array quat: The quaternion to measure.
    :rtype: float, numpy.array
    :return: If a 1d array was passed, it will be a scalar.
        Otherwise the result will be an array of scalars with shape
        vec.ndim with the last dimension being size 1.
    """
    return vector_length(quat)

def normalize(quat):
    """Ensure a quaternion is unit length (length ~= 1.0).

    The quaternion is **not** changed in place.

    :param numpy.array quat: The quaternion to normalize.
    :rtype: numpy.array
    :return: The normalized quaternion(s).
    """
    return vector_normalize(quat)

def normalise(quat):    # TODO: mark as deprecated
    """Ensure a quaternion is unit length (length ~= 1.0).

    The quaternion is **not** changed in place.

    :param numpy.array quat: The quaternion to normalize.
    :rtype: numpy.array
    :return: The normalized quaternion(s).
    """
    return vector_normalize(quat)

def rotation_angle(quat):
    """Calculates the rotation around the quaternion's axis.

    :param numpy.array quat: The quaternion.
    :rtype: float.
    :return: The quaternion's rotation about the its axis in radians.
    """
    # extract the W component
    thetaOver2 = np.arccos(quat[3])
    return thetaOver2 * 2.0

@all_parameters_as_numpy_arrays
def rotation_axis(quat):
    """Calculates the axis of the quaternion's rotation.

    :param numpy.array quat: The quaternion.
    :rtype: numpy.array.
    :return: The quaternion's rotation axis.
    """
    # extract W component
    sinThetaOver2Sq = 1.0 - (quat[3] ** 2)

    # check for zero before we sqrt
    if sinThetaOver2Sq <= 0.0:
        # identity quaternion or numerical imprecision.
        # return a valid vector
        # we'll treat -Z as the default
        return np.array([0.0, 0.0, -1.0], dtype=quat.dtype)

    oneOverSinThetaOver2 = 1.0 / np.sqrt(sinThetaOver2Sq)

    # we use the x,y,z values
    return np.array(
        [
            quat[0] * oneOverSinThetaOver2,
            quat[1] * oneOverSinThetaOver2,
            quat[2] * oneOverSinThetaOver2
        ],
        dtype=quat.dtype
    )

def dot(quat1, quat2):
    """Calculate the dot product of quaternions.

    This is the same as a vector dot product.

    :param numpy.array quat1: The first quaternion(s).
    :param numpy.array quat2: The second quaternion(s).
    :rtype: float, numpy.array
    :return: If a 1d array was passed, it will be a scalar.
        Otherwise the result will be an array of scalars with shape
        vec.ndim with the last dimension being size 1.
    """
    return vector_dot(quat1, quat2)

@all_parameters_as_numpy_arrays
def conjugate(quat):
    """Calculates a quaternion with the opposite rotation.

    :param numpy.array quat: The quaternion.
    :rtype: numpy.array.
    :return: A quaternion representing the conjugate.
    """

    # invert x,y,z and leave w as is
    return np.array(
        [
            -quat[0],
            -quat[1],
            -quat[2],
            quat[3]
        ],
        dtype=quat.dtype
    )

@parameters_as_numpy_arrays('quat')
def exp(quat):
    """Calculate the exponential of the quaternion

    :param numpy.array quat: The quaternion.
    :rtype: numpy.array.
    :return: The exponential of the quaternion
    """
    e = np.exp(quat[3])
    vector_norm = np.linalg.norm(quat[:3])

    if np.isclose(vector_norm, 0):
        return np.array(
            [0, 0, 0, e],
            dtype = quat.dtype
        )

    s = np.sin(vector_norm) / vector_norm
    return e * np.array(
        [
            quat[0] * s,
            quat[1] * s,
            quat[2] * s,
            np.cos(vector_norm),
        ],
        dtype = quat.dtype
    )

@parameters_as_numpy_arrays('quat')
def power(quat, exponent):
    """Multiplies the quaternion by the exponent.

    The quaternion is **not** changed in place.

    :param numpy.array quat: The quaternion.
    :param float scalar: The exponent.
    :rtype: numpy.array.
    :return: A quaternion representing the original quaternion
        to the specified power.
    """
    # check for identify quaternion
    if np.fabs(quat[3]) > 0.9999:
        # assert for the time being
        assert False
        print("rotation axis was identity")

        return quat

    alpha = np.arccos(quat[3])
    newAlpha = alpha * exponent
    multi = np.sin(newAlpha) / np.sin(alpha)

    return np.array(
        [
            quat[0] * multi,
            quat[1] * multi,
            quat[2] * multi,
            np.cos(newAlpha)
        ],
        dtype=quat.dtype
    )

def inverse(quat):
    """Calculates the inverse quaternion.

    The inverse of a quaternion is defined as
    the conjugate of the quaternion divided
    by the magnitude of the original quaternion.

    :param numpy.array quat: The quaternion to invert.
    :rtype: numpy.array.
    :return: The inverse of the quaternion.
    """
    return conjugate(quat) / length(quat)

@all_parameters_as_numpy_arrays
def negate(quat):
    """Calculates the negated quaternion.

    This is essentially the quaternion * -1.0.

    :param numpy.array quat: The quaternion.
    :rtype: numpy.array
    :return: The negated quaternion.
    """
    return quat * -1.0

def is_identity(quat):
    return np.allclose(quat, [0.,0.,0.,1.])

@all_parameters_as_numpy_arrays
def apply_to_vector(quat, vec):
    """Rotates a vector by a quaternion.

    :param numpy.array quat: The quaternion.
    :param numpy.array vec: The vector.
    :rtype: numpy.array
    :return: The vector rotated by the quaternion.
    :raise ValueError: raised if the vector is an unsupported size
    """
    def apply(quat, vec4):
        result = cross(quat, cross(vec4, conjugate(quat)))
        return result

    if vec.size == 3:
        # convert to vector4
        # ignore w component by setting it to 0.
        vec = np.array([vec[0], vec[1], vec[2], 0.0], dtype=vec.dtype)
        vec = apply(quat, vec)
        vec = vec[:3]
        return vec
    elif vec.size == 4:
        vec = apply(quat, vec)
        return vec
    else:
        raise ValueError("Vector size unsupported")

"""Represents a Quaternion rotation.

The Quaternion class provides a number of convenient functions and
conversions.
::

    import numpy as np
    from pyrr import Quaternion, Matrix33, Matrix44, Vector3, Vector4

    q = Quaternion()

    # explicit creation
    q = Quaternion.from_x_rotation(np.pi / 2.0)
    q = Quaternion.from_matrix(Matrix33.identity())
    q = Quaternion.from_matrix(Matrix44.identity())

    # inferred conversions
    q = Quaternion(Quaternion())
    q = Quaternion(Matrix33.identity())
    q = Quaternion(Matrix44.identity())

    # apply one quaternion to another
    q1 = Quaternion.from_y_rotation(np.pi / 2.0)
    q2 = Quaternion.from_x_rotation(np.pi / 2.0)
    q3 = q1 * q2

    # extract a matrix from the quaternion
    m33 = q3.matrix33
    m44 = q3.matrix44

    # convert from matrix back to quaternion
    q4 = Quaternion(m44)

    # rotate a quaternion by a matrix
    q = Quaternion() * Matrix33.identity()
    q = Quaternion() * Matrix44.identity()

    # apply quaternion to a vector
    v3 = Quaternion() * Vector3()
    v4 = Quaternion() * Vector4()

    # undo a rotation
    q = Quaternion.from_x_rotation(np.pi / 2.0)
    v = q * Vector3([1.,1.,1.])
    # ~q is the same as q.conjugate
    original = ~q * v
    assert np.allclose(original, v)

    # get the dot product of 2 Quaternions
    dot = Quaternion() | Quaternion.from_x_rotation(np.pi / 2.0)
"""


class Quaternion(BaseQuaternion):
    @classmethod
    def from_x_rotation(cls, theta, dtype=None):
        """Creates a new Quaternion with a rotation around the X-axis.
        """
        return cls(create_from_x_rotation(theta, dtype))

    @classmethod
    def from_y_rotation(cls, theta, dtype=None):
        """Creates a new Quaternion with a rotation around the Y-axis.
        """
        return cls(create_from_y_rotation(theta, dtype))

    @classmethod
    def from_z_rotation(cls, theta, dtype=None):
        """Creates a new Quaternion with a rotation around the Z-axis.
        """
        return cls(create_from_z_rotation(theta, dtype))

    @classmethod
    def from_axis_rotation(cls, axis, theta, dtype=None):
        """Creates a new Quaternion with a rotation around the specified axis.
        """
        return cls(create_from_axis_rotation(axis, theta, dtype))

    @classmethod
    def from_axis(cls, axis, dtype=None):
        """Creates a new Quaternion from an axis with angle magnitude.
        """
        return cls(create_from_axis(axis, dtype))

    @classmethod
    def from_matrix(cls, matrix, dtype=None):
        """Creates a Quaternion from the specified Matrix (Matrix33 or Matrix44).
        """
        return cls(create_from_matrix(matrix, dtype))

    @classmethod
    def from_eulers(cls, eulers, dtype=None):
        """Creates a Quaternion from the specified Euler angles.
        """
        return cls(create_from_eulers(eulers, dtype))

    @classmethod
    def from_inverse_of_eulers(cls, eulers, dtype=None):
        """Creates a Quaternion from the inverse of the specified Euler angles.
        """
        return cls(create_from_inverse_of_eulers(eulers, dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix33, matrix44
            if obj.shape in ((4,4,), (3,3,)) or isinstance(obj, (BaseMatrix3, BaseMatrix4)):
                obj = create_from_matrix(obj, dtype=dtype)
        else:
            obj = create(dtype=dtype)
        obj = obj.view(cls)
        return super(Quaternion, cls).__new__(cls, obj)

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

    ########################
    # Quaternions
    @dispatch((BaseQuaternion, np.ndarray, list))
    def __sub__(self, other):
        return Quaternion(super(Quaternion, self).__sub__(other))

    @dispatch((BaseQuaternion, list))
    def __mul__(self, other):
        return self.cross(other)

    @dispatch((BaseQuaternion, list))
    def __or__(self, other):
        return self.dot(other)

    def __invert__(self):
        return self.conjugate

    @dispatch((BaseQuaternion, np.ndarray, list))
    def __ne__(self, other):
        # For quaternions q and -q represent the same rotation
        return bool(np.any(super(Quaternion, self).__ne__(other)))\
               or bool(np.all(super(Quaternion, self).__eq__(-other)))

    @dispatch((BaseQuaternion, np.ndarray, list))
    def __eq__(self, other):
        # For quaternions q and -q represent the same rotation
        return bool(np.all(super(Quaternion, self).__eq__(other))) \
               or bool(np.all(super(Quaternion, self).__eq__(-other)))

    ########################
    # Matrices
    @dispatch(BaseMatrix)
    def __mul__(self, other):
        return self * Quaternion(other)

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(apply_to_vector(self, other))

    ########################
    # Methods and Properties
    @property
    def length(self):
        """Returns the length of this Quaternion.
        """
        return length(self)

    def normalize(self):
        """normalizes this Quaternion in-place.
        """
        self[:] = normalize(self)

    @property
    def normalized(self):
        """Returns a normalized version of this Quaternion as a new Quaternion.
        """
        return Quaternion(normalize(self))

    def normalise(self):    # TODO: mark as deprecated
        """normalizes this Quaternion in-place.
        """
        self[:] = normalize(self)

    @property
    def normalised(self):    # TODO: mark as deprecated
        """Returns a normalized version of this Quaternion as a new Quaternion.
        """
        return Quaternion(normalize(self))

    @property
    def angle(self):
        """Returns the angle around the axis of rotation of this Quaternion as a float.
        """
        return rotation_angle(self)

    @property
    def axis(self):
        """Returns the axis of rotation of this Quaternion as a Vector3.
        """
        return Vector3(rotation_axis(self))

    def cross(self, other):
        """Returns the cross of this Quaternion and another.

        This is the equivalent of combining Quaternion rotations (like Matrix multiplication).
        """
        return Quaternion(cross(self, other))

    def lerp(self, other, t):
        """Interpolates between quat1 and quat2 by t.
        The parameter t is clamped to the range [0, 1]
        """
        return Quaternion(lerp(self, other, t))

    def slerp(self, other, t):
        """Spherically interpolates between quat1 and quat2 by t.
        The parameter t is clamped to the range [0, 1]
        """
        return Quaternion(slerp(self, other, t))

    def dot(self, other):
        """Returns the dot of this Quaternion and another.
        """
        return dot(self, other)

    @property
    def conjugate(self):
        """Returns the conjugate of this Quaternion.

        This is a Quaternion with the opposite rotation.
        """
        return Quaternion(conjugate(self))

    @property
    def inverse(self):
        """Returns the inverse of this 
        """
        return Quaternion(inverse(self))

    def exp(self):
        """Returns a new Quaternion representing the exponentional of this Quaternion
        """
        return Quaternion(exp(self))

    def power(self, exponent):
        """Returns a new Quaternion representing this Quaternion to the power of the exponent.
        """
        return Quaternion(power(self, exponent))

    @property
    def negative(self):
        """Returns the negative of the Quaternion.
        """
        return Quaternion(negate(self))

    @property
    def is_identity(self):
        """Returns True if the Quaternion has no rotation (0.,0.,0.,1.).
        """
        return is_identity(self)