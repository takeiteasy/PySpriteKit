# spritekit/matrix.py
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

from .vector import *
from .quaternion import BaseQuaternion, Quaternion
from .quaternion import normalize as quaternion_normalize
from .quaternion import create_from_matrix as quaternion_create_from_matrix

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

@parameters_as_numpy_arrays('quat')
def matrix33_create_from_quaternion(quat, dtype=None):
    """Creates a matrix with the same rotation as a quaternion.

    :param quat: The quaternion to create the matrix from.
    :rtype: numpy.array
    :return: A matrix with shape (3,3) with the quaternion's rotation.
    """
    dtype = dtype or quat.dtype

    # the quaternion must be normalized
    if not np.isclose(np.linalg.norm(quat), 1.):
        quat = quaternion_normalize(quat)

    # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    sqw = qw**2
    sqx = qx**2
    sqy = qy**2
    sqz = qz**2
    qxy = qx * qy
    qzw = qz * qw
    qxz = qx * qz
    qyw = qy * qw
    qyz = qy * qz
    qxw = qx * qw

    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs
    m10 = 2.0 * (qxy + qzw) * invs
    m01 = 2.0 * (qxy - qzw) * invs
    m20 = 2.0 * (qxz - qyw) * invs
    m02 = 2.0 * (qxz + qyw) * invs
    m21 = 2.0 * (qyz + qxw) * invs
    m12 = 2.0 * (qyz - qxw) * invs

    return np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22],
    ], dtype=dtype)

@parameters_as_numpy_arrays('quat')
def matrix33_create_from_inverse_of_quaternion(quat, dtype=None):
    """Creates a matrix with the inverse rotation of a quaternion.

    :param numpy.array quat: The quaternion to make the matrix from (shape 4).
    :rtype: numpy.array
    :return: A matrix with shape (3,3) that respresents the inverse of
        the quaternion.
    """
    dtype = dtype or quat.dtype

    x, y, z, w = quat

    x2 = x**2
    y2 = y**2
    z2 = z**2
    wx = w * x
    wy = w * y
    xy = x * y
    wz = w * z
    xz = x * z
    yz = y * z

    return np.array(
        [
            # m1
            [
                # m11 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                1.0 - 2.0 * (y2 + z2),
                # m21 = 2.0 * (q.x * q.y + q.w * q.z)
                2.0 * (xy + wz),
                # m31 = 2.0 * (q.x * q.z - q.w * q.y)
                2.0 * (xz - wy),
            ],
            # m2
            [
                # m12 = 2.0 * (q.x * q.y - q.w * q.z)
                2.0 * (xy - wz),
                # m22 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
                1.0 - 2.0 * (x2 + z2),
                # m32 = 2.0 * (q.y * q.z + q.w * q.x)
                2.0 * (yz + wx),
            ],
            # m3
            [
                # m13 = 2.0 * ( q.x * q.z + q.w * q.y)
                2.0 * (xz + wy),
                # m23 = 2.0 * (q.y * q.z - q.w * q.x)
                2.0 * (yz - wx),
                # m33 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
                1.0 - 2.0 * (x2 + y2),
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
    def from_quaternion(cls, quat, dtype=None):
        """Creates a Matrix from a Quaternion.
        """
        return matrix33_create_from_quaternion(quat, dtype=dtype)

    @classmethod
    def from_inverse_of_quaternion(cls, quat, dtype=None):
        """Creates a Matrix from the inverse of the specified Quaternion.
        """
        return matrix33_create_from_inverse_of_quaternion(quat, dtype=dtype)

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

def matrix44_create_from_matrix33(mat, dtype=None):
    """Creates a Matrix4 from a Matrix3.

    The translation will be 0,0,0.

    :rtype: numpy.array
    :return: A matrix with shape (4,4) with the input matrix rotation.
    """
    mat4 = np.identity(4, dtype=dtype)
    mat4[0:3, 0:3] = mat
    return mat4

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

@parameters_as_numpy_arrays('quat')
def matrix44_create_from_quaternion(quat, dtype=None):
    """Creates a matrix with the same rotation as a quaternion.

    :param quat: The quaternion to create the matrix from.
    :rtype: numpy.array
    :return: A matrix with shape (4,4) with the quaternion's rotation.
    """
    dtype = dtype or quat.dtype
    # set to identity matrix
    # this will populate our extra rows for us
    mat = np.identity(4, dtype=dtype)
    # we'll use Matrix3 for our conversion
    mat[0:3, 0:3] = matrix33_create_from_quaternion(quat, dtype)
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

def matrix44_decompose(m):
    """Decomposes an affine transformation matrix into its scale, rotation and
    translation components.

    :param numpy.array m: A matrix.
    :return: tuple (scale, rotation, translation)
        numpy.array scale vector3
        numpy.array rotation quaternion
        numpy.array translation vector3
    """
    m = np.asarray(m)

    scale = np.linalg.norm(m[:3, :3], axis=1)

    det = np.linalg.det(m)
    if det < 0:
        scale[0] *= -1

    position = m[3, :3]

    rotation = m[:3, :3] * (1 / scale)[:, None]

    return scale, quaternion_create_from_matrix(rotation), position

class Matrix4(BaseObject, MatrixCore):
    @classmethod
    def from_eulers(cls, eulers, dtype=None):
        """Creates a Matrix from the specified Euler angles.
        """
        return m324(matrix33_create_from_eulers(eulers, dtype=dtype), dtype=dtype)
    
    @classmethod
    def from_quaternion(cls, quat, dtype=None):
        """Creates a Matrix from a Quaternion.
        """
        return m324(matrix33_create_from_quaternion(quat, dtype=dtype), dtype=dtype)
    
    @classmethod
    def from_inverse_of_quaternion(cls, quat, dtype=None):
        """Creates a Matrix from the inverse of the specified Quaternion.
        """
        return m324(matrix33_create_from_inverse_of_quaternion(quat, dtype=dtype), dtype=dtype)
    
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
    
    ########################
    # Creation
    @classmethod
    def from_matrix33(cls, matrix, dtype=None):
        """Creates a Matrix4 from a Matrix3.
        """
        return cls(matrix44_create_from_matrix33(matrix, dtype))

    @classmethod
    def perspective_projection(cls, fovy, aspect, near, far, dtype=None):
        """Creates a Matrix4 for use as a perspective projection matrix.
        """
        return cls(matrix44_create_perspective_projection(fovy, aspect, near, far, dtype))

    @classmethod
    def perspective_projection_bounds(cls, left, right, top, bottom, near, far, dtype=None):
        """Creates a Matrix4 for use as a perspective projection matrix.
        """
        return cls(matrix44_create_perspective_projection_from_bounds(left, right, top, bottom, near, far, dtype))

    @classmethod
    def orthogonal_projection(cls, left, right, top, bottom, near, far, dtype=None):
        """Creates a Matrix4 for use as an orthogonal projection matrix.
        """
        return cls(matrix44_create_orthogonal_projection(left, right, top, bottom, near, far, dtype))

    @classmethod
    def look_at(cls, eye, target, up, dtype=None):
        """Creates a Matrix4 for use as a lookAt matrix.
        """
        return cls(matrix44_create_look_at(eye, target, up, dtype))

    @classmethod
    def from_translation(cls, translation, dtype=None):
        """Creates a Matrix4 from the specified translation.
        """
        return cls(matrix44_create_from_translation(translation, dtype=dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix33
            if obj.shape == (3,3) or isinstance(obj, Matrix3):
                obj = matrix44_create_from_matrix33(obj, dtype=dtype)
            # quaternion
            elif obj.shape == (4,) or isinstance(obj, Quaternion):
                obj = matrix44_create_from_quaternion(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Matrix4, cls).__new__(cls, obj)

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

    def __invert__(self):
        return self.inverse

    ########################
    # Matrices
    @dispatch((BaseMatrix, np.ndarray, list))
    def __add__(self, other):
        return Matrix4(super(Matrix4, self).__add__(Matrix4(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __sub__(self, other):
        return Matrix4(super(Matrix4, self).__sub__(Matrix4(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __mul__(self, other):
        return Matrix4(np.dot(self, other))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Matrix4, self).__ne__(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Matrix4, self).__eq__(other)))

    ########################
    # Quaternions
    @dispatch(BaseQuaternion)
    def __mul__(self, other):
        m = other.matrix44
        return self * m

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(matrix44_apply_to_vector(self, other))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Matrix4(super(Matrix4, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Matrix4(super(Matrix4, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Matrix4(super(Matrix4, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Matrix4(super(Matrix4, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Matrix4(super(Matrix4, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def matrix33(self):
        """Returns a Matrix3 representing this matrix.
        """
        return Matrix3(self)

    @property
    def matrix44(self):
        """Returns the Matrix4.

        This can be handy if you're not sure what type of Matrix class you have
        but require a Matrix4.
        """
        return self

    @property
    def quaternion(self):
        """Returns a Quaternion representing this matrix.
        """
        return Quaternion(self)

    def decompose(self):
        """Decomposes an affine transformation matrix into its scale, rotation and
        translation components.

        :param numpy.array m: A matrix.
        :return: tuple (scale, rotation, translation)
            Vector3 scale
            Quaternion rotation
            Vector3 translation
        """
        scale, rotate, translate = matrix44_decompose(self)
        return Vector3(scale), Quaternion(rotate), Vector3(translate)

class Matrix(Matrix4):
    pass

@parameters_as_numpy_arrays('vec')
def matrix33_apply_to_vector(mat, vec):
    """Apply a matrix to a vector.

    The matrix's rotation are applied to the vector.
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
        return np.dot(vec, mat)
    else:
        raise ValueError("Vector size unsupported")

def matrix33_create_from_matrix44(mat, dtype=None):
    """Creates a Matrix33 from a Matrix44.

    :rtype: numpy.array
    :return: A matrix with shape (3,3) with the input matrix rotation.
    """
    mat = np.asarray(mat)
    return np.array(mat[0:3,0:3], dtype=dtype)

class Matrix3(BaseObject, MatrixCore):
    ########################
    # Creation
    @classmethod
    def from_matrix44(cls, matrix, dtype=None):
        """Creates a Matrix33 from a Matrix44.

        The Matrix44 translation will be lost.
        """
        return cls(matrix33_create_from_matrix44(matrix, dtype))

    def __new__(cls, value=None, dtype=None):
        if value is not None:
            obj = value
            if not isinstance(value, np.ndarray):
                obj = np.array(value, dtype=dtype)

            # matrix44
            if obj.shape == (4,4) or isinstance(obj, Matrix4):
                obj = matrix33_create_from_matrix44(obj, dtype=dtype)
            # quaternion
            elif obj.shape == (4,) or isinstance(obj, Quaternion):
                obj = matrix33_create_from_quaternion(obj, dtype=dtype)
        else:
            obj = np.zeros(cls._shape, dtype=dtype)
        obj = obj.view(cls)
        return super(Matrix3, cls).__new__(cls, obj)

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

    def __invert__(self):
        return self.inverse

    ########################
    # Matrices
    @dispatch((BaseMatrix, np.ndarray, list))
    def __add__(self, other):
        return Matrix3(super(Matrix3, self).__add__(Matrix3(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __sub__(self, other):
        return Matrix3(super(Matrix3, self).__sub__(Matrix3(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __mul__(self, other):
        return Matrix3(np.dot(other, self))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __ne__(self, other):
        return bool(np.any(super(Matrix3, self).__ne__(other)))

    @dispatch((BaseMatrix, np.ndarray, list))
    def __eq__(self, other):
        return bool(np.all(super(Matrix3, self).__eq__(other)))

    ########################
    # Quaternions
    @dispatch(BaseQuaternion)
    def __mul__(self, other):
        m = other.matrix33
        return self * m

    ########################
    # Vectors
    @dispatch(BaseVector)
    def __mul__(self, other):
        return type(other)(matrix33_apply_to_vector(self, other))

    ########################
    # Number
    @dispatch((Number, np.number))
    def __add__(self, other):
        return Matrix3(super(Matrix3, self).__add__(other))

    @dispatch((Number, np.number))
    def __sub__(self, other):
        return Matrix3(super(Matrix3, self).__sub__(other))

    @dispatch((Number, np.number))
    def __mul__(self, other):
        return Matrix3(super(Matrix3, self).__mul__(other))

    @dispatch((Number, np.number))
    def __truediv__(self, other):
        return Matrix3(super(Matrix3, self).__truediv__(other))

    @dispatch((Number, np.number))
    def __div__(self, other):
        return Matrix3(super(Matrix3, self).__div__(other))

    ########################
    # Methods and Properties
    @property
    def matrix33(self):
        """Returns the Matrix33.

        This can be handy if you're not sure what type of Matrix class you have
        but require a Matrix33.
        """
        return self

    @property
    def matrix44(self):
        """Returns a Matrix44 representing this matrix.
        """
        return Matrix4(self)

    @property
    def quaternion(self):
        """Returns a Quaternion representing this matrix.
        """
        return Quaternion(self)