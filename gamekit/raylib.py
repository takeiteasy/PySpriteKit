import pyray as r
from raylib import *
from raylib.colors import *
import os
import pathlib

__PATH = pathlib.Path(__file__).parent

__camera = None

__data_dir = ""

def _gen_file_paths(name, extensions, folders):
    paths = []
    for folder in folders:
        for ext in extensions:
            paths.append(folder + os.path.sep + name + ext)
            paths.append(__data_dir + os.path.sep + folder + os.path.sep + name + ext)
            paths.append(str(__PATH / folder / name) + ext)
    return paths

def find_file(name, extensions, folders):
    for file in _gen_file_paths(name, extensions, folders):
        print("trying ",file)
        if os.path.isfile(file):
            return file
    raise Exception(f"file {file} does not exist")

def Texture(file):
    texture = r.load_texture(find_file(file, ['.png', '.jpg', ''], ['.', 'data/images', 'images']))
    return texture

class Vector(list):
    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    @property
    def w(self):
        return self[3]

    @w.setter
    def w(self, value):
        self[3] = value

def _fix_key(kname):
    # return is a reserved word, so alias enter to return
    #if kname == 'enter':
    #    kname = 'return'
    kname = kname.upper()
    if not kname.startswith("KEY_"):
        kname = "KEY_" + kname
    return kname


class Keyboard:
    """
    Handles input from keyboard
    """
    def __getattr__(self, kname):
        f = _fix_key(kname)
        return rl.IsKeyDown(getattr(rl, f))


    def key_down(self, kname):
        """
        Test if key is currently down
        """
        f = _fix_key(kname)
        return IsKeyDown(getattr(rl, f))

    def key_pressed(self, kname):
        """
        Test if key was pressed recently
        """
        f = _fix_key(kname)
        return IsKeyPressed(getattr(rl, f))


class Keys:
    def __getattr__(self, kname):
        k = getattr(rl, _fix_key(kname))
        print(k)
        return k
    
class Gamepad:
    """
    Handles input from gamepads
    """
    def __init__(self, id):
        self.id = id

    def test(self):
        if r.is_gamepad_available(self.id):
            print("Detected gamepad", self.id, ffi.string(r.get_gamepad_name(self.id)))

    @property
    def up(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_LEFT_FACE_UP)

    @property
    def down(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_LEFT_FACE_DOWN)

    @property
    def left(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_LEFT_FACE_LEFT)

    @property
    def right(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_LEFT_FACE_RIGHT)

    @property
    def y(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_RIGHT_FACE_UP)

    @property
    def a(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_RIGHT_FACE_DOWN)

    @property
    def x(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_RIGHT_FACE_LEFT)

    @property
    def b(self):
        return r.is_gamepad_button_down(self.id, rl.GAMEPAD_BUTTON_RIGHT_FACE_RIGHT)

    @property
    def left_stick(self):
        return Vector([r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_LEFT_X),
                       r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_LEFT_Y)])

    @property
    def right_stick(self):
        return Vector([r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_RIGHT_X),
                       r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_RIGHT_Y)])

class Mouse:
    """
    Handles input from mouse
    """
    def get_position_on_ground(self, ground_level):
        pos = r.get_mouse_position()
        ray = r.get_mouse_ray(pos, __camera[0])
        rayhit = r.get_collision_ray_ground(ray, ground_level)
        return Vector([rayhit.position.x, rayhit.position.y, rayhit.position.z])

    @property
    def ground_position(self):
        return self.get_position_on_ground(0)

    @property
    def left_button(self):
        return r.is_mouse_button_down(rl.MOUSE_LEFT_BUTTON)

    @property
    def right_button(self):
        return r.is_mouse_button_down(rl.MOUSE_RIGHT_BUTTON)

    @property
    def middle_button(self):
        return r.is_mouse_button_down(rl.MOUSE_MIDDLE_BUTTON)

    @property
    def clicked(self):
        return r.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON)

    def check_collision(self, actor):
        if not actor.loaded:
            actor.load_data()
        pos = r.get_mouse_position()
        ray = r.get_mouse_ray(pos, __camera[0])
        return r.check_collision_ray_box(ray, actor.calc_bounding_box())