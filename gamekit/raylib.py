import pyray as r
import raylib as rl
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

__image_extensions = ['.png', '.bmp', '.tga', '.jpg', '.jpeg', '.gif', '.qoi', '.psd', '.dds', '.hdr', '.ktx', '.astc', '.pkm', '.pvr']
__model_extensions = ['.obj', '.glb', '.gltf', '.iqm', '.vox', '.m3d']
__vshader_extensions = ['.vs.glsl', '.vsh', '.vert']
__fshader_extensions = ['.fs.glsl', '.fsh', '.frag']

def _file_locations(name):
    return ['.', f"data/{name}", name]

def Image(file):
    return r.load_image(find_file(file, __image_extensions, _file_locations('images')))

def Texture(file):
    return r.load_texture(find_file(file, __image_extensions, _file_locations('images')))

def Shader(vertex_file, fragment_file):
    return r.load_shader(find_file(vertex_file, __vshader_extensions, _file_locations('shaders')),
                         find_file(fragment_file, __fshader_extensions, _file_locations('shaders')))

def Model(file):
    return r.load_model(find_file(file, __model_extensions, _file_locations('models')))

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
    @classmethod
    def __getattr__(cls, kname):
        return rl.IsKeyDown(getattr(rl, _fix_key(kname)))


    @classmethod
    def key_down(cls, kname):
        """
        Test if key is currently down
        """
        return rl.IsKeyDown(getattr(rl, _fix_key(kname)))

    @classmethod
    def key_pressed(cls, kname):
        """
        Test if key was pressed recently
        """
        return rl.IsKeyPressed(getattr(rl, _fix_key(kname)))


class Keys:
    def __getattr__(self, kname):
        return getattr(rl, _fix_key(kname))
    
class Gamepad:
    """
    Handles input from gamepads
    """
    def __init__(self, id):
        self.id = id

    def test(self):
        if r.is_gamepad_available(self.id):
            print("Detected gamepad", self.id, rl.ffi.string(r.get_gamepad_name(self.id)))

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
        return r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_LEFT_X), r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_LEFT_Y)

    @property
    def right_stick(self):
        return r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_RIGHT_X), r.get_gamepad_axis_movement(self.id, rl.GAMEPAD_AXIS_RIGHT_Y)

class Mouse:
    """
    Handles input from mouse
    """
    @classmethod
    def get_position_on_ground(cls, ground_level):
        pos = r.get_mouse_position()
        ray = r.get_mouse_ray(pos, __camera[0])
        rayhit = r.get_collision_ray_ground(ray, ground_level)
        return rayhit.position.x, rayhit.position.y, rayhit.position.z

    @classmethod
    def ground_position(cls, ground_level):
        return cls.get_position_on_ground(ground_level)

    @classmethod
    def left_button(cls):
        return r.is_mouse_button_down(rl.MOUSE_LEFT_BUTTON)

    @classmethod
    def right_button(cls):
        return r.is_mouse_button_down(rl.MOUSE_RIGHT_BUTTON)

    @classmethod
    def middle_button(cls):
        return r.is_mouse_button_down(rl.MOUSE_MIDDLE_BUTTON)

    @classmethod
    def clicked(cls):
        return r.is_mouse_button_pressed(rl.MOUSE_LEFT_BUTTON)

    @classmethod
    def check_collision(cls, actor):
        if not actor.loaded:
            actor.load_data()
        pos = r.get_mouse_position()
        ray = r.get_mouse_ray(pos, __camera[0])
        return r.check_collision_ray_box(ray, actor.calc_bounding_box())