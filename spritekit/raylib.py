# spritekit/raylib.py
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

import pyray as r
import raylib as rl
from pyglsl import VertexStage, FragmentStage
import os
import pathlib

__SKPATH__ = pathlib.Path(__file__).parent
__SKDATA__ = "assets"

__image_extensions = ['.png', '.bmp', '.tga', '.jpg', '.jpeg', '.gif', '.qoi', '.psd', '.dds', '.hdr', '.ktx', '.astc', '.pkm', '.pvr']
__model_extensions = ['.obj', '.glb', '.gltf', '.iqm', '.vox', '.m3d']
__vshader_extensions = ['.vs.glsl', '.vsh', '.vert']
__fshader_extensions = ['.fs.glsl', '.fsh', '.frag']
__sound_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.xm', '.mod', '.qoa']
__font_extensions = ['.ttf', '.otf', '.fnt']
__cache = {}

def _gen_file_paths(name, extensions, folders):
    paths = []
    for folder in folders:
        for ext in extensions:
            paths.append(folder + os.path.sep + name + ext)
            paths.append(__SKDATA__ + os.path.sep + folder + os.path.sep + name + ext)
            paths.append(str(__SKPATH__ / folder / name) + ext)
    return paths
    
def find_file(name, extensions, folders):
    _, ext = os.path.splitext(name)
    if ext and ext in extensions:
        if os.path.isfile(name):
            return name
    for file in _gen_file_paths(name, extensions, folders):
        print("trying ",file)
        if os.path.isfile(file):
            return file
    raise Exception(f"file {file} does not exist")

def cache_result(func):
    def wrapper(*args, **kwargs):
        key = args[0]  # Get the file path argument
        if key in __cache:
            return __cache[key]
        result = func(*args, **kwargs)
        __cache[key] = result
        return result
    return wrapper

def _unload_asset(key: str):
    k = __cache[k]
    if isinstance(k, r.Model):
        r.unload_model(k)
    elif isinstance(k, r.Texture):
        r.unload_texture(k)
    elif isinstance(k, r.Image):
        r.unload_image(k)
    elif isinstance(k, r.Font):
        r.unload_font(k)
    elif isinstance(k, r.Wave):
        r.unload_wave(k)
    elif isinstance(k, r.Sound):
        r.unload_sound(k)
    elif isinstance(k, r.Music):
        r.unload_music_stream(k)
    else:
        try:
            del k
        except:
            pass
    __cache.pop(key)

def unload_cache(key: str = None):
    if key:
        _unload_asset(key)
    else:
        for key in list(__cache.keys()):
            _unload_asset(key)

def _file_locations(name):
    return ['.', f"assets/{name}", name]

@cache_result
def Image(file: str):
    return r.load_image(find_file(file, __image_extensions, _file_locations('textures')))

@cache_result
def Texture(file: str):
    return r.load_texture(find_file(file, __image_extensions, _file_locations('textures')))

def Shader(vertex_file: str, fragment_file: str):
    return r.load_shader(find_file(vertex_file, __vshader_extensions, _file_locations('shaders')),
                         find_file(fragment_file, __fshader_extensions, _file_locations('shaders')))

def CompileShader(vertex: str | VertexStage, fragment: str | FragmentStage):
    return r.load_shader_from_memory(vertex.compile() if isinstance(vertex, VertexStage) else vertex,
                                     fragment.compile() if isinstance(fragment, FragmentStage) else fragment)

@cache_result
def Model(file: str):
    return r.load_model(find_file(file, __model_extensions, _file_locations('models')))

@cache_result
def Wave(file: str):
    return r.load_wave(find_file(file, __sound_extensions, _file_locations('audio')))

@cache_result
def Sound(file):
    if isinstance(file, r.Wave):
        return r.load_sound_from_wave(file)
    else:
        def _load(file):
            return r.load_sound(find_file(file, __sound_extensions, _file_locations('audio')))
        return _load(file)

@cache_result
def Music(file: str):
    return r.load_music_stream(find_file(file, __sound_extensions, _file_locations('audio')))

@cache_result
def Font(file: str):
    return r.load_font(find_file(file, __font_extensions, _file_locations('fonts')))

def _fix_key(kname):
    # return is a reserved word, so alias enter to return
    #if kname == 'enter':
    #    kname = 'return'
    kname = kname.upper()
    if not kname.startswith("KEY_"):
        kname = "KEY_" + kname
    return kname

class IFuckingHateYouPython:
    def __init__(self, stupidfuckingidiotfuction):
        self.stupidfuckingidiotfuction = stupidfuckingidiotfuction

    def __getattr__(self, kname):
        return getattr(rl, self.stupidfuckingidiotfuction(kname))

Keys = IFuckingHateYouPython(stupidfuckingidiotfuction=_fix_key)

def _fix_flag(kname):
    kname = kname.upper()
    if not kname.startswith("FLAG_"):
        kname = "FLAG_" + kname
    return kname

Flags = IFuckingHateYouPython(stupidfuckingidiotfuction=_fix_flag)

class Keyboard:
    """
    Handles input from keyboard
    """
    @classmethod
    def _fix_kname(cls, kname):
        return getattr(rl, _fix_key(kname) if isinstance(kname, str) else kname)

    @classmethod
    def __getattr__(cls, kname: str | r.KeyboardKey | int):
        return rl.IsKeyDown(getattr(rl, _fix_key(kname) if isinstance(kname, str) else kname))

    @classmethod
    def key_down(cls, kname: str | r.KeyboardKey | int):
        """
        Test if key is currently down
        """
        return rl.IsKeyDown(cls._fix_kname(kname) if isinstance(kname, str) else kname)

    @classmethod
    def key_pressed(cls, kname: str | r.KeyboardKey | int):
        """
        Test if key was pressed recently
        """
        return rl.IsKeyPressed(cls._fix_kname(kname) if isinstance(kname, str) else kname)
    
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

def Color(_r: int | float, g: int | float, b: int | float, a: int | float = 255):
    return r.Color(*[x if isinstance(x, int) else int(x * 255.) for x in [_r, g, b, a]])

def Rectangle(x: int | float, y: int | float, width: int | float, height: int | float):
    return r.Rectangle(int(x), int(y), int(width), int(height))