# spritekit/cache.py
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

import os
import pathlib
import platform

import moderngl
from raudio import Wave, Music
from PIL import Image, ImageFont

__cache__ = {}
__SKPATH__ = pathlib.Path(__file__).parent
__data_path__ = "assets"

def _generate_paths(name, root, extension):
    return root + os.path.sep + name + extension, __data_path__ + os.path.sep + root + os.path.sep + name + extension, str(__SKPATH__ / root / name) + extension

def find_file(file_path, folder_names, extensions):
    if os.path.isfile(file_path):
        return file_path
    _, ext = os.path.splitext(file_path)
    if ext != '':
        if ext not in extensions:
            raise RuntimeError(f"File '{file_path}' has invalid extension {ext}, supported extensions: {', '.join(extensions)}")
        else:
            extensions = ext
    folders = [d for _dir in folder_names for d in ['.', f"{__data_path__}/{_dir}", _dir]]
    paths = []
    for folder in folders:
        if isinstance(extensions, list):
            for extension in extensions:
                paths.extend(_generate_paths(file_path, folder, extension))
        else:
            paths.extend(_generate_paths(file_path, folder, ""))
    found = list(set([os.path.abspath(p) for p in paths if os.path.isfile(p)]))
    if not found:
        raise RuntimeError(f"File '{file_path}' not found")
    return found[0]

def _check_cache(type_name, path):
    if not type_name in __cache__:
        __cache__[type_name] = {}
    return __cache__[type_name][path] if path in __cache__[type_name] else None

def _ensure_cached(type_name, folder_names, extensions):
    def decorator(func):
        def wrapper(path, **kwargs):
            found = find_file(path, folder_names, extensions)
            cached = _check_cache(type_name, found)
            if cached:
                return cached
            result = func(found, **kwargs)
            __cache__[type_name][found] = result
            return result
        return wrapper
    return decorator

__image_extensions__ = ['.blp', '.bmp', '.dib', '.bufr', '.cur', '.pcx', '.dcx', '.dds', '.ps', '.eps', '.fit', '.fits', '.fli', '.flc', '.ftc', '.ftu', '.gbr', '.gif', '.grib', '.h5', '.hdf', '.png', '.apng', '.jp2', '.j2k', '.jpc', '.jpf', '.jpx', '.j2c', '.icns', '.ico', '.im', '.iim', '.jfif', '.jpe', '.jpg', '.jpeg', '.mpg', '.mpeg', '.tif', '.tiff', '.mpo', '.msp', '.palm', '.pcd', '.pdf', '.pxr', '.pbm', '.pgm', '.ppm', '.pnm', '.pfm', '.psd', '.qoi', '.bw', '.rgb', '.rgba', '.sgi', '.ras', '.tga', '.icb', '.vda', '.vst', '.webp', '.wmf', '.emf', '.xbm', '.xpm']
__audio_extensions__ = ['.wav', '.mp3', '.ogg', '.flac', '.xm', '.mod', '.qoa']
__image_folders__ = ("textures", "images", "sprites")
__audio_folders__ = ("audio", "music", "sfx")

def _load_image(name, flip=True):
    img = Image.open(name).convert('RGBA')
    if flip:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img

@_ensure_cached(type_name="images",
                folder_names=__image_folders__,
                extensions=__image_extensions__)
def load_image(name, flip=True):
    return _load_image(name, flip)

@_ensure_cached(type_name="textures",
                folder_names=__image_folders__,
                extensions=__image_extensions__)
def load_texture(name, flip=True, sampler=(moderngl.NEAREST, moderngl.NEAREST), mipmaps=True, cache_image=False):
    ctx = moderngl.get_context()
    img = load_image(name, flip=flip) if cache_image else _load_image(name, flip)
    texture = ctx.texture(img.size, 4, img.tobytes())
    if mipmaps:
        texture.build_mipmaps()
    texture.filter = sampler
    return texture

@_ensure_cached(type_name="waves",
                folder_names=__audio_folders__,
                extensions=__audio_extensions__)
def load_wave(name):
    return Wave(name)

@_ensure_cached(type_name="music",
                folder_names=__audio_folders__,
                extensions=__audio_extensions__)
def load_music(name):
    return Music(name)

def _system_font_paths():
    def _clean(paths):
        return [path for path in paths if os.path.isdir(path)]
    match platform.system():
        case "Windows":
            return _clean([os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")])
        case "Darwin":
            return _clean(["/Library/Fonts",
                           "/System/Library/Fonts",
                           os.path.expanduser("~/Library/Fonts")])
        case "Linux":
            return _clean(["/usr/share/fonts",
                           "/usr/local/share/fonts",
                           os.path.expanduser("~/.fonts"),
                           os.path.expanduser("~/.local/share/fonts")])
        case _:
            return []

__pil_fonts__ = [".pil", ".pbm"]
__truetype_fonts__ = [".ttf", ".ttc", ".otf", ".pfa", ".pfb", ".cff", ".fnt", ".fon", ".bdf", ".pcf", ".woff", ".woff2", ".dfont"]
__font_folders__ = ("fonts", *_system_font_paths())
__truetype_cache__ = {}

def load_truetype_font(name, size):
    found = find_file(name, __font_folders__, __truetype_fonts__)
    if found in __truetype_cache__:
        if size in __truetype_cache__[found]:
            return __truetype_cache__[found][size]
    font = ImageFont.truetype(found, size)
    if not found in __truetype_cache__:
        __truetype_cache__[found] = {}
    __truetype_cache__[found][size] = font
    return font

@_ensure_cached(type_name="fonts",
                folder_names=__font_folders__,
                extensions=__pil_fonts__)
def load_bitmap_font(name):
    return ImageFont.load(name)

def load_font(name, size=None):
    if size is None:
        return load_bitmap_font(name)
    else:
        return load_truetype_font(name, size)

def set_data_path(path):
    global __data_path__
    __data_path__ = path

def clear_cache():
    __cache__.clear()
    __truetype_cache__.clear()

__all__ = ["find_file",
           "load_image",
           "load_texture",
           "load_wave",
           "load_music",
           "clear_cache",
           "set_data_path",
           "load_font",
           "load_bitmap_font",
           "load_truetype_font"]