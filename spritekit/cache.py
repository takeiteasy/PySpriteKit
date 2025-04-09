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
import moderngl
from raudio import Wave, Music
from PIL import Image

__cache__ = {}
__SKPATH__ = pathlib.Path(__file__).parent
__SKDATA__ = "assets"

def _generate_paths(name, root, extension):
    return root + os.path.sep + name + extension, __SKDATA__ + os.path.sep + root + os.path.sep + name + extension, str(__SKPATH__ / root / name) + extension

def _find_file(file_path, folder_name, extensions):
    if os.path.isfile(file_path):
        return file_path
    _, ext = os.path.splitext(file_path)
    if ext not in extensions:
        raise RuntimeError(f"file {file_path} has invalid extension {ext}, supported extensions: {', '.join(extensions)}")
    else:
        extensions = ext
    folders = ['.', f"assets/{folder_name}", folder_name]
    paths = []
    for folder in folders:
        if isinstance(extensions, list):
            for extension in extensions:
                paths.extend(_generate_paths(file_path, folder, extension))
        else:
            paths.extend(_generate_paths(file_path, folder, ""))
    found = list(set([os.path.abspath(p) for p in paths if os.path.isfile(p)]))
    match len(found):
        case 0:
            raise RuntimeError(f"file {file_path} not found")
        case 1:
            return found[0]
        case _:
            raise RuntimeError(f"file {file_path} has multiple matches: {found}")

__image_extensions__ = ['.blp', '.bmp', '.dib', '.bufr', '.cur', '.pcx', '.dcx', '.dds', '.ps', '.eps', '.fit', '.fits', '.fli', '.flc', '.ftc', '.ftu', '.gbr', '.gif', '.grib', '.h5', '.hdf', '.png', '.apng', '.jp2', '.j2k', '.jpc', '.jpf', '.jpx', '.j2c', '.icns', '.ico', '.im', '.iim', '.jfif', '.jpe', '.jpg', '.jpeg', '.mpg', '.mpeg', '.tif', '.tiff', '.mpo', '.msp', '.palm', '.pcd', '.pdf', '.pxr', '.pbm', '.pgm', '.ppm', '.pnm', '.pfm', '.psd', '.qoi', '.bw', '.rgb', '.rgba', '.sgi', '.ras', '.tga', '.icb', '.vda', '.vst', '.webp', '.wmf', '.emf', '.xbm', '.xpm']
__audio_extensions__ = ['.wav', '.mp3', '.ogg', '.flac', '.xm', '.mod', '.qoa']

def load_image(name, flip=True):
    pass

def unload_cache():
    for v in __cache__.values():
        del v
    __cache__.clear()

__all__ = ["load_image", "load_texture", "load_audio", "load_music", "unload_cache"]