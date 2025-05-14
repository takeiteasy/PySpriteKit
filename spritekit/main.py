# spritekit/main.py
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

from typing import Optional, override
from queue import Queue
import atexit

from . import _renderer as renderer
from . import _glfw as glfw
from .scene import Scene
from ._window import *

__window__: Optional[Window] = None

def _window_attrib(func):
    def wrapper(*args, **kwargs):
        if __window__ is None:
            raise RuntimeError("No window created")
        return func()
    return wrapper

@_window_attrib
def get_window():
    return __window__

@_window_attrib
def window_should_close():
    return __window__.should_close

@_window_attrib
def window_width():
    return __window__.width

@_window_attrib
def window_height():
    return __window__.height

@_window_attrib
def window_size():
    return __window__.size

class _ManagedWindow(Window):
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 title: str = "PySpriteKit",
                 versions: Optional[tuple[int, int, bool]] = None,
                 frame_limit: Optional[int | float] = None,
                 hints: Optional[dict] = None,
                 quit_key: Optional[Keys] = Keys.ESCAPE):
        if not bool(glfw.glfwInit()):
            raise glfw.NotInitializedError()
        atexit.register(glfw.glfwTerminate)
        if not versions:
            versions = (3, 3, True), (3, 2, True), (3, 1, False), (3, 0, False)
        else:
            if not isinstance(versions, list):
                versions = [versions]
        for vermaj, vermin, iscore in versions:
            try:
                Window.hint()
                Window.hint(context_version=(vermaj, vermin))
                if iscore:
                    Window.hint(forward_compat=True)
                    Window.hint(opengl_profile=Window.CORE_PROFILE)
                break
            except (glfw.PlatformError, glfw.VersionUnavailableError, ValueError) as e:
                iscore_str = 'CORE' if iscore else ''
                print("%s.%s %s: %s" % (vermaj, vermin, iscore_str, e))
        else:
            raise SystemExit("Proper OpenGL 3.x context not found")

        Window.__init__(self, width, height, title, hints=hints, frame_limit=frame_limit)
        self.set_key_callback(_ManagedWindow.key_callback)
        self.set_char_callback(_ManagedWindow.char_callback)
        self.set_scroll_callback(_ManagedWindow.scroll_callback)
        self.set_mouse_button_callback(_ManagedWindow.mouse_button_callback)
        self.set_cursor_enter_callback(_ManagedWindow.cursor_enter_callback)
        self.set_cursor_pos_callback(_ManagedWindow.cursor_pos_callback)
        self.set_window_size_callback(_ManagedWindow.window_size_callback)
        self.set_window_pos_callback(_ManagedWindow.window_pos_callback)
        self.set_window_close_callback(_ManagedWindow.window_close_callback)
        self.set_window_refresh_callback(_ManagedWindow.window_refresh_callback)
        self.set_window_focus_callback(_ManagedWindow.window_focus_callback)
        self.set_window_iconify_callback(_ManagedWindow.window_iconify_callback)
        self.set_framebuffer_size_callback(_ManagedWindow.framebuffer_size_callback)
        self._events = Queue()
        self._quit_key = quit_key

    def events(self):
        while not self._events.empty():
            event = self._events.get()
            yield event

    def all_events(self):
        return list(self._events.queue)

    @override
    def swap_buffers(self):
        glfw.glfwSwapBuffers(self.handle)
        self._events = Queue()

    def __add_event(self, event: EventType):
        self._events.put(event)

    def key_callback(self, key, scancode, action, mods):
        if self._quit_key is not None and key == self._quit_key and action == glfw.GLFW_PRESS:
            window_close()
        self.__add_event(KeyEvent(key=key,
                                  scancode=scancode,
                                  action=action,
                                  mods=mods))

    def char_callback(self, char):
        self.__add_event(CharEvent(char=char))

    def scroll_callback(self, off_x, off_y):
        self.__add_event(ScrollEvent(dx=off_x,
                                     dy=off_y))

    def mouse_button_callback(self, button, action, mods):
        self.__add_event(MouseButtonEvent(button=button,
                                          action=action,
                                          mods=mods))

    def cursor_enter_callback(self, status):
        self.__add_event(CursorEnterEvent(status=status))

    def cursor_pos_callback(self, pos_x, pos_y):
        self.__add_event(CursorPosEvent(x=pos_x,
                                        y=pos_y))

    def window_size_callback(self, wsz_w, wsz_h):
        self.__add_event(WindowSizeEvent(width=wsz_w,
                                         height=wsz_h))

    def window_pos_callback(self, pos_x, pos_y):
        self.__add_event(WindowPosEvent(x=pos_x,
                                        y=pos_y))

    def window_close_callback(self):
        self.__add_event(WindowCloseEvent())

    def window_refresh_callback(self):
        self.__add_event(WindowRefreshEvent())

    def window_focus_callback(self, status):
        self.__add_event(WindowFocusEvent(status=status))

    def window_iconify_callback(self, status):
        self.__add_event(WindowIconifyEvent(status=status))

    def framebuffer_size_callback(self, fbs_x, fbs_y):
        self.__add_event(FrameBufferSizeEvent(width=fbs_x,
                                              height=fbs_y))

__scene__: Optional[Scene] = None
__next_scene__ = None

def _load_scene(cls):
    global __scene__
    if __scene__ is not None:
        __scene__.exit()
    __scene__ = cls()
    __scene__.enter()

def load_scene(cls):
    global __next_scene__
    __next_scene__ = cls

def main(cls):
    global __scene__, __next_scene__, __window__
    __window__ = _ManagedWindow(*cls.window_size, cls.window_title, hints=cls.window_hints, frame_limit=cls.frame_limit)
    renderer.init()
    _load_scene(cls)
    while not __window__.should_close:
        __window__.poll_events()
        for e in __window__.events():
            __scene__.event(e)
        __scene__.step(__window__.delta_time)
        __scene__.draw()
        renderer.flush()
        __window__.swap_buffers()
        if __next_scene__ is not None:
            _load_scene(__next_scene__)
            __next_scene__ = None
    return cls

@_window_attrib
def window_close():
    __window__.should_close = True

__all__ = ['main', 'window_close', 'load_scene', "get_window",
           "window_should_close", "window_width", "window_height", "window_size"]