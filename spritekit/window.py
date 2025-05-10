# spritekit/window.py
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
# pyglfw
# MIT License
# 
# Copyright (C) 2013 Roman Valov
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from threading import local
from typing import Optional, Union, Dict
from dataclasses import dataclass
import atexit

from . import _glfw as glfw

class EventType:
    pass

@dataclass
class KeyEvent(EventType):
    key: int
    scancode: int
    action: int
    mods: int

@dataclass
class CharEvent(EventType):
    char: int

@dataclass
class ScrollEvent(EventType):
    dx: float
    dy: float

@dataclass
class MouseButtonEvent(EventType):
    button: int
    action: int
    mods: int

@dataclass
class CursorEnterEvent(EventType):
    status: bool

@dataclass
class CursorPosEvent(EventType):
    x: int
    y: int

@dataclass
class WindowSizeEvent(EventType):
    width: int
    height: int

@dataclass
class WindowPosEvent(EventType):
    x: int
    y: int

@dataclass
class WindowCloseEvent(EventType):
    pass

@dataclass
class WindowRefreshEvent(EventType):
    pass

@dataclass
class WindowFocusEvent(EventType):
    status: bool

@dataclass
class WindowIconifyEvent(EventType):
    status: bool

@dataclass
class FrameBufferSizeEvent(EventType):
    width: int
    height: int


if bytes is str:
    # noinspection PyUnresolvedReferences
    _unichr = unichr
    # noinspection PyUnresolvedReferences
    _unistr = unicode
else:
    _unichr = chr
    _unistr = str

def _utf(obj):
    if bytes is not str:
        obj = obj.encode()
    return obj

def _str(obj):
    if bytes is not str:
        obj = obj.decode()
    return obj

class WindowType:
    pass

class _HintsBase:
    _hint_map_ = {
        'resizable':           glfw.GLFW_RESIZABLE,
        'visible':             glfw.GLFW_VISIBLE,
        'decorated':           glfw.GLFW_DECORATED,
        'red_bits':            glfw.GLFW_RED_BITS,
        'green_bits':          glfw.GLFW_GREEN_BITS,
        'blue_bits':           glfw.GLFW_BLUE_BITS,
        'alpha_bits':          glfw.GLFW_ALPHA_BITS,
        'depth_bits':          glfw.GLFW_DEPTH_BITS,
        'stencil_bits':        glfw.GLFW_STENCIL_BITS,
        'accum_red_bits':      glfw.GLFW_ACCUM_RED_BITS,
        'accum_green_bits':    glfw.GLFW_ACCUM_GREEN_BITS,
        'accum_blue_bits':     glfw.GLFW_ACCUM_BLUE_BITS,
        'accum_alpha_bits':    glfw.GLFW_ACCUM_ALPHA_BITS,
        'aux_buffers':         glfw.GLFW_AUX_BUFFERS,
        'samples':             glfw.GLFW_SAMPLES,
        'refresh_rate':        glfw.GLFW_REFRESH_RATE,
        'stereo':              glfw.GLFW_STEREO,
        'srgb_capable':        glfw.GLFW_SRGB_CAPABLE,
        'client_api':          glfw.GLFW_CLIENT_API,
        'context_ver_major':   glfw.GLFW_CONTEXT_VERSION_MAJOR,
        'context_ver_minor':   glfw.GLFW_CONTEXT_VERSION_MINOR,
        'context_robustness':  glfw.GLFW_CONTEXT_ROBUSTNESS,
        'debug_context':       glfw.GLFW_OPENGL_DEBUG_CONTEXT,
        'forward_compat':      glfw.GLFW_OPENGL_FORWARD_COMPAT,
        'opengl_profile':      glfw.GLFW_OPENGL_PROFILE,
    }

    _over_map_ = {
        'context_version':     (glfw.GLFW_CONTEXT_VERSION_MAJOR,
                                glfw.GLFW_CONTEXT_VERSION_MINOR,),

        'rgba_bits':           (glfw.GLFW_RED_BITS,
                                glfw.GLFW_GREEN_BITS,
                                glfw.GLFW_BLUE_BITS,
                                glfw.GLFW_ALPHA_BITS,),

        'rgba_accum_bits':     (glfw.GLFW_ACCUM_RED_BITS,
                                glfw.GLFW_ACCUM_GREEN_BITS,
                                glfw.GLFW_ACCUM_BLUE_BITS,
                                glfw.GLFW_ACCUM_ALPHA_BITS,),
    }

    def __init__(self, **kwargs):
        self._hints = {}

        for k, v in kwargs.items():
            is_hint = k in self.__class__._hint_map_
            is_over = k in self.__class__._over_map_
            if is_hint or is_over:
                setattr(self, k, v)

    def __getitem__(self, index):
        if index in self.__class__._hint_map_.values():
            return self._hints.get(index, None)
        else:
            raise TypeError()

    def __setitem__(self, index, value):
        if index in self.__class__._hint_map_.values():
            if value is None:
                if index in self._hints:
                    del self._hints[index]
            elif isinstance(value, int):
                self._hints[index] = value
        else:
            raise TypeError()

    def __delitem__(self, index):
        if index in self.__class__._hint_map_.values():
            if index in self._hints:
                del self._hints[index]
        else:
            raise TypeError()

def _hntprops_(hint_map, over_map):
    prop_map = {}

    def _hint_property(hint):
        def _get(self):
            return self[hint]

        def _set(self, value):
            self[hint] = value

        def _del(self):
            del self[hint]

        return property(_get, _set, _del)

    for prop, hint in hint_map.items():
        prop_map[prop] = _hint_property(hint)

    def _over_property(over):
        def _get(self):
            value = [self[hint] for hint in over]
            return tuple(value)

        def _set(self, value):
            for hint, v in zip(over, value):
                self[hint] = v

        def _del(self):
            for hint in over:
                del self[hint]

        return property(_get, _set, _del)

    for prop, over in over_map.items():
        prop_map[prop] = _over_property(over)

    return prop_map

Hints = type('Hints',
             (_HintsBase,),
             _hntprops_(_HintsBase._hint_map_, _HintsBase._over_map_))

class Mice:
    def __init__(self, handle):
        self.handle = handle
        self.ntotal = glfw.GLFW_MOUSE_BUTTON_LAST + 1

    def __len__(self):
        return (self.ntotal)

    def __getitem__(self, index):
        if isinstance(index, int):
            if (index < 0):
                index += self.ntotal
            elif (index >= self.ntotal):
                raise IndexError("Index %i is out of range" % index)

            return bool(glfw.glfwGetMouseButton(self.handle, index))
        elif isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.ntotal))]
        else:
            raise TypeError("Index %i is not supported" % index)

    LEFT = glfw.GLFW_MOUSE_BUTTON_LEFT

    @property
    def left(self):
        return self[glfw.GLFW_MOUSE_BUTTON_LEFT]

    RIGHT = glfw.GLFW_MOUSE_BUTTON_RIGHT

    @property
    def right(self):
        return self[glfw.GLFW_MOUSE_BUTTON_RIGHT]

    MIDDLE = glfw.GLFW_MOUSE_BUTTON_MIDDLE

    @property
    def middle(self):
        return self[glfw.GLFW_MOUSE_BUTTON_MIDDLE]

class _Keys:
    def __init__(self, handle):
        self.handle = handle

    def __getitem__(self, index):
        if isinstance(index, int):
            return bool(glfw.glfwGetKey(self.handle, index))

def _keyattrs_():
    _keyattribs_ = {}
    _key_prefix_ = 'GLFW_KEY_'
    _key_prelen_ = len(_key_prefix_)

    for name, item in vars(glfw).items():
        if name.startswith(_key_prefix_):
            _name_ = name[_key_prelen_:]
            if _name_[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                _name_ = 'NUM_' + _name_
            _name_ = _name_.upper()
            _prop_ = _name_.lower()

            _keyattribs_[_name_] = item
            if _prop_ == 'last' or _prop_ == 'unknown':
                continue
            # noinspection PyUnresolvedReferences
            _keyattribs_[_prop_] = property(lambda self, item=item: self[item])

    return _keyattribs_

Keys = type('Keys', (_Keys,), _keyattrs_())

class Joystick:
    def __init__(self, joyidx):
        self.joyidx = joyidx

    def __nonzero__(self):
        return bool(glfw.glfwJoystickPresent(self.joyidx))

    def __bool__(self):
        return bool(glfw.glfwJoystickPresent(self.joyidx))

    @property
    def name(self):
        return _str(glfw.glfwGetJoystickName(self.joyidx))

    @property
    def axes(self):
        return glfw.glfwGetJoystickAxes(self.joyidx)

    @property
    def buttons(self):
        return glfw.glfwGetJoystickButtons(self.joyidx)

def _monitor_obj(moni):
    monobj = super(Monitor, Monitor).__new__(Monitor)
    monobj.handle = moni.get_void_p()
    return monobj

class VideoMode:
    __slots__ = 'width', 'height', 'bits', 'refresh_rate'

    def __init__(self, vm):
        self.width = vm.width
        self.height = vm.height
        self.bits = (vm.redBits, vm.greenBits, vm.blueBits)
        self.refresh_rate = vm.refreshRate

class Monitor:
    _callback_ = None

    CONNECTED = glfw.GLFW_CONNECTED
    DISCONNECTED = glfw.GLFW_DISCONNECTED

    def __eq__(self, other):
        return self.handle.value == other.handle.value

    def __ne__(self, other):
        return not (self == other)

    @staticmethod
    def set_callback(callback):
        if not callback:
            Monitor._callback_ = None
        else:
            def wrap(handle, *args, **kwargs):
                callback(_monitor_obj(handle), *args, **kwargs)
            Monitor._callback_ = glfw.GLFWmonitorfun(wrap)
        glfw.glfwSetMonitorCallback(Monitor._callback_)

    def __init__(self):
        raise TypeError("Objects of this class cannot be created")

    @property
    def pos(self):
        return glfw.glfwGetMonitorPos(self.handle)

    @property
    def name(self):
        return _str(glfw.glfwGetMonitorName(self.handle))

    @property
    def physical_size(self):
        return glfw.glfwGetMonitorPhysicalSize(self.handle)

    @property
    def video_mode(self):
        return VideoMode(glfw.glfwGetVideoMode(self.handle))

    @property
    def video_modes(self):
        return [VideoMode(vm) for vm in glfw.glfwGetVideoModes(self.handle)]

    def set_gamma(self, gamma):
        glfw.glfwSetGamma(self.handle, gamma)

    @property
    def gamma_ramp(self):
        return glfw.glfwGetGammaRamp(self.handle)

    @gamma_ramp.setter
    def gamma_ramp(self, rgb_ramp):
        glfw.glfwSetGammaRamp(self.handle, rgb_ramp)

    @staticmethod
    def all():
        return [_monitor_obj(moni) for moni in glfw.glfwGetMonitors()]

    @staticmethod
    def primary():
        return _monitor_obj(glfw.glfwGetPrimaryMonitor())

class Window(WindowType):
    _instance_ = {}
    _contexts_ = local()

    def __init__(self, width: int, height: int, title: str,
                 monitor: Optional[Monitor] = None,
                 shared: Optional[WindowType] = None,
                 hints: Optional[Dict] = None,
                 callbacks: Optional[Dict] = None,
                 frame_limit: Optional[int | float] = None):
        mon_handle = monitor and monitor.handle or None
        shr_handle = shared and shared.handle or None
        win_handle = glfw.glfwCreateWindow(width, height, _utf(title), mon_handle, shr_handle)

        self.handle = win_handle.get_void_p()
        self.__class__._instance_[self.handle.value] = self
        self.make_current()

        self.mice = Mice(self.handle)
        self.keys = Keys(self.handle)

        if hints:
            self.__class__.hint(hints=hints)

        if callbacks:
            self.set_callbacks(**callbacks)
        
        self._delta_time = 0.
        self._frame_limit = None
        self.set_frame_limit(frame_limit)
        self.frame_prev_time = glfw.glfwGetTime()
        self.frame_current_time = self.frame_prev_time
        self.frame_count = 0
        self.frame_accum = 0

    def __enter__(self):
        if not hasattr(Window._contexts_, 'ctxstack'):
            Window._contexts_.ctxstack = []
        Window._contexts_.ctxstack += [self.find_current()]

        glfw.glfwMakeContextCurrent(self.handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not Window._contexts_.ctxstack:
            raise RuntimeError('Corrupted context stack')

        _ctx = Window._contexts_.ctxstack.pop()
        glfw.glfwMakeContextCurrent(_ctx and _ctx.handle or _ctx)
        return False
    
    @property
    def frame_limit(self):
        return self._frame_limit

    def set_frame_limit(self, limit: Optional[Union[str, int]]):
        self._frame_limit = limit
        self.frame_step = 0 if self._frame_limit is None else 1.0 / self._frame_limit
    
    @property
    def delta_time(self):
        return self._delta_time

    @classmethod
    def swap_current(cls, _ctx):
        if hasattr(Window._contexts_, 'ctxstack') and \
                Window._contexts_.ctxstack:
            raise RuntimeError('This function cannot be used inside `with`')
        glfw.glfwMakeContextCurrent(_ctx and _ctx.handle or _ctx)
        return _ctx

    def make_current(self):
        return self.swap_current(self)

    @classmethod
    def find_current(cls):
        find_handle = glfw.glfwGetCurrentContext().get_void_p()
        if bool(find_handle):
            return cls._instance_.get(find_handle.value)
        else:
            return None

    def close(self):
        glfw.glfwDestroyWindow(self.handle)

    @property
    def should_close(self):
        return bool(glfw.glfwWindowShouldClose(self.handle))

    @should_close.setter
    def should_close(self, flag):
        glfw.glfwSetWindowShouldClose(self.handle, flag)

    def swap_buffers(self):
        self.frame_prev_time = self.frame_current_time
        self.frame_current_time = glfw.glfwGetTime()
        self._delta_time = self.frame_current_time - self.frame_prev_time
        if self.frame_limit is not None:
            self.frame_accum += self._delta_time
            self.frame_count += 1
            if self.frame_accum >= 1.0:
                self.frame_accum -= 1.0
                self.frame_count = 0
            while glfw.glfwGetTime() < self.frame_current_time + self.frame_step:
                pass
        glfw.glfwSwapBuffers(self.handle)

    def swap_interval(self, interval):
        with self:
            glfw.glfwSwapInterval(interval)

    def set_title(self, title):
        glfw.glfwSetWindowTitle(self.handle, title)

    @property
    def framebuffer_size(self):
        return glfw.glfwGetFramebufferSize(self.handle)

    @property
    def pos(self):
        return glfw.glfwGetWindowPos(self.handle)

    @pos.setter
    def pos(self, x_y):
        glfw.glfwSetWindowPos(self.handle, *x_y)

    @property
    def size(self):
        return glfw.glfwGetWindowSize(self.handle)

    @size.setter
    def size(self, x_y):
        glfw.glfwSetWindowSize(self.handle, *x_y)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def iconify(self):
        glfw.glfwIconifyWindow(self.handle)

    def restore(self):
        glfw.glfwRestoreWindow(self.handle)

    def _get_attrib(self, attrib):
        return glfw.glfwGetWindowAttrib(self.handle, attrib)

    @property
    def iconified(self):
        return bool(self.set_get_attrib(glfw.GLFW_ICONIFIED))

    @iconified.setter
    def iconified(self, flag):
        if flag:
            self.iconify()
        else:
            self.restore()

    def hide(self):
        glfw.glfwHideWindow(self.handle)

    def show(self):
        glfw.glfwShowWindow(self.handle)

    @property
    def visible(self):
        return bool(self.set_get_attrib(glfw.GLFW_VISIBLE))

    @visible.setter
    def visible(self, flag):
        if flag:
            self.show()
        else:
            self.hide()

    @property
    def has_focus(self):
        return bool(self.set_get_attrib(glfw.GLFW_FOCUSED))

    @property
    def resizable(self):
        return bool(self.set_get_attrib(glfw.GLFW_RESIZABLE))

    @property
    def decorated(self):
        return bool(self.set_get_attrib(glfw.GLFW_DECORATED))

    @property
    def context_version(self):
        return (self.set_get_attrib(glfw.GLFW_CONTEXT_VERSION_MAJOR),
                self.set_get_attrib(glfw.GLFW_CONTEXT_VERSION_MINOR),
                self.set_get_attrib(glfw.GLFW_CONTEXT_REVISION))

    @property
    def debug_context(self):
        return bool(self.set_get_attrib(glfw.GLFW_OPENGL_DEBUG_CONTEXT))

    @property
    def forward_compat(self):
        return bool(self.set_get_attrib(glfw.GLFW_OPENGL_FORWARD_COMPAT))

    OPENGL_API = glfw.GLFW_OPENGL_API
    OPENGL_ES_API = glfw.GLFW_OPENGL_ES_API

    @property
    def client_api(self):
        return self.set_get_attrib(glfw.GLFW_CLIENT_API)

    CORE_PROFILE = glfw.GLFW_OPENGL_CORE_PROFILE
    COMPAT_PROFILE = glfw.GLFW_OPENGL_COMPAT_PROFILE
    ANY_PROFILE = glfw.GLFW_OPENGL_ANY_PROFILE

    @property
    def opengl_profile(self):
        return self.set_get_attrib(glfw.GLFW_OPENGL_PROFILE)

    NO_ROBUSTNESS = glfw.GLFW_NO_ROBUSTNESS
    NO_RESET_NOTIFICATION = glfw.GLFW_NO_RESET_NOTIFICATION
    LOSE_CONTEXT_ON_RESET = glfw.GLFW_LOSE_CONTEXT_ON_RESET

    @property
    def context_robustness(self):
        return self.set_get_attrib(glfw.GLFW_CONTEXT_ROBUSTNESS)

    @staticmethod
    def hint(hints=None, **kwargs):
        if hints and kwargs:
            raise ValueError("Hints should be passed via object or via kwargs")

        if not hints:
            hints = Hints(**kwargs)

        if not hints._hints:
            glfw.glfwDefaultWindowHints()

        for hint, value in hints._hints.items():
            glfw.glfwWindowHint(hint, value)

    @property
    def monitor(self):
        moni = glfw.glfwGetWindowMonitor(self.handle)
        if bool(moni):
            return _monitor_obj(moni)
        else:
            return None

    @property
    def clipboard(self):
        return _str(glfw.glfwGetClipboardString(self.handle))

    @clipboard.setter
    def clipboard(self, buffer):
        glfw.glfwSetClipboardString(self.handler, _utf(buffer))

    _cursor_modes_get = {
        glfw.GLFW_CURSOR_DISABLED: None,
        glfw.GLFW_CURSOR_HIDDEN: False,
        glfw.GLFW_CURSOR_NORMAL: True,
    }

    _cursor_modes_set = {
        None: glfw.GLFW_CURSOR_DISABLED,
        False: glfw.GLFW_CURSOR_HIDDEN,
        True: glfw.GLFW_CURSOR_NORMAL,
    }

    @property
    def cursor_mode(self):
        libapi_cm = glfw.glfwGetInputMode(self.handle, glfw.GLFW_CURSOR)
        return self.set_cursor_modes_get.get(libapi_cm, None)

    @cursor_mode.setter
    def cursor_mode(self, mode):
        pyglfw_cm = self.set_cursor_modes_set.get(mode, None)
        glfw.glfwSetInputMode(self.handle, glfw.GLFW_CURSOR, pyglfw_cm)

    @property
    def sticky_keys(self):
        return bool(glfw.glfwGetInputMode(self.handle, glfw.GLFW_STICKY_KEYS))

    @sticky_keys.setter
    def sticky_keys(self, flag):
        glfw.glfwSetInputMode(self.handle, glfw.GLFW_STICKY_KEYS, flag)

    @property
    def sticky_mice(self):
        return bool(glfw.glfwGetInputMode(self.handle,
                                     glfw.GLFW_STICKY_MOUSE_BUTTONS))

    @sticky_mice.setter
    def sticky_mice(self, flag):
        glfw.glfwSetInputMode(self.handle, glfw.GLFW_STICKY_MOUSE_BUTTONS, flag)

    @property
    def cursor_pos(self):
        return glfw.glfwGetCursorPos(self.handle)

    @cursor_pos.setter
    def cursor_pos(self, x_y):
        glfw.glfwSetCursorPos(self.handle, *x_y)

    PRESS = glfw.GLFW_PRESS
    RELEASE = glfw.GLFW_RELEASE
    REPEAT = glfw.GLFW_REPEAT

    MOD_SHIFT = glfw.GLFW_MOD_SHIFT
    MOD_CONTROL = glfw.GLFW_MOD_CONTROL
    MOD_ALT = glfw.GLFW_MOD_ALT
    MOD_SUPER = glfw.GLFW_MOD_SUPER

    @classmethod
    def _wcb(cls, functype, func):
        if not func:
            return None

        def wrap(handle, *args, **kwargs):
            window = cls._instance_.get(handle.get_void_p().value, None)
            func(window, *args, **kwargs)
        return functype(wrap)

    def set_key_callback(self, callback):
        self.set_key_callback = self._wcb(glfw.GLFWkeyfun, callback)
        glfw.glfwSetKeyCallback(self.handle, self.set_key_callback)

    def set_char_callback(self, callback):
        def wrap(self, char):
            char = _unichr(char)
            callback(self, char)
        self.set_char_callback = self._wcb(glfw.GLFWcharfun, wrap)
        glfw.glfwSetCharCallback(self.handle, self.set_char_callback)

    def set_scroll_callback(self, callback):
        self.set_scroll_callback = self._wcb(glfw.GLFWscrollfun, callback)
        glfw.glfwSetScrollCallback(self.handle, self.set_scroll_callback)

    def set_cursor_enter_callback(self, callback):
        def wrap(self, flag):
            flag = bool(flag)
            callback(self, flag)
        self.set_cursor_enter_callback = self._wcb(glfw.GLFWcursorenterfun, wrap)
        glfw.glfwSetCursorEnterCallback(self.handle, self.set_cursor_enter_callback)

    def set_cursor_pos_callback(self, callback):
        self.set_cursor_pos_callback = self._wcb(glfw.GLFWcursorposfun, callback)
        glfw.glfwSetCursorPosCallback(self.handle, self.set_cursor_pos_callback)

    def set_mouse_button_callback(self, callback):
        self.set_mouse_button_callback = self._wcb(glfw.GLFWmousebuttonfun, callback)
        glfw.glfwSetMouseButtonCallback(self.handle, self.set_mouse_button_callback)

    def set_window_pos_callback(self, callback):
        self.set_window_pos_callback = self._wcb(glfw.GLFWwindowposfun, callback)
        glfw.glfwSetWindowPosCallback(self.handle, self.set_window_pos_callback)

    def set_window_size_callback(self, callback):
        self.set_window_size_callback = self._wcb(glfw.GLFWwindowsizefun, callback)
        glfw.glfwSetWindowSizeCallback(self.handle, self.set_window_size_callback)

    def set_window_close_callback(self, callback):
        self.set_window_close_callback = self._wcb(glfw.GLFWwindowclosefun, callback)
        glfw.glfwSetWindowCloseCallback(self.handle, self.set_window_close_callback)

    def set_window_refresh_callback(self, callback):
        self.set_window_refresh_callback = self._wcb(glfw.GLFWwindowrefreshfun, callback)
        glfw.glfwSetWindowRefreshCallback(self.handle, self.set_window_refresh_callback)

    def set_window_focus_callback(self, callback):
        def wrap(self, flag):
            flag = bool(flag)
            callback(self, flag)
        self.set_window_focus_callback = self._wcb(glfw.GLFWwindowfocusfun, wrap)
        glfw.glfwSetWindowFocusCallback(self.handle, self.set_window_focus_callback)

    def set_window_iconify_callback(self, callback):
        def wrap(self, flag):
            flag = bool(flag)
            callback(self, flag)
        self.set_window_iconify_callback = self._wcb(glfw.GLFWwindowiconifyfun, wrap)
        glfw.glfwSetWindowIconifyCallback(self.handle, self.set_window_iconify_callback)

    def set_framebuffer_size_callback(self, callback):
        self.set_framebuffer_size_callback = self._wcb(glfw.GLFWframebuffersizefun, callback)
        glfw.glfwSetFramebufferSizeCallback(self.handle, self.set_framebuffer_size_callback)

    def set_callbacks(self, **kwargs):
        callback_map = {
            'key': self.set_key_callback,
            'char': self.set_char_callback,
            'scroll': self.set_scroll_callback,
            'cursor_enter': self.set_cursor_enter_callback,
            'cursor_pos': self.set_cursor_pos_callback,
            'mouse_button': self.set_mouse_button_callback,
            'window_pos': self.set_window_pos_callback,
            'window_size': self.set_window_size_callback,
            'window_close': self.set_window_close_callback,
            'window_refresh': self.set_window_refresh_callback,
            'window_focus': self.set_window_focus_callback,
            'window_iconify': self.set_window_iconify_callback,
            'framebuffer_size': self.set_framebuffer_size_callback
        }
        for k, v in kwargs.items():
            if k in callback_map.keys():
                callback_map[k](v)
            else:
                raise ValueError(f"Invalid callback \"{k}\"")

    @staticmethod
    def api_version():
        return glfw.glfwGetVersion()

    @staticmethod
    def api_version_string():
        return _str(glfw.glfwGetVersionString())

    @staticmethod
    def poll_events():
        glfw.glfwPollEvents()

    @staticmethod
    def wait_events():
        glfw.glfwWaitEvents()

    def quit(self):
        self.should_close = True

__window__ = None

def init_window(width: int = 640,
                 height: int = 480,
                 title: str = "PySpriteKit",
                 versions: Optional[tuple[int, int, bool]] = None,
                 frame_limit: Optional[int | float] = None,
                 hints: Optional[dict] = None):
    global __window__
    if __window__ is not None:
        raise RuntimeError("Window already initialized")
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
    __window__ = Window(width, height, title, hints=hints, frame_limit=frame_limit)

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

__all__ = ["KeyEvent", "CharEvent", "ScrollEvent", "MouseButtonEvent",
           "CursorEnterEvent", "CursorPosEvent", "WindowSizeEvent", "WindowPosEvent",
           "WindowCloseEvent", "WindowRefreshEvent", "WindowFocusEvent",
           "WindowIconifyEvent", "FrameBufferSizeEvent", "Hints", "Keys", "Mice",
           "Joystick", "Monitor", "VideoMode", "get_window", "init_window",
           "window_should_close", "window_width", "window_height", "window_size"]