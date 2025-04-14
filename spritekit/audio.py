# spritekit/audio.py
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

from typing import Optional, Callable

from .actor import Actor
from .timer import TimerActor
from .cache import *

import raudio as r

class BaseAudio(Actor):
    def __init__(self,
                 pitch: float = 1.,
                 volume: float = 1.,
                 pan: float = .5,
                 **kwargs):
        Actor.__init__(self, **kwargs)
        self._sound = None
        self._pitch = min(max(pitch, 0.), 1.)
        self._volume = min(max(volume, 0.), 1.)
        self._pan = min(max(pan, 0.), 1.)
     
    @property
    def volume(self):
        return self._sound.get_volume()
    
    @volume.setter
    def volume(self, value: float):
        self._sound.set_volume(value)
    
    @property
    def pitch(self):
        return self._sound.get_pitch()
    
    @pitch.setter
    def pitch(self, value: float):
        self._sound.set_pitch(value)
    
    @property
    def pan(self):
        return self._sound.get_pan()
    
    @pan.setter
    def pan(self, value: float):
        self._sound.set_pan(value)
    
    @property
    def playing(self):
        return self._sound.is_playing()
    
    @playing.setter
    def playing(self, value: bool):
        if value:
            self.resume()
        else:
            self.pause()

    def play(self):
        if not self.playing:
            self._sound.play()
    
    def stop(self):
        if self.playing:
            self._sound.stop()
    
    def pause(self):
        if self.playing:
            self._sound.pause()
    
    def resume(self):
        if not self.playing:
            self._sound.resume()
    
    def __len__(self):
        return len(self._sound)

class SoundActor(BaseAudio):
    def __init__(self,
                 source: str | r.Wave,
                 loop: bool = False,
                 times: int = 1,
                 wait: float | int = 0.,
                 **kwargs):
        BaseAudio.__init__(self, on_added=self._start, **kwargs)
        self._sound = r.Sound(load_wave(source) if isinstance(source, str) else source)
        self._loop = loop
        assert times > 0, "Times must be a positive integer"
        self._times = times - 1
        assert wait >= 0, "Wait must be a positive float or int"
        self._wait = wait
        self._started = False
        self._wait_timer = False
    
    def _start(self):
        self._wait_timer = False
        self._started = True
        self.play()
    
    def _restart(self):
        if self._wait > 0. and not self._wait_timer:
            self.add(TimerActor(duration=self._wait, on_complete=self._start))
            self._wait_timer = True
        else:
            self._start()
    
    def step(self, delta: float):
        super().step(delta)
        if self._wait_timer:
            return
        if self._started and not self.playing:
            if self._loop:
                self._restart()
            elif self._times > 0:
                self._times -= 1
                self._restart()
            else:
                self.remove()

class MusicActor(BaseAudio):    
    def __init__(self,
                 source: str,
                 loop: bool = False,
                 auto_start: bool = True,
                 on_complete: Optional[Callable[[], None]] = None,
                 remove_on_complete: bool = True,
                 **kwargs):
        BaseAudio.__init__(self, on_added=self.start if auto_start else None, on_removed=self.stop, **kwargs)
        self._sound = load_music(source)
        self._loop = loop
        self._autostart = auto_start
        self._on_complete = on_complete
        self._remove_on_complete = remove_on_complete

    @property
    def loop(self):
        return self._loop
    
    @loop.setter
    def loop(self, value: bool):
        self._loop = value
    
    @property
    def position(self):
        return self._sound.get_position()
    
    @position.setter
    def position(self, value: float):
        self._sound.set_position(value)

    def start(self):
        if not self.playing:
            self._sound.play()

    def step(self, delta: float):
        super().step(delta)
        if self.playing:
            self._sound.update()
            if not self.playing:
                if self._on_complete is not None:
                    self._on_complete()
                if self.loop:
                    self._sound.seek(0)
                    self._sound.play()
                else:
                    if self._remove_on_complete:
                        self.remove()

__all__ = ["SoundActor", "MusicActor"]