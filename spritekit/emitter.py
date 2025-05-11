# spritekit/emitter.py
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

from typing import Callable, Type

from .actor import Actor
from .timer import TimerNode

class EmitterNode(TimerNode):
    def __init__(self,
                 repeat: bool | int = True,
                 emit: Callable[[], Actor] | Actor | tuple[Type[Actor], dict] = None,
                 **kwargs):
        self._user_on_complete = kwargs.pop("on_complete", None)
        if callable(emit):
            self._emit = staticmethod(emit)
        elif isinstance(emit, Actor):
            self._copy = emit.clone()
            self._emit = staticmethod(self._clone)
        else:
            self._type = emit[0]
            self._args = emit[1]
            self._emit = staticmethod(self._create)
        super().__init__(on_complete=self._spawn, repeat=repeat, **kwargs)
        if self.auto_start:
            self._emit()
    
    def _clone(self):
        return self._copy.clone()
    
    def _create(self):
        return self._type(**self._args)
    
    def _spawn(self):
        assert self.parent is not None, "Emitter must have a parent"
        assert callable(self._emit), "Emitter must be callable"
        self.parent.add(self._emit())
        if self._user_on_complete:
            self._user_on_complete()

__all__ = ["EmitterNode"]
