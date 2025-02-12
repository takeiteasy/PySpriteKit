from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import transitions

@dataclass
class State:
    name: str | Enum
    on_enter: Optional[Callable[[Any], Any]] = None
    on_exit: Optional[Callable[[Any], Any]] = None,
    ignore_invalid_triggers: Optional[bool] = False
    final: Optional[bool] = False

    def explode(self):
        return self.__dict__.items()

    @property
    def value(self):
        return self.name

@dataclass
class Transition:
    trigger: str
    source: str | Enum | list
    dest: str | Enum
    conditions: Optional[str | list[str]] = None
    unless: Optional[str | list[str]] = None
    before: Optional[str | list[str]] = None
    after: Optional[str | list[str]] = None
    prepare: Optional[str | list[str]] = None
    kwargs: Optional[dict] = field(default_factory=dict)

    def explode(self):
        transition_args = {k: v for k, v in self.__dict__.items() if k != 'kwargs' and v is not None}
        transition_args.update(self.kwargs)  # unpack kwargs
        return transition_args

class FiniteStateMachine:
    states: list[str | State] = []
    transitions: list[dict | Transition] = []

    def __init__(self, **kwargs):
        if self.__class__.states:
            if not "initial" in kwargs:
                kwargs["initial"] = self.__class__.states[0]
            self.fsm = transitions.Machine(model=self,
                                           states=self.__class__.states,
                                           transitions=[t.explode() if isinstance(t, Transition) else t for t in self.__class__.transitions],
                                           **kwargs)
        else:
            self.fsm = None