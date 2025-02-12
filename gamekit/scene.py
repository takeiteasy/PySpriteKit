from .actor import ActorType
from .parent import Parent
from .fsm import FiniteStateMachine
import pyray as r

__scene__ = []
__next_scene = None
__drop_scene = None

class Scene(FiniteStateMachine, Parent):
    window_attrs: dict = {}

    def __init__(self, **kwargs):
        Parent.__init__(self)
        FiniteStateMachine.__init__(self, **kwargs)
        self.projection = r.matrix_identity()
    
    def add_child(self, node: ActorType):
        node.scene = self
        self.children.append(node)

    def enter(self):
        pass

    def reenter(self):
        pass

    def background(self):
        pass

    def exit(self):
        pass

    def step(self, delta):
        pass

    def draw(self):
        for child in self.children:
            child.draw()

def push_scene(scene: Scene):
    global __next_scene
    if __next_scene is not None:
        raise RuntimeError("Next scene already queued")
    __next_scene = scene

def drop_scene():
    global __scene__, __drop_scene
    if __drop_scene is not None:
        raise RuntimeError("Drop scene already queued")
    __drop_scene = __scene__[-1:]

def first_scene():
    global __scene__, __drop_scene
    __drop_scene = __scene__[1:]

def get_scene():
    if not __scene__:
        raise RuntimeError("No active Scene")
    return __scene__[0]

def main_scene(cls):
    global __scene__, __drop_scene, __next_scene
    if __scene__:
        raise RuntimeError("There can only be one @main_scene")
    r.init_window(cls.window_attrs['width'] if "width" in cls.window_attrs else 800,
                  cls.window_attrs['height'] if "height" in cls.window_attrs else 600,
                  cls.window_attrs['title'] if "title" in cls.window_attrs else "GameKit")
    r.set_config_flags(cls.window_attrs['flags'] if "flags" in cls.window_attrs else r.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    if "fps" in cls.window_attrs:
        r.set_target_fps(cls.window_attrs['fps'])
    if "exit_key" in cls.window_attrs:
        r.set_exit_key(cls.window_attrs['exit_key'])
    scn = cls()
    __scene__.append(scn)
    scn.enter()
    while not r.window_should_close() and __scene__:
        scn.step(r.get_frame_time())
        r.begin_drawing()
        scn.draw()
        r.end_drawing()
        if __drop_scene:
            if isinstance(__drop_scene, list):
                for _scn in reversed(__drop_scene):
                    _scn.exit()
            elif isinstance(__drop_scene, Scene):
                __drop_scene.exit()
            else:
                raise RuntimeError("Invalid Scene")
            __scene__ = __scene__[:-len(__drop_scene)]
            if __scene__:
                scn = __scene__[-1]
                scn.reenter()
            __drop_scene = None
        if __next_scene:
            if isinstance(__next_scene, Scene):
                if __scene__:
                    __scene__[-1].background()
                __scene__.append(__next_scene)
                scn = __next_scene
                scn.enter()
                __next_scene = None
            else:
                raise RuntimeError("Invalid Scene")
    return cls