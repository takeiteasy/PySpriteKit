import moderngl
import quickwindow
import glm
from pyglsl import VertexStage, FragmentStage
from spritekit.shader import *
import math
import platform
import numpy as np
from PIL import Image

def _rotate_point(x, y, c, s):
    return glm.vec2(x * c - y * s, x * s + y * c)

def _rect_vertices(x, y, w, h, rotation=0., scale=1., tr=(0, 0, 1, 1), color=(1, 1, 1, 1)):
    hw = w / 2
    hh = h / 2
    x1 = (x - hw) * scale
    x2 = (x + hw) * scale
    y1 = (y - hh) * scale
    y2 = (y + hh) * scale
    c = math.cos(rotation)
    s = math.sin(rotation)
    p1 = _rotate_point(x1, y1, c, s)
    p2 = _rotate_point(x2, y1, c, s)
    p3 = _rotate_point(x1, y2, c, s)
    p4 = _rotate_point(x2, y2, c, s)
    return [*p1, tr[0], tr[1], *color,
            *p2, tr[2], tr[1], *color,
            *p3, tr[0], tr[3], *color,
            *p3, tr[0], tr[3], *color,
            *p4, tr[2], tr[3], *color,
            *p2, tr[2], tr[1], *color]

def _vbo(ctx, vertices):
    return ctx.buffer(np.array(vertices, dtype=np.float32).tobytes() if isinstance(vertices, list) else vertices.astype('f4').tobytes())

class Mesh:
    def __init__(self, program, vertices):
        self._ctx = moderngl.get_context()
        vbo = _vbo(self._ctx, vertices) 
        self._vao = self._ctx.vertex_array(program, [(vbo, '2f 2f 4f', 'position', 'texcoords', 'in_color')])

    def draw(self, mvp=None, texture=None):
        if texture is not None:
            self._vao.program['use_texture'] = 1 
            texture.use()
        else:
            self._vao.program['use_texture'] = 0
        if mvp is None:
            mvp = glm.mat4()
        self._vao.program['mvp'].write(mvp)
        self._vao.render()

class Renderer:
    def __init__(self):
        self._ctx = moderngl.get_context()
        self._program = self._ctx.program(vertex_shader=VertexStage(default_vertex).compile(),
                                          fragment_shader=FragmentStage(default_fragment).compile())
        self._size = quickwindow.size()
        self._view = None
        self._update_view()
        self._dirty = True
        self._clear_color = glm.vec4(1., 0., 0., 1.)
    
    def _update_view(self):
        halfw, halfh = self._size[0] / 2, self._size[1] / 2
        self._view = glm.ortho(-halfw, halfw, -halfh, halfh, -1., 1.)
        self._dirty = False
    
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
        if platform.system() == "Darwin":
            self._size = (self._size[0] * 2, self._size[1] * 2)
        self._dirty = True
    
    @property
    def clear_color(self):
        return self._clear_color
    
    @clear_color.setter
    def clear_color(self, value):
        self._clear_color = tuple(min(max(float(v) / 255. if isinstance(v, int) else v, 0.), 1.) for v in [*value])
    
    @property
    def view(self):
        return self._view

    def load_texture(self, image, flip=True):
        if isinstance(image, str):
            image = Image.open(image)
        if flip:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return self._ctx.texture(image.size, 4, image.convert('RGBA').tobytes())

    def flush(self):
        if self._dirty:
            self._update_view()
        self._ctx.clear(viewport=self._size, color=self._clear_color)

with quickwindow.quick_window(quit_key=quickwindow.Keys.ESCAPE) as wnd:
    renderer = Renderer()
    image = Image.open('assets/textures/pear.jpg')
    texture = renderer.load_texture(image)
    rect = _rect_vertices(0, 0, 100, 100)
    mesh = Mesh(renderer._program, rect)
    for delta, events in wnd.loop():
        renderer.flush()
        mesh.draw(mvp=renderer.view, texture=texture)