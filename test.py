import spritekit as sk
from typing import override

@sk.main
class TestScene(sk.Scene):
    config = {
        "width": 800,
        "height": 600,
        "title": "Test",
        "frame_limit": 60
    }