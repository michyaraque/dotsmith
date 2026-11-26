"""Dotsmith logo: the 64 px pixel-art source (dotsmith-64.png, drawn with Krea 2 Turbo) -> dotsmith.png (256) + dotsmith.ico."""
from pathlib import Path
from PIL import Image

HERE = Path(__file__).parent
src = Image.open(HERE / 'dotsmith-64.png').convert('RGBA')
big = src.resize((256, 256), Image.NEAREST)
big.save(HERE / 'dotsmith.png', optimize=True)
# whole multiples stay crisp; the in-between sizes are averaged so no pixel row drops out
frames = [src.resize((s, s), Image.NEAREST) if s % 64 == 0 else big.resize((s, s), Image.BOX) for s in (16, 24, 32, 48, 64, 128, 256)]
frames[-1].save(HERE / 'dotsmith.ico', sizes=[f.size for f in frames], append_images=frames[:-1])
print('ok')
