"""Draw Rula's app icon: launcher/rula.ico (for Rula.exe) and the window's icon.png.

    python scripts/make_icon.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\YuGothB.ttc",
         r"C:\Windows\Fonts\arialbd.ttf"]


def draw(size: int = 1024) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pad = size * 0.06
    d.rounded_rectangle((pad, pad, size - pad, size - pad), radius=size * 0.22,
                        fill=(20, 20, 22, 255))
    font = ImageFont.truetype(next(p for p in FONTS if os.path.exists(p)), int(size * 0.58))
    box = d.textbbox((0, 0), "R", font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = (size - w) / 2 - box[0] - size * 0.04
    y = (size - h) / 2 - box[1]
    d.text((x, y), "R", font=font, fill=(242, 242, 244, 255))
    r = size * 0.075  # the pink full stop, as in the window's wordmark
    cx, cy = size * 0.73, size * 0.70
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(229, 143, 177, 255))
    return im


if __name__ == "__main__":
    im = draw()
    os.makedirs(os.path.join(ROOT, "launcher"), exist_ok=True)
    im.save(os.path.join(ROOT, "launcher", "rula.ico"),
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    im.resize((128, 128), Image.LANCZOS).save(
        os.path.join(ROOT, "src", "rula", "gui", "static", "icon.png"))
    print("launcher/rula.ico, src/rula/gui/static/icon.png")
