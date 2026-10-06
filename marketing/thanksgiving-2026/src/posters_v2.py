"""Two ad posters for Instagram (1080x1350). Emotional headline first, kitchen as proof."""
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import numpy as np

SRC = "/home/user/data-work/"
S = "/tmp/claude-0/-home-user-data-work/6d3e5537-b273-5292-9458-851e6e4d1899/scratchpad/"
W, H = 1080, 1350
F = "/usr/share/fonts/opentype/inter/"
RED, ORANGE, CREAM, INK, WHITE = (226, 29, 43), (242, 107, 29), (255, 247, 238), (28, 24, 22), (255, 255, 255)
def font(n, s): return ImageFont.truetype(F + n, s)
FX = lambda s: font("InterDisplay-ExtraBold.otf", s)
FH = lambda s: font("InterDisplay-SemiBold.otf", s)
FM = lambda s: font("Inter-Medium.otf", s)
FS = lambda s: font("Inter-SemiBold.otf", s)
FI = lambda s: font("Inter-MediumItalic.otf", s)
LOGO = Image.open(S + "logo_h.png").convert("RGBA")

def grade(im, mood):
    if mood == "cold":
        im = ImageEnhance.Color(im).enhance(0.45); im = ImageEnhance.Contrast(im).enhance(1.12)
        r, g, b = im.split(); r = r.point(lambda v: int(v * 0.92)); b = b.point(lambda v: min(255, int(v * 1.06 + 6)))
    else:
        im = ImageEnhance.Contrast(im).enhance(1.06); im = ImageEnhance.Color(im).enhance(1.12)
        r, g, b = im.split(); r = r.point(lambda v: min(255, int(v * 1.05 + 5))); b = b.point(lambda v: int(v * 0.93))
    return Image.merge("RGB", (r, g, b))

def load(fn, mood="warm"):
    p = fn if fn.startswith("/") else SRC + fn
    return grade(ImageOps.exif_transpose(Image.open(p)).convert("RGB"), mood)

def cover(im, w, h, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
    x0 = int((im.width - w) * cx); y0 = int((im.height - h) * cy)
    return im.crop((x0, y0, x0 + w, y0 + h))

def grad(w, h, a0, a1, color=(0, 0, 0), start=0.0, vertical=True):
    a = np.linspace(0, 1, h if vertical else w)
    a = np.clip((a - start) / (1 - start), 0, 1)
    al = (a0 + (a1 - a0) * a) * 255
    g = np.zeros((h, w, 4), np.uint8); g[..., :3] = color
    g[..., 3] = al[:, None] if vertical else al[None, :]
    return Image.fromarray(g, "RGBA")

def grain(im, amt=4):
    a = np.asarray(im).astype(np.float32) + np.random.default_rng(1).normal(0, amt, (im.height, im.width, 1))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def footer(im, d, y, dark=True):
    col = WHITE if dark else INK
    sub = (255, 205, 160) if dark else ORANGE
    pw, ph = 430, 92
    d.rounded_rectangle((64, y, 64 + pw, y + ph), radius=46, fill=RED)
    d.text((64 + pw // 2, y + ph // 2), "(289) 270-0990", font=FX(40), fill=WHITE, anchor="mm")
    d.text((64 + pw + 30, y + ph // 2 - 18), "mifoodstudio.com", font=FS(34), fill=col, anchor="lm")
    d.text((64 + pw + 30, y + ph // 2 + 22), "Unit 1, 6905 Millcreek Dr, Mississauga", font=FM(26), fill=sub, anchor="lm")

# ------------------------------------------------------------ POSTER 1 — the promise
def poster1():
    im = cover(load("20260808_143157.jpg"), W, H, 0.45, 0.5).convert("RGBA")
    im = Image.alpha_composite(im, grad(W, H, 0.0, 0.92, (14, 8, 5), start=0.28))
    im = Image.alpha_composite(im, grad(W, H, 0.45, 0.0, (0, 0, 0)))
    im = grain(im.convert("RGB"), 3)
    d = ImageDraw.Draw(im)
    lg = LOGO.resize((300, int(LOGO.height * 300 / LOGO.width)), Image.LANCZOS)
    pill = Image.new("RGBA", im.size, (0, 0, 0, 0)); pd = ImageDraw.Draw(pill)
    pd.rounded_rectangle((48, 44, 48 + 300 + 32, 44 + lg.height + 32), radius=20, fill=(255, 255, 255, 235))
    im = Image.alpha_composite(im.convert("RGBA"), pill).convert("RGB"); im.paste(lg, (64, 60), lg)
    d = ImageDraw.Draw(im)
    tag = "CANADIAN THANKSGIVING · OCT 12"
    f = FS(24); tw = f.getlength(tag)
    d.rounded_rectangle((W - 48 - tw - 40, 58, W - 48, 112), radius=27, fill=RED)
    d.text((W - 48 - 20, 85), tag, font=f, fill=WHITE, anchor="rm")
    y = 640
    d.rectangle((64, y - 36, 200, y - 26), fill=RED)
    for line, col, f in [("Big family.", WHITE, FX(112)), ("Small kitchen.", WHITE, FX(112)), ("We fixed that.", (255, 176, 96), FX(112))]:
        d.text((67, y + 4), line, font=f, fill=(0, 0, 0)); d.text((64, y), line, font=f, fill=col); y += 118
    y += 20
    for s in ["Cook your whole Thanksgiving feast in our", "professional kitchen in Mississauga,", "then take it home and enjoy the day."]:
        d.text((64, y), s, font=FM(34), fill=(240, 232, 225)); y += 44
    footer(im, d, 1188)
    return im

# ------------------------------------------------------------ POSTER 2 — at home / here
def poster2():
    im = Image.new("RGB", (W, H), (12, 9, 8))
    top_h = 560
    top = cover(load(S + "c_home_up.jpg", "cold"), W, top_h, 0.5, 0.55)
    bot = cover(load(S + "c_cooking.png", "warm"), W, H - top_h, 0.55, 0.35)
    im.paste(top, (0, 0)); im.paste(bot, (0, top_h))
    im = im.convert("RGBA")
    im = Image.alpha_composite(im, grad(W, top_h, 0.15, 0.80, (10, 8, 10), start=0.35).crop((0, 0, W, top_h)) if False else Image.new("RGBA", (W, H), (0, 0, 0, 0)))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ov.paste(grad(W, top_h, 0.05, 0.85, (8, 6, 10), start=0.3), (0, 0))
    ov.paste(grad(W, H - top_h, 0.0, 0.92, (14, 8, 5), start=0.35), (0, top_h))
    im = Image.alpha_composite(im, ov)
    im = grain(im.convert("RGB"), 3)
    d = ImageDraw.Draw(im)
    # divider
    d.rectangle((0, top_h - 3, W, top_h + 3), fill=RED)
    # labels
    for txt, y in [("AT HOME", 40), ("AT MIFOODSTUDIO", top_h + 40)]:
        f = FS(24); tw = f.getlength(txt)
        d.rounded_rectangle((48, y, 48 + tw + 40, y + 50), radius=25, fill=(0, 0, 0, 160) if txt == "AT HOME" else RED)
        d.text((68, y + 25), txt, font=f, fill=WHITE, anchor="lm")
    # top headline
    y = top_h - 250
    d.text((67, y + 4), "Same turkey.", font=FX(96), fill=(0, 0, 0)); d.text((64, y), "Same turkey.", font=FX(96), fill=(225, 225, 230)); y += 104
    d.text((67, y + 4), "Different day.", font=FX(96), fill=(0, 0, 0)); d.text((64, y), "Different day.", font=FX(96), fill=(225, 225, 230))
    # bottom headline
    y = H - 440
    d.rectangle((64, y - 34, 200, y - 24), fill=RED)
    d.text((67, y + 4), "Change the kitchen,", font=FX(84), fill=(0, 0, 0)); d.text((64, y), "Change the kitchen,", font=FX(84), fill=WHITE); y += 90
    d.text((67, y + 4), "not the recipe.", font=FX(84), fill=(0, 0, 0)); d.text((64, y), "not the recipe.", font=FX(84), fill=(255, 176, 96)); y += 110
    d.text((64, y), "Thanksgiving slots are limited. Book yours today.", font=FM(32), fill=(240, 232, 225))
    footer(im, d, H - 150)
    lg = LOGO.resize((260, int(LOGO.height * 260 / LOGO.width)), Image.LANCZOS)
    pill = Image.new("RGBA", im.size, (0, 0, 0, 0)); pd = ImageDraw.Draw(pill)
    pd.rounded_rectangle((W - 48 - 260 - 28, 36, W - 48, 36 + lg.height + 28), radius=18, fill=(255, 255, 255, 235))
    im = Image.alpha_composite(im.convert("RGBA"), pill).convert("RGB"); im.paste(lg, (W - 48 - 260 - 14, 50), lg)
    return im

poster1().save(S + "poster_1_promise.jpg", quality=94)
poster2().save(S + "poster_2_home_vs_here.jpg", quality=94)
print("posters ok")
