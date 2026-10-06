"""Two Instagram feed posts (1080x1350, 4:5) for MiFoodStudio Thanksgiving."""
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import numpy as np

SRC = "/home/user/data-work/"
W, H = 1080, 1350
F = "/usr/share/fonts/opentype/inter/"
RED, ORANGE, CREAM, INK, WHITE = (226, 29, 43), (242, 107, 29), (255, 247, 238), (28, 24, 22), (255, 255, 255)
def font(n, s): return ImageFont.truetype(F + n, s)
FX = lambda s: font("InterDisplay-ExtraBold.otf", s)
FB = lambda s: font("InterDisplay-Bold.otf", s)
FM = lambda s: font("Inter-Medium.otf", s)
FS = lambda s: font("Inter-SemiBold.otf", s)
FI = lambda s: font("Inter-MediumItalic.otf", s)
LOGO = Image.open("logo_h.png").convert("RGBA")

def load(fn):
    im = ImageOps.exif_transpose(Image.open(SRC + fn)).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.08); im = ImageEnhance.Color(im).enhance(1.1)
    r, g, b = im.split()
    r = r.point(lambda v: min(255, int(v * 1.04 + 4))); b = b.point(lambda v: int(v * 0.95))
    return Image.merge("RGB", (r, g, b))

def cover(im, w, h, cx=0.5, cy=0.5):
    s = max(w / im.width, h / im.height)
    im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
    x0 = int((im.width - w) * cx); y0 = int((im.height - h) * cy)
    return im.crop((x0, y0, x0 + w, y0 + h))

def gradient(w, h, top_alpha, bot_alpha, color=(0, 0, 0), start=0.0):
    a = np.linspace(0, 1, h)[:, None]
    a = np.clip((a - start) / (1 - start), 0, 1)
    al = (top_alpha + (bot_alpha - top_alpha) * a) * 255
    g = np.zeros((h, w, 4), np.uint8); g[..., :3] = color; g[..., 3] = np.repeat(al, w, 1)
    return Image.fromarray(g, "RGBA")

def pill_logo(im, x, y, lw=330):
    lg = LOGO.resize((lw, int(LOGO.height * lw / LOGO.width)), Image.LANCZOS)
    pad = 18
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    d.rounded_rectangle((x, y, x + lw + 2 * pad, y + lg.height + 2 * pad), radius=22, fill=(255, 255, 255, 240))
    im = Image.alpha_composite(im.convert("RGBA"), ov)
    im.paste(lg, (x + pad, y + pad), lg)
    return im

def fit(txt, fnt_fn, size, maxw):
    f = fnt_fn(size)
    while f.getlength(txt) > maxw and size > 30:
        size -= 2; f = fnt_fn(size)
    return f

# ------------------------------------------------------------------ POST 1: hero offer
def post1():
    im = cover(load("Burner1.png"), W, H, 0.45, 0.35)
    im = im.convert("RGBA")
    im = Image.alpha_composite(im, gradient(W, H, 0.0, 0.88, (12, 8, 6), start=0.30))
    im = Image.alpha_composite(im, gradient(W, H, 0.35, 0.0, (0, 0, 0)))  # soften top for logo
    im = pill_logo(im, 48, 48, 330)
    d = ImageDraw.Draw(im)
    # date tag top-right
    tag = "CANADIAN THANKSGIVING · OCT 12"
    f = FS(26); tw = f.getlength(tag)
    d.rounded_rectangle((W - 48 - tw - 44, 60, W - 48, 118), radius=29, fill=RED)
    d.text((W - 48 - 22, 89), tag, font=f, fill=WHITE, anchor="rm")
    y = 610
    d.rectangle((64, y - 40, 230, y - 28), fill=RED)
    for line, col in [("COOK YOUR", WHITE), ("THANKSGIVING", (255, 170, 90)), ("FEAST IN A", WHITE), ("PRO KITCHEN.", WHITE)]:
        f = FX(100)
        d.text((67, y + 5), line, font=f, fill=(0, 0, 0, 150)); d.text((64, y), line, font=f, fill=col)
        y += 102
    y += 22
    sub = ["Rent our commercial kitchen in Mississauga for", "your family's big-meal prep. Six burners, a combi", "oven, a walk-in cooler and room for everyone."]
    for s in sub:
        d.text((64, y), s, font=FM(34), fill=(240, 232, 225)); y += 44
    # CTA
    y += 26
    pw, ph = 470, 96
    d.rounded_rectangle((64, y, 64 + pw, y + ph), radius=48, fill=RED)
    d.text((64 + pw // 2, y + ph // 2), "BOOK YOUR SLOT", font=FX(40), fill=WHITE, anchor="mm")
    d.text((64 + pw + 36, y + ph // 2 - 20), "(289) 270-0990", font=FS(36), fill=WHITE, anchor="lm")
    d.text((64 + pw + 36, y + ph // 2 + 22), "mifoodstudio.com", font=FM(30), fill=(255, 200, 150), anchor="lm")
    return im.convert("RGB")

# ------------------------------------------------------------------ POST 2: what you get
def post2():
    im = Image.new("RGB", (W, H), CREAM)
    glow = Image.new("RGB", (W, H), CREAM); gd = ImageDraw.Draw(glow)
    gd.ellipse((-400, -400, 500, 500), fill=(255, 224, 196)); gd.ellipse((W - 500, H - 500, W + 400, H + 400), fill=(255, 214, 190))
    im = Image.blend(im, glow.filter(ImageFilter.GaussianBlur(140)), 0.9)
    d = ImageDraw.Draw(im)
    lg = LOGO.resize((420, int(LOGO.height * 420 / LOGO.width)), Image.LANCZOS)
    im.paste(lg, (48, 44), lg)
    tag = "THANKSGIVING 2026"
    f = FS(26); tw = f.getlength(tag)
    d.rounded_rectangle((W - 48 - tw - 44, 56, W - 48, 114), radius=29, fill=RED)
    d.text((W - 48 - 22, 85), tag, font=f, fill=WHITE, anchor="rm")
    y = 190
    d.rectangle((48, y - 30, 190, y - 18), fill=RED)
    d.text((48, y), "EVERYTHING YOUR", font=FX(84), fill=INK); y += 88
    d.text((48, y), "HOME KITCHEN ISN'T.", font=FX(84), fill=RED); y += 110
    d.text((48, y), "One stove and a crowded counter won't feed 20.", font=FM(32), fill=(90, 70, 60))
    d.text((48, y + 42), "Cook the whole feast here, then celebrate at home.", font=FM(32), fill=(90, 70, 60))
    # 2x2 grid
    gy = 510; gap = 20; cw = (W - 96 - gap) // 2; ch = 290
    cells = [("Burner1.png", "6 OPEN BURNERS", 0.5, 0.5), ("Oven.png", "COMMERCIAL COMBI OVEN", 0.6, 0.45),
             ("Cooler.png", "WALK-IN COOLER", 0.65, 0.55), ("Main Prep Area1.png", "PREP TABLES FOR THE FAMILY", 0.55, 0.55)]
    for i, (fn, label, cx, cy) in enumerate(cells):
        x = 48 + (i % 2) * (cw + gap); yy = gy + (i // 2) * (ch + gap)
        ph = cover(load(fn), cw, ch, cx, cy).convert("RGBA")
        ph = Image.alpha_composite(ph, gradient(cw, ch, 0.0, 0.82, (10, 6, 4), start=0.45))
        mask = Image.new("L", (cw, ch), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, cw - 1, ch - 1), radius=26, fill=255)
        im.paste(ph.convert("RGB"), (x, yy), mask)
        dd = ImageDraw.Draw(im)
        dd.rectangle((x + 24, yy + ch - 78, x + 70, yy + ch - 72), fill=ORANGE)
        dd.text((x + 24, yy + ch - 34 - 6), label, font=fit(label, FX, 34, cw - 48), fill=WHITE, anchor="ls")
    d = ImageDraw.Draw(im)
    # footer
    y = gy + 2 * ch + gap + 36
    pw, ph2 = 560, 92
    d.rounded_rectangle((48, y, 48 + pw, y + ph2), radius=46, fill=RED)
    d.text((48 + pw // 2, y + ph2 // 2), "BOOK YOUR THANKSGIVING SLOT", font=FX(30), fill=WHITE, anchor="mm")
    d.text((48 + pw + 30, y + ph2 // 2 - 18), "(289) 270-0990", font=FS(34), fill=INK, anchor="lm")
    d.text((48 + pw + 30, y + ph2 // 2 + 22), "mifoodstudio.com", font=FM(28), fill=ORANGE, anchor="lm")
    y += ph2 + 28
    d.text((48, y), "Unit 1, 6905 Millcreek Drive, Mississauga, ON · Meadowvale Business Park", font=fit("Unit 1, 6905 Millcreek Drive, Mississauga, ON · Meadowvale Business Park", FM, 28, W - 96), fill=(110, 90, 80))
    d.text((48, y + 38), "The MiFoodStudio team wishes you a very Happy Thanksgiving!", font=FI(28), fill=ORANGE)
    return im

post1().save("insta_post_1_hero.jpg", quality=94)
post2().save("insta_post_2_features.jpg", quality=94)
print("posts ok")
