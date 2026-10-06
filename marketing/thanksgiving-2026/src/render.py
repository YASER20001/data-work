"""MiFoodStudio Thanksgiving Reel renderer.
Composes 1080x1920 frames with PIL (Ken Burns on real photos, warm grade,
animated captions, brand end card) and pipes them to ffmpeg.
usage: python3 render.py out.mp4 [--preview]   (preview = 1 frame per shot)
"""
import sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance

SRC = "/home/user/data-work/"
W, H, FPS = 1080, 1920, 30
F = "/usr/share/fonts/opentype/inter/"
RED, ORANGE, CREAM, INK = (226, 29, 43), (242, 107, 29), (255, 247, 238), (28, 24, 22)

def font(name, size): return ImageFont.truetype(F + name, size)
FX = lambda s: font("InterDisplay-ExtraBold.otf", s)
FB = lambda s: font("InterDisplay-Bold.otf", s)
FM = lambda s: font("Inter-Medium.otf", s)
FS = lambda s: font("Inter-SemiBold.otf", s)

# ---------------------------------------------------------------- shots
# (file, duration s, [cx0,cy0,zoom0] -> [cx1,cy1,zoom1] as fractions of source,
#  caption lines, accent kicker)
# zoom = fraction of source *height* (for landscape) that the 9:16 frame covers.
SHOTS = [
    dict(f="Burner1.png",            d=5.0, a=(0.42, 0.55, 1.00), b=(0.40, 0.52, 0.84),
         caps=[(0.0, 2.5, ["THANKSGIVING", "FOR 20?"], None),
               (2.5, 5.0, ["YOUR HOME OVEN", "CAN'T."], None)]),
    dict(f="20260808_143055.jpg",    d=2.5, a=(0.40, 0.50, 1.00), b=(0.60, 0.50, 0.96),
         caps=[(0.0, 2.5, ["THIS ONE", "CAN."], None)]),
    dict(f="Equpment Close3.png",    d=2.5, a=(0.30, 0.55, 0.95), b=(0.45, 0.55, 0.80),
         caps=[(0.0, 2.5, ["6 OPEN", "BURNERS"], "01")]),
    dict(f="Oven.png",               d=2.5, a=(0.62, 0.50, 1.00), b=(0.62, 0.48, 0.86),
         caps=[(0.0, 2.5, ["COMMERCIAL", "COMBI OVEN"], "02")]),
    dict(f="Cooler.png",             d=2.5, a=(0.58, 0.55, 1.00), b=(0.66, 0.58, 0.84),
         caps=[(0.0, 2.5, ["WALK-IN", "COOLER"], "03")]),
    dict(f="Main Prep Area1.png",    d=2.5, a=(0.60, 0.55, 1.00), b=(0.50, 0.55, 0.88),
         caps=[(0.0, 2.5, ["PREP TABLES FOR", "THE WHOLE FAMILY"], "04")]),
    dict(f="Washing1.png",           d=2.5, a=(0.50, 0.55, 1.00), b=(0.55, 0.60, 0.86),
         caps=[(0.0, 2.5, ["NO MESS", "AT HOME."], "05")]),
    dict(f="20260808_143157.jpg",    d=5.0, a=(0.62, 0.50, 1.00), b=(0.45, 0.50, 0.90),
         caps=[(0.0, 5.0, ["COOK HERE.", "CELEBRATE AT HOME."], None)],
         sub="Rental kitchen · Meadowvale, Mississauga"),
    dict(f="20260808_143445.jpg",    d=3.0, a=(0.50, 0.50, 0.95), b=(0.50, 0.50, 1.00),
         caps=[(0.0, 3.0, ["THANKSGIVING SLOTS", "ARE LIMITED"], None)], dark=0.55,
         sub="Book now · (289) 270-0990"),
    dict(f=None,                     d=5.0),   # end card
]
XFADE = 0.35

# ---------------------------------------------------------------- helpers
def ease(x):  # smoothstep
    return x * x * (3 - 2 * x)

_cache = {}
def load(fn):
    if fn not in _cache:
        im = ImageOps.exif_transpose(Image.open(SRC + fn)).convert("RGB")
        # warm, lifted grade
        im = ImageEnhance.Contrast(im).enhance(1.08)
        im = ImageEnhance.Color(im).enhance(1.10)
        r, g, b = im.split()
        r = r.point(lambda v: min(255, int(v * 1.04 + 4)))
        b = b.point(lambda v: int(v * 0.95))
        im = Image.merge("RGB", (r, g, b))
        # cap size for speed
        if im.height > 2200:
            im = im.resize((int(im.width * 2200 / im.height), 2200), Image.LANCZOS)
        _cache[fn] = im
    return _cache[fn]

def kenburns(im, a, b, p):
    p = ease(p)
    cx = a[0] + (b[0] - a[0]) * p
    cy = a[1] + (b[1] - a[1]) * p
    z  = a[2] + (b[2] - a[2]) * p
    # crop height = z * source height, keep 9:16
    ch = im.height * z
    cw = ch * W / H
    if cw > im.width:           # portrait-ish sources: width-bound
        cw = im.width * z; ch = cw * H / W
    x0 = min(max(cx * im.width - cw / 2, 0), im.width - cw)
    y0 = min(max(cy * im.height - ch / 2, 0), im.height - ch)
    return im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR)

_vig = None
def vignette():
    global _vig
    if _vig is None:
        y, x = np.mgrid[0:H, 0:W]
        d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        v = np.clip(1 - 0.45 * np.clip(d - 0.55, 0, 1) ** 1.5, 0, 1)
        # extra darkening bottom third so captions read
        v *= 1 - 0.35 * np.clip((y - H * 0.55) / (H * 0.45), 0, 1)
        _vig = v[..., None].astype(np.float32)
    return _vig

def grade(frame, dark=0.0):
    a = np.asarray(frame).astype(np.float32)
    a = a * vignette() * (1 - dark)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def shadow_text(draw, xy, txt, fnt, fill, anchor="la"):
    x, y = xy
    draw.text((x + 3, y + 5), txt, font=fnt, fill=(0, 0, 0, 160), anchor=anchor)
    draw.text((x, y), txt, font=fnt, fill=fill, anchor=anchor)

def caption(frame, lines, t_in, kicker=None, sub=None):
    """lines slide up + fade in over 0.35 s; big uppercase, left aligned, lower third."""
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    size = 112
    fnt = FX(size)
    while max(fnt.getlength(l) for l in lines) > W - 144 and size > 60:
        size -= 4; fnt = FX(size)
    lh = int(size * 1.08)
    base_y = H - 560
    x = 72
    for i, line in enumerate(lines):
        p = ease(min(max((t_in - i * 0.08) / 0.35, 0), 1))
        y = base_y + i * lh + int((1 - p) * 40)
        al = int(255 * p)
        col = (255, 255, 255, al)
        if i == len(lines) - 1 and len(lines) > 1 and lines[-1].endswith(('.', '?')):
            col = (255, 255, 255, al)
        d.text((x + 3, y + 5), line, font=fnt, fill=(0, 0, 0, int(170 * p)))
        d.text((x, y), line, font=fnt, fill=col)
    # red accent bar
    p = ease(min(t_in / 0.4, 1))
    d.rectangle((x, base_y - 34, x + int(140 * p), base_y - 22), fill=RED + (255,))
    if kicker:
        d.text((x + 160, base_y - 46), kicker + "  /  05", font=FS(30), fill=(255, 255, 255, int(220 * p)))
    if sub:
        d.text((x, base_y + len(lines) * lh + 18), sub, font=FM(36), fill=(255, 225, 200, int(235 * p)))
    return Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")

_logo = None
def logo():
    global _logo
    if _logo is None:
        _logo = Image.open("logo_h.png").convert("RGBA")
    return _logo

def end_card(t):
    im = Image.new("RGB", (W, H), CREAM)
    d = ImageDraw.Draw(im)
    # soft orange glow top-right, red band bottom
    glow = Image.new("RGB", (W, H), CREAM)
    gd = ImageDraw.Draw(glow)
    gd.ellipse((W - 700, -500, W + 300, 500), fill=(255, 226, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(160))
    im = Image.blend(im, glow, 0.9)
    d = ImageDraw.Draw(im)
    p = ease(min(t / 0.5, 1))
    lg = logo()
    lw = 820; lh = int(lg.height * lw / lg.width)
    lg = lg.resize((lw, lh), Image.LANCZOS)
    im.paste(lg, ((W - lw) // 2, 430 + int((1 - p) * 30)), lg)
    p2 = ease(min(max((t - 0.35) / 0.5, 0), 1))
    ink = tuple(int(c * p2 + 255 * (1 - p2) * 0.97) for c in INK)
    d.text((W // 2, 760), "BOOK YOUR", font=FX(96), fill=ink, anchor="ma")
    d.text((W // 2, 860), "THANKSGIVING SLOT", font=FX(96), fill=tuple(int(c * p2 + 255 * (1 - p2)) for c in RED), anchor="ma")
    d.text((W // 2, 990), "Canadian Thanksgiving · Monday, October 12", font=FM(38), fill=ink, anchor="ma")
    p3 = ease(min(max((t - 0.8) / 0.5, 0), 1))
    # CTA pill
    pw, ph = 760, 118
    px, py = (W - pw) // 2, 1110
    pill = Image.new("RGBA", (W, H), (0, 0, 0, 0)); pd = ImageDraw.Draw(pill)
    pd.rounded_rectangle((px, py, px + pw, py + ph), radius=59, fill=RED + (int(255 * p3),))
    im = Image.alpha_composite(im.convert("RGBA"), pill).convert("RGB")
    d = ImageDraw.Draw(im)
    d.text((W // 2, py + ph // 2), "(289) 270-0990", font=FX(58), fill=(255, 255, 255, int(255 * p3)), anchor="mm")
    col = tuple(int(c * p3 + 255 * (1 - p3)) for c in INK)
    d.text((W // 2, 1290), "mifoodstudio.com", font=FS(50), fill=col, anchor="ma")
    d.text((W // 2, 1370), "Unit 1, 6905 Millcreek Drive, Mississauga, ON", font=FM(36), fill=col, anchor="ma")
    d.text((W // 2, 1425), "Meadowvale Business Park", font=FM(36), fill=col, anchor="ma")
    d.text((W // 2, 1560), "The MiFoodStudio team wishes you a very Happy Thanksgiving", font=font("Inter-MediumItalic.otf", 32), fill=tuple(int(c * p3 + 255 * (1 - p3)) for c in ORANGE), anchor="ma")
    return im

def shot_frame(s, t):
    if s["f"] is None:
        return end_card(t)
    p = t / s["d"]
    fr = kenburns(load(s["f"]), s["a"], s["b"], p)
    fr = grade(fr, s.get("dark", 0.0))
    for (c0, c1, lines, kick) in s.get("caps", []):
        if c0 <= t < c1 + 0.001:
            fr = caption(fr, lines, t - c0, kick, s.get("sub"))
    return fr

# ---------------------------------------------------------------- render
def main():
    out = sys.argv[1]
    preview = "--preview" in sys.argv
    starts = []; t = 0
    for s in SHOTS:
        starts.append(t); t += s["d"]
    total = t
    nframes = int(total * FPS)
    if preview:
        for i, s in enumerate(SHOTS):
            shot_frame(s, min(1.2, s["d"] - 0.1)).save(f"prev_{i:02d}.jpg", quality=85)
            if s.get("caps") and len(s["caps"]) > 1:
                shot_frame(s, s["caps"][1][0] + 1.0).save(f"prev_{i:02d}b.jpg", quality=85)
        print("preview frames written"); return
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(nframes):
        tt = n / FPS
        i = max(k for k in range(len(SHOTS)) if starts[k] <= tt)
        s = SHOTS[i]; local = tt - starts[i]
        fr = shot_frame(s, local)
        # crossfade into next shot
        if i + 1 < len(SHOTS) and local > s["d"] - XFADE:
            q = (local - (s["d"] - XFADE)) / XFADE
            nxt = shot_frame(SHOTS[i + 1], 0.0)
            fr = Image.blend(fr, nxt, ease(q))
        pr.stdin.write(np.asarray(fr).tobytes())
        if n % 150 == 0: print(f"{n}/{nframes}", flush=True)
    pr.stdin.close(); pr.wait()
    print("done", total, "s")

if __name__ == "__main__":
    main()
