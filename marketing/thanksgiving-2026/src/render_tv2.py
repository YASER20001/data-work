"""MiFoodStudio Thanksgiving spot v2 — 30 s, 1920x1080, 2.39:1 letterbox.
Adds camera life: handheld drift, punch-in on cuts, steam, light sweeps,
candle flicker; hard cuts on the beat; no dissolves in the montage.
usage: python3 render_tv2.py out.mp4 [--preview]
"""
import sys, subprocess, glob, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance

SRC = "/home/user/data-work/"
S = "/tmp/claude-0/-home-user-data-work/6d3e5537-b273-5292-9458-851e6e4d1899/scratchpad/"
W, H, FPS = 1920, 1080, 30
PH = 804; PY = (H - PH) // 2
F = "/usr/share/fonts/opentype/inter/"
RED, ORANGE, CREAM, INK = (226, 29, 43), (242, 107, 29), (255, 247, 238), (28, 24, 22)
def font(n, s): return ImageFont.truetype(F + n, s)
FHEAD = lambda s: font("InterDisplay-SemiBold.otf", s)
FX = lambda s: font("InterDisplay-ExtraBold.otf", s)
FM = lambda s: font("Inter-Medium.otf", s)
FS = lambda s: font("Inter-SemiBold.otf", s)
def ease(x): return x * x * (3 - 2 * x)
def ease_out(x): return 1 - (1 - x) ** 3

# ------------------------------------------------------------------ sources
_cache = {}
def grade_img(im, mood):
    if mood == "cold":
        im = ImageEnhance.Color(im).enhance(0.42); im = ImageEnhance.Contrast(im).enhance(1.15)
        r, g, b = im.split(); r = r.point(lambda v: int(v * 0.92)); b = b.point(lambda v: min(255, int(v * 1.06 + 6)))
    else:
        im = ImageEnhance.Contrast(im).enhance(1.07); im = ImageEnhance.Color(im).enhance(1.14)
        r, g, b = im.split(); r = r.point(lambda v: min(255, int(v * 1.05 + 5))); b = b.point(lambda v: int(v * 0.92))
    return Image.merge("RGB", (r, g, b))

def still(fn, mood):
    key = (fn, mood)
    if key not in _cache:
        p = fn if fn.startswith("/") else SRC + fn
        im = grade_img(ImageOps.exif_transpose(Image.open(p)).convert("RGB"), mood)
        if im.width > 2600: im = im.resize((2600, int(im.height * 2600 / im.width)), Image.LANCZOS)
        _cache[key] = im
    return _cache[key]

HOME = sorted(glob.glob(S + "tv/home/*.png"))
def clip_frame(t, mood):
    im = Image.open(HOME[min(len(HOME) - 1, int(t * FPS))]).convert("RGB")
    return grade_img(im, mood)

# ------------------------------------------------------------------ camera
def handheld(t, amt=1.0, seed=0):
    """smooth pseudo-random drift in fractions of frame + tiny scale wobble."""
    s = seed * 1.37
    dx = (math.sin(2 * math.pi * 0.21 * t + s) * 0.55 + math.sin(2 * math.pi * 0.53 * t + 2 * s) * 0.3 + math.sin(2 * math.pi * 1.1 * t + 3 * s) * 0.15)
    dy = (math.cos(2 * math.pi * 0.17 * t + s) * 0.55 + math.cos(2 * math.pi * 0.47 * t + 2 * s) * 0.3 + math.sin(2 * math.pi * 0.9 * t + s) * 0.15)
    dz = math.sin(2 * math.pi * 0.13 * t + s) * 0.004
    return dx * 0.006 * amt, dy * 0.006 * amt, dz * amt

def kb(im, a, b, p, t, hh=1.0, seed=0, punch=0.0):
    p = ease(p)
    dx, dy, dz = handheld(t, hh, seed)
    cx = a[0] + (b[0] - a[0]) * p + dx; cy = a[1] + (b[1] - a[1]) * p + dy
    z = (a[2] + (b[2] - a[2]) * p) * (1 + dz) * (1 - punch)
    cw = im.width * z; ch = cw * PH / 1920
    if ch > im.height: ch = im.height * z; cw = ch * 1920 / PH
    x0 = min(max(cx * im.width - cw / 2, 0), im.width - cw)
    y0 = min(max(cy * im.height - ch / 2, 0), im.height - ch)
    return im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((1920, PH), Image.BILINEAR)

# ------------------------------------------------------------------ atmosphere
_steam = None
def steam_layers():
    global _steam
    if _steam is None:
        rng = np.random.default_rng(5); layers = []
        for k in range(3):
            n = rng.random((PH // 4, 1920 // 4)).astype(np.float32)
            im = Image.fromarray((n * 255).astype(np.uint8)).resize((1920, PH * 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(28 + 10 * k))
            a = np.asarray(im).astype(np.float32) / 255.0
            a = (a - a.min()) / (a.max() - a.min() + 1e-6)
            layers.append(a)
        _steam = layers
    return _steam

def steam(frame, t, strength=0.22, region=(0.35, 1.0)):
    """rising soft steam, screen-blended, strongest in lower part of frame."""
    a = frame.astype(np.float32)
    acc = np.zeros((PH, 1920), np.float32)
    for k, lay in enumerate(steam_layers()):
        speed = 90 + 40 * k
        off = int((t * speed) % PH)
        sl = np.roll(lay, -off, axis=0)[:PH]
        acc += sl * (0.5 + 0.25 * k)
    acc /= 2.0
    y = np.linspace(0, 1, PH)[:, None]
    mask = np.clip((y - region[0]) / (region[1] - region[0]), 0, 1) ** 1.2
    acc = acc * mask * strength
    a = a + (255 - a) * acc[..., None] * 0.9
    return a

def light_sweep(frame, t, t0, dur=1.6, strength=0.16):
    """diagonal soft highlight travelling left->right across stainless."""
    if not (t0 <= t <= t0 + dur): return frame
    q = (t - t0) / dur
    x = np.linspace(0, 1, 1920)[None, :]; y = np.linspace(0, 1, PH)[:, None]
    pos = -0.3 + 1.6 * q
    band = np.exp(-((x + 0.35 * y - pos) / 0.09) ** 2) * strength * math.sin(math.pi * q)
    return frame + (255 - frame) * band[..., None]

def flicker(frame, t, amt=0.05, seed=3):
    f = 1 + amt * (0.6 * math.sin(2 * math.pi * 7.3 * t + seed) + 0.4 * math.sin(2 * math.pi * 11.7 * t + 2 * seed) + 0.3 * math.sin(2 * math.pi * 2.1 * t))
    return frame * f

_vig = None
def vignette():
    global _vig
    if _vig is None:
        y, x = np.mgrid[0:PH, 0:1920]
        d = np.sqrt(((x - 960) / 960) ** 2 + ((y - PH / 2) / (PH / 2)) ** 2)
        _vig = np.clip(1 - 0.38 * np.clip(d - 0.55, 0, 1) ** 1.4, 0, 1)[..., None].astype(np.float32)
    return _vig
_grain = {}
def grain(i):
    if i not in _grain: _grain[i] = np.random.default_rng(i).normal(0, 1, (PH, 1920, 1)).astype(np.float32)
    return _grain[i]

# ------------------------------------------------------------------ shots
# (start, end, kind, args, supers, fx) ; fx keys: steam, sweep(t0), flicker, hh(amt), punch
B = 0.625  # beat
T0 = 6.2   # music start
SHOTS = [
    (0.0, 2.6, "clip", dict(mood="cold", a=(0.5, 0.5, 1.0), b=(0.5, 0.5, 0.93)),
        [(0.5, 2.4, "Thanksgiving. Twenty guests.", "super")], dict(hh=0.4)),
    (2.6, 4.8, "still", dict(f=S + "c_stress.png", mood="cold", a=(0.5, 0.40, 0.96), b=(0.5, 0.38, 0.80)),
        [(0.35, 2.0, "One oven.", "super")], dict(hh=1.6, punch=0.03)),
    (4.8, T0, "black", dict(), [(0.15, 1.35, "There's a bigger kitchen in Mississauga.", "center")], {}),
    (T0, T0 + 6 * B, "still", dict(f="20260808_143157.jpg", mood="warm", a=(0.42, 0.52, 0.80), b=(0.56, 0.52, 0.72)),
        [(1.9, 3.6, "MiFoodStudio", "brand")], dict(hh=0.7, sweep=0.5, steam=0.08, punch=0.04)),
    (T0 + 6 * B, T0 + 8 * B, "still", dict(f="Burner1.png", mood="warm", a=(0.44, 0.52, 1.0), b=(0.42, 0.50, 0.88)),
        [], dict(hh=0.8, steam=0.26, flicker=0.04, punch=0.05)),
    (T0 + 8 * B, T0 + 10 * B, "still", dict(f="Equpment Close3.png", mood="warm", a=(0.30, 0.55, 0.92), b=(0.44, 0.55, 0.82)),
        [], dict(hh=0.8, sweep=0.0, steam=0.12, punch=0.05)),
    (T0 + 10 * B, T0 + 12 * B, "still", dict(f="Oven.png", mood="warm", a=(0.62, 0.45, 1.0), b=(0.62, 0.44, 0.88)),
        [], dict(hh=0.8, steam=0.16, sweep=0.2, punch=0.05)),
    (T0 + 12 * B, T0 + 14 * B, "still", dict(f="Cooler.png", mood="warm", a=(0.74, 0.60, 0.62), b=(0.72, 0.58, 0.54)),
        [], dict(hh=0.8, punch=0.05)),
    (T0 + 14 * B, T0 + 18 * B, "still", dict(f=S + "c_cooking.png", mood="warm", a=(0.56, 0.45, 0.98), b=(0.60, 0.45, 0.84)),
        [(0.5, 2.3, "Cook your whole feast here.", "super")], dict(hh=1.2, flicker=0.02, punch=0.04)),
    (T0 + 18 * B, T0 + 22 * B, "still", dict(f="20260808_143445.jpg", mood="warm", a=(0.60, 0.52, 0.78), b=(0.40, 0.52, 0.78)),
        [(0.4, 2.2, "Room for everyone.", "super")], dict(hh=0.6, sweep=0.3, steam=0.07, punch=0.04)),
    (T0 + 22 * B, T0 + 30 * B, "still", dict(f=S + "c_turkey.png", mood="warm", a=(0.50, 0.58, 0.92), b=(0.52, 0.52, 0.76)),
        [(0.6, 2.5, "Then celebrate at home.", "super"), (2.8, 4.7, "No mess. No stress.", "super")],
        dict(hh=0.9, steam=0.30, flicker=0.06, punch=0.04)),
    (T0 + 30 * B, 30.0, "end", dict(), [], {}),
]

def picture(pic, fidx, mood, fx, t):
    a = np.asarray(pic).astype(np.float32)
    if fx.get("steam"): a = steam(a, t, fx["steam"], region=(0.3, 1.0) if fx["steam"] > 0.2 else (0.45, 1.0))
    if "sweep" in fx: a = light_sweep(a, t, fx["sweep"])
    if fx.get("flicker"): a = flicker(a, t, fx["flicker"])
    a = a * vignette() + grain(fidx % 8) * (7.0 if mood == "cold" else 3.5)
    if mood == "cold": a = a * 0.92 + 12
    return np.clip(a, 0, 255).astype(np.uint8)

def draw_supers(img, supers, t):
    d = ImageDraw.Draw(img, "RGBA")
    for (ti, to, text, style) in supers:
        if not (ti <= t <= to): continue
        a_in = ease_out(min((t - ti) / 0.5, 1)); a_out = ease(min((to - t) / 0.35, 1))
        al = int(255 * min(a_in, a_out))
        if style == "super":
            f = FHEAD(60); x, y = 120, PY + PH - 160 + int((1 - a_in) * 18)
            d.text((x + 2, y + 3), text, font=f, fill=(0, 0, 0, int(al * 0.6))); d.text((x, y), text, font=f, fill=(255, 255, 255, al))
        elif style == "brand":
            f = FX(80); x, y = 120, PY + PH - 190 + int((1 - a_in) * 18)
            d.text((x + 2, y + 3), text, font=f, fill=(0, 0, 0, int(al * 0.6))); d.text((x, y), text, font=f, fill=(255, 255, 255, al))
            d.text((x + 3, y + 96), "CLOUD & RENTAL KITCHEN  ·  MISSISSAUGA", font=FS(28), fill=(255, 210, 170, al))
        elif style == "center":
            f = FHEAD(66)
            d.text((W // 2, H // 2 + int((1 - a_in) * 10)), text, font=f, fill=(255, 255, 255, al), anchor="mm")
    return img

_logo = None
def logo():
    global _logo
    if _logo is None: _logo = Image.open(S + "logo_h.png").convert("RGBA")
    return _logo

def end_card(t):
    im = Image.new("RGB", (W, H), CREAM)
    glow = Image.new("RGB", (W, H), CREAM); gd = ImageDraw.Draw(glow)
    gd.ellipse((W - 900, -500, W + 300, 500), fill=(255, 226, 200)); gd.ellipse((-500, H - 500, 500, H + 400), fill=(255, 220, 196))
    im = Image.blend(im, glow.filter(ImageFilter.GaussianBlur(170)), 0.9)
    d = ImageDraw.Draw(im, "RGBA")
    p = ease_out(min(t / 0.7, 1))
    lg = logo(); lw = 760; lg = lg.resize((lw, int(lg.height * lw / lg.width)), Image.LANCZOS)
    tmp = Image.new("RGBA", lg.size); tmp.paste(lg, (0, 0)); tmp.putalpha(tmp.split()[3].point(lambda v: int(v * p)))
    im.paste(tmp, ((W - lw) // 2, 250 + int((1 - p) * 24)), tmp)
    # light sweep across the logo
    if 0.6 < t < 1.6:
        q = (t - 0.6) / 1.0
        a = np.asarray(im).astype(np.float32)
        x = np.linspace(0, 1, W)[None, :]; y = np.linspace(0, 1, H)[:, None]
        band = np.exp(-((x + 0.2 * y - (-0.2 + 1.4 * q)) / 0.05) ** 2) * 0.35 * math.sin(math.pi * q)
        ymask = np.exp(-((y - 0.3) / 0.12) ** 2)
        a = a + (255 - a) * (band * ymask)[..., None]
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im, "RGBA")
    p2 = ease_out(min(max((t - 0.5) / 0.6, 0), 1))
    d.text((W // 2, 470 + int((1 - p2) * 12)), "Book your Thanksgiving slot.", font=FHEAD(72), fill=INK + (int(255 * p2),), anchor="ma")
    p3 = ease_out(min(max((t - 1.0) / 0.6, 0), 1))
    pw, ph = 520, 96; px, py = (W - pw) // 2, 600
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=48, fill=RED + (int(255 * p3),))
    d.text((W // 2, py + ph // 2), "(289) 270-0990", font=FX(48), fill=(255, 255, 255, int(255 * p3)), anchor="mm")
    d.text((W // 2, 740), "mifoodstudio.com", font=FS(42), fill=INK + (int(255 * p3),), anchor="ma")
    d.text((W // 2, 805), "Unit 1, 6905 Millcreek Drive, Mississauga, ON  ·  Meadowvale Business Park", font=FM(30), fill=(90, 70, 60, int(255 * p3)), anchor="ma")
    p4 = ease_out(min(max((t - 1.6) / 0.8, 0), 1))
    d.text((W // 2, 930), "Happy Thanksgiving from the MiFoodStudio team", font=font("Inter-MediumItalic.otf", 30), fill=ORANGE + (int(255 * p4),), anchor="ma")
    return im

def radial_zoom(img, amount, steps=5):
    """cheap radial/zoom blur: average of progressively scaled copies."""
    base = img; acc = np.asarray(img).astype(np.float32)
    for k in range(1, steps):
        z = 1 + amount * k / (steps - 1)
        w, h = int(W * z), int(H * z)
        sc = base.resize((w, h), Image.BILINEAR).crop(((w - W) // 2, (h - H) // 2, (w - W) // 2 + W, (h - H) // 2 + H))
        acc += np.asarray(sc).astype(np.float32)
    return Image.fromarray(np.clip(acc / steps, 0, 255).astype(np.uint8))

def zoom_through(out_img, in_img, q):
    qe = ease(q)
    a = radial_zoom(out_img, 0.06 + 0.18 * qe)
    # incoming starts slightly zoomed-in and settles
    zi = 1 + 0.10 * (1 - qe)
    w, h = int(W * zi), int(H * zi)
    b = in_img.resize((w, h), Image.BILINEAR).crop(((w - W) // 2, (h - H) // 2, (w - W) // 2 + W, (h - H) // 2 + H))
    b = radial_zoom(b, 0.12 * (1 - qe))
    return Image.blend(a, b, qe)

def render_shot(sh, t, fidx):
    s0, s1, kind, a, supers, fx = sh
    if kind == "end": return end_card(t)
    if kind == "black": return draw_supers(Image.new("RGB", (W, H), 0), supers, t)
    mood = a["mood"]; p = t / (s1 - s0)
    punch = fx.get("punch", 0.0) * math.exp(-t / 0.22)
    src = clip_frame(t, mood) if kind == "clip" else still(a["f"], mood)
    pic = kb(src, a["a"], a["b"], p, s0 + t, fx.get("hh", 1.0), seed=int(s0 * 10), punch=punch)
    canvas = np.zeros((H, W, 3), np.uint8); canvas[PY:PY + PH] = picture(pic, fidx, mood, fx, t)
    return draw_supers(Image.fromarray(canvas), supers, t)

def main():
    out = sys.argv[1]; preview = "--preview" in sys.argv
    if preview:
        for i, sh in enumerate(SHOTS):
            L = sh[1] - sh[0]
            for k, tt in enumerate([min(0.9, L * 0.4), L * 0.8]):
                render_shot(sh, min(tt, L - 0.05), 0).save(S + f"tv2prev_{i:02d}{'ab'[k]}.jpg", quality=85)
        print("preview ok"); return
    total = SHOTS[-1][1]; n = int(total * FPS)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n):
        tt = f / FPS
        i = max(k for k, sh in enumerate(SHOTS) if sh[0] <= tt)
        sh = SHOTS[i]; local = tt - sh[0]
        img = render_shot(sh, local, f)
        # zoom-through transitions inside the warm act (shots 3..10): outgoing zooms in with
        # radial blur, incoming zooms out; a flowing move instead of a hard cut
        TD = 0.28
        L = sh[1] - sh[0]
        if 3 <= i <= 10 and local > L - TD and i + 1 < len(SHOTS) and SHOTS[i + 1][2] != "end":
            q = (local - (L - TD)) / TD
            nxt = render_shot(SHOTS[i + 1], 0.0, f)
            img = zoom_through(img, nxt, q)
        elif i == 10 and local > L - 0.4:
            q = (local - (L - 0.4)) / 0.4
            img = Image.blend(img, render_shot(SHOTS[i + 1], 0.0, f), ease(q))
        if tt < 0.5: img = Image.blend(Image.new("RGB", (W, H), 0), img, ease(tt / 0.5))
        if tt > total - 0.6: img = Image.blend(img, Image.new("RGB", (W, H), 0), ease((tt - (total - 0.6)) / 0.6))
        pr.stdin.write(np.asarray(img).tobytes())
        if f % 150 == 0: print(f"{f}/{n}", flush=True)
    pr.stdin.close(); pr.wait(); print("done")

if __name__ == "__main__":
    main()
