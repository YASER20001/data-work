"""MiFoodStudio Thanksgiving TV spot — 30 s, 1920x1080, 2.39:1 letterbox, film grain.
Three acts: the problem (cold), the reveal (warm), the payoff; then the ask.
usage: python3 render_tv.py out.mp4 [--preview]
"""
import sys, subprocess, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance

SRC = "/home/user/data-work/"
S = "/tmp/claude-0/-home-user-data-work/6d3e5537-b273-5292-9458-851e6e4d1899/scratchpad/"
W, H, FPS = 1920, 1080, 30
PH = 804                      # picture height inside the letterbox
PY = (H - PH) // 2            # top bar height (138)
F = "/usr/share/fonts/opentype/inter/"
RED, ORANGE, CREAM, INK = (226, 29, 43), (242, 107, 29), (255, 247, 238), (28, 24, 22)
def font(n, s): return ImageFont.truetype(F + n, s)
FSUP = lambda s: font("Inter-Medium.otf", s)        # supers: quiet, medium weight
FHEAD = lambda s: font("InterDisplay-SemiBold.otf", s)
FX = lambda s: font("InterDisplay-ExtraBold.otf", s)
FM = lambda s: font("Inter-Medium.otf", s)
FS = lambda s: font("Inter-SemiBold.otf", s)

def ease(x): return x * x * (3 - 2 * x)

# ---------------------------------------------------------------- sources
_cache = {}
def still(fn, mood):
    key = (fn, mood)
    if key not in _cache:
        p = fn if fn.startswith("/") else SRC + fn
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        if mood == "cold":
            im = ImageEnhance.Color(im).enhance(0.45)
            im = ImageEnhance.Contrast(im).enhance(1.15)
            r, g, b = im.split()
            r = r.point(lambda v: int(v * 0.92)); b = b.point(lambda v: min(255, int(v * 1.06 + 6)))
            im = Image.merge("RGB", (r, g, b))
        else:
            im = ImageEnhance.Contrast(im).enhance(1.06); im = ImageEnhance.Color(im).enhance(1.12)
            r, g, b = im.split()
            r = r.point(lambda v: min(255, int(v * 1.05 + 5))); b = b.point(lambda v: int(v * 0.93))
            im = Image.merge("RGB", (r, g, b))
        if im.width > 2600:
            im = im.resize((2600, int(im.height * 2600 / im.width)), Image.LANCZOS)
        _cache[key] = im
    return _cache[key]

def seq(name):
    return sorted(glob.glob(S + f"tv/{name}/*.png"))
SEQ = {"home": seq("home"), "dinner": seq("dinner")}

def kb(im, a, b, p):
    """Ken Burns crop to 1920x804. a/b = (cx, cy, zoom) zoom = fraction of width covered."""
    p = ease(p)
    cx = a[0] + (b[0] - a[0]) * p; cy = a[1] + (b[1] - a[1]) * p; z = a[2] + (b[2] - a[2]) * p
    cw = im.width * z; ch = cw * PH / 1920
    if ch > im.height: ch = im.height * z; cw = ch * 1920 / PH
    x0 = min(max(cx * im.width - cw / 2, 0), im.width - cw)
    y0 = min(max(cy * im.height - ch / 2, 0), im.height - ch)
    return im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((1920, PH), Image.BILINEAR)

def clip_frame(name, t, mood, zoom=(1.0, 1.06)):
    frames = SEQ[name]
    idx = min(len(frames) - 1, int(t * FPS))
    im = Image.open(frames[idx]).convert("RGB")
    if mood == "cold":
        im = ImageEnhance.Color(im).enhance(0.5); im = ImageEnhance.Contrast(im).enhance(1.12)
        r, g, b = im.split(); r = r.point(lambda v: int(v * 0.93)); b = b.point(lambda v: min(255, int(v * 1.05 + 6)))
        im = Image.merge("RGB", (r, g, b))
    else:
        im = ImageEnhance.Color(im).enhance(1.08)
        r, g, b = im.split(); r = r.point(lambda v: min(255, int(v * 1.04 + 4))); b = b.point(lambda v: int(v * 0.94))
        im = Image.merge("RGB", (r, g, b))
    return im

# ---------------------------------------------------------------- shots
# each: (start, end, kind, args, supers) ; supers = list of (t_in, t_out, text, style)
SHOTS = [
    (0.0,  3.0, "clip",  dict(name="home", mood="cold", a=(0.5, 0.5, 1.0), b=(0.5, 0.5, 0.92)),
        [(0.7, 2.9, "Thanksgiving. Twenty guests.", "super")]),
    (3.0,  6.0, "still", dict(f=S + "c_stress.png", mood="cold", a=(0.5, 0.42, 1.0), b=(0.5, 0.40, 0.86)),
        [(0.4, 2.8, "One oven.", "super")]),
    (6.0,  7.0, "black", dict(), [(0.15, 1.0, "There's a bigger kitchen in Mississauga.", "center")]),
    (7.0, 10.5, "still", dict(f="20260808_143157.jpg", mood="warm", a=(0.40, 0.52, 0.80), b=(0.58, 0.52, 0.74)),
        [(1.6, 3.5, "MiFoodStudio", "brand")]),
    (10.5, 12.0, "still", dict(f="Burner1.png", mood="warm", a=(0.45, 0.55, 1.0), b=(0.42, 0.52, 0.86)), []),
    (12.0, 13.5, "still", dict(f="Oven.png", mood="warm", a=(0.62, 0.45, 1.0), b=(0.62, 0.45, 0.88)), []),
    (13.5, 15.0, "still", dict(f="Cooler.png", mood="warm", a=(0.65, 0.62, 1.0), b=(0.70, 0.64, 0.86)), []),
    (15.0, 17.5, "still", dict(f=S + "c_cooking.png", mood="warm", a=(0.55, 0.45, 0.98), b=(0.58, 0.45, 0.86)),
        [(0.4, 2.5, "Cook your whole feast here.", "super")]),
    (17.5, 20.0, "still", dict(f="Main Prep Area1.png", mood="warm", a=(0.50, 0.55, 1.0), b=(0.55, 0.55, 0.86)),
        [(0.3, 2.5, "Room for everyone.", "super")]),
    (20.0, 24.6, "clip",  dict(name="dinner", mood="warm", a=(0.5, 0.5, 1.0), b=(0.5, 0.5, 0.94)),
        [(0.6, 2.6, "Then celebrate at home.", "super"), (2.7, 4.6, "No mess. No stress.", "super")]),
    (24.6, 30.0, "end", dict(), []),
]
XF = 0.3   # crossfade only inside the warm acts; cold->black->warm are hard cuts

# ---------------------------------------------------------------- compositing
_grain = [None] * 8
def grain(i):
    if _grain[i] is None:
        rng = np.random.default_rng(i)
        _grain[i] = rng.normal(0, 1, (PH, 1920, 1)).astype(np.float32)
    return _grain[i]

_vig = None
def vignette():
    global _vig
    if _vig is None:
        y, x = np.mgrid[0:PH, 0:1920]
        d = np.sqrt(((x - 960) / 960) ** 2 + ((y - PH / 2) / (PH / 2)) ** 2)
        _vig = np.clip(1 - 0.35 * np.clip(d - 0.6, 0, 1) ** 1.4, 0, 1)[..., None].astype(np.float32)
    return _vig

def picture(im, fidx, mood):
    a = np.asarray(im).astype(np.float32)
    a = a * vignette()
    g = grain(fidx % 8) * (7.0 if mood == "cold" else 4.0)
    a = a + g
    # cold act: slightly lifted blacks (filmic)
    if mood == "cold":
        a = a * 0.92 + 12
    return np.clip(a, 0, 255).astype(np.uint8)

def frame_canvas():
    return np.zeros((H, W, 3), np.uint8)

def put_picture(canvas, pic):
    canvas[PY:PY + PH] = pic
    return canvas

def draw_supers(img, supers, t):
    d = ImageDraw.Draw(img, "RGBA")
    for (ti, to, text, style) in supers:
        if not (ti <= t <= to): continue
        a_in = ease(min((t - ti) / 0.45, 1)); a_out = ease(min((to - t) / 0.4, 1))
        al = int(255 * min(a_in, a_out))
        if style == "super":
            f = FHEAD(60)
            x, y = 120, PY + PH - 160 + int((1 - a_in) * 14)
            d.text((x + 2, y + 3), text, font=f, fill=(0, 0, 0, int(al * 0.6)))
            d.text((x, y), text, font=f, fill=(255, 255, 255, al))
        elif style == "brand":
            f = FX(80)
            x, y = 120, PY + PH - 190 + int((1 - a_in) * 14)
            d.text((x + 2, y + 3), text, font=f, fill=(0, 0, 0, int(al * 0.6)))
            d.text((x, y), text, font=f, fill=(255, 255, 255, al))
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
    p = ease(min(t / 0.6, 1))
    lg = logo(); lw = 760; lg = lg.resize((lw, int(lg.height * lw / lg.width)), Image.LANCZOS)
    tmp = Image.new("RGBA", lg.size); tmp.paste(lg, (0, 0));
    alpha = tmp.split()[3].point(lambda v: int(v * p)); tmp.putalpha(alpha)
    im.paste(tmp, ((W - lw) // 2, 250 + int((1 - p) * 20)), tmp)
    p2 = ease(min(max((t - 0.5) / 0.6, 0), 1))
    d.text((W // 2, 470), "Book your Thanksgiving slot.", font=FHEAD(72), fill=INK + (int(255 * p2),), anchor="ma")
    p3 = ease(min(max((t - 1.0) / 0.6, 0), 1))
    pw, ph = 520, 96; px, py = (W - pw) // 2, 600
    d.rounded_rectangle((px, py, px + pw, py + ph), radius=48, fill=RED + (int(255 * p3),))
    d.text((W // 2, py + ph // 2), "(289) 270-0990", font=FX(48), fill=(255, 255, 255, int(255 * p3)), anchor="mm")
    d.text((W // 2, 740), "mifoodstudio.com", font=FS(42), fill=INK + (int(255 * p3),), anchor="ma")
    d.text((W // 2, 805), "Unit 1, 6905 Millcreek Drive, Mississauga, ON  ·  Meadowvale Business Park", font=FM(30), fill=(90, 70, 60, int(255 * p3)), anchor="ma")
    p4 = ease(min(max((t - 1.6) / 0.8, 0), 1))
    d.text((W // 2, 930), "Happy Thanksgiving from the MiFoodStudio team", font=font("Inter-MediumItalic.otf", 30), fill=ORANGE + (int(255 * p4),), anchor="ma")
    return im

def render_shot(sh, t, fidx):
    s0, s1, kind, a, supers = sh
    if kind == "end": return end_card(t)
    if kind == "black":
        img = Image.fromarray(frame_canvas())
        return draw_supers(img, supers, t)
    mood = a["mood"]; p = t / (s1 - s0)
    if kind == "clip":
        im = clip_frame(a["name"], t, mood)
        pic = kb(im, a["a"], a["b"], p)
    else:
        pic = kb(still(a["f"], mood), a["a"], a["b"], p)
    canvas = put_picture(frame_canvas(), picture(pic, fidx, mood))
    img = Image.fromarray(canvas)
    return draw_supers(img, supers, t)

def main():
    out = sys.argv[1]; preview = "--preview" in sys.argv
    if preview:
        for i, sh in enumerate(SHOTS):
            s0, s1 = sh[0], sh[1]
            for k, tt in enumerate([0.9, (s1 - s0) * 0.75]):
                render_shot(sh, min(tt, s1 - s0 - 0.05), 0).save(S + f"tvprev_{i:02d}{'ab'[k]}.jpg", quality=85)
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
        # crossfade within warm acts only (shots 3..9 -> next), and into the end card
        if i + 1 < len(SHOTS) and i >= 3 and local > (sh[1] - sh[0]) - XF:
            q = (local - ((sh[1] - sh[0]) - XF)) / XF
            img = Image.blend(img, render_shot(SHOTS[i + 1], 0.0, f), ease(q))
        # fade from black at very start, fade to black at the very end
        if tt < 0.6: img = Image.blend(Image.new("RGB", (W, H), 0), img, ease(tt / 0.6))
        if tt > total - 0.6: img = Image.blend(img, Image.new("RGB", (W, H), 0), ease((tt - (total - 0.6)) / 0.6))
        pr.stdin.write(np.asarray(img).tobytes())
        if f % 150 == 0: print(f"{f}/{n}", flush=True)
    pr.stdin.close(); pr.wait(); print("done")

if __name__ == "__main__":
    main()
