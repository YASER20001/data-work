"""Original warm acoustic-pop bed for the MiFoodStudio Thanksgiving reel.
Pure numpy additive synthesis -> 48k stereo WAV. ~33 s, 96 BPM, key of C.
"""
import numpy as np, wave, sys

SR = 48000
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
DUR = 33.0
N = int(DUR * SR)
t_all = np.arange(N) / SR

def note(f):  # midi -> hz
    return 440 * 2 ** ((f - 69) / 12)

def env(n, a, d, s, r, hold):
    """ADSR in seconds, hold = total note length."""
    out = np.zeros(n)
    i = np.arange(n) / SR
    att = np.clip(i / max(a, 1e-4), 0, 1)
    dec = np.where(i > a, s + (1 - s) * np.exp(-(i - a) / max(d, 1e-4)), 1)
    rel = np.where(i > hold, np.exp(-(i - hold) / max(r, 1e-4)), 1)
    return att * dec * rel

def place(buf, sig, start):
    i = int(start * SR)
    j = min(N, i + len(sig))
    if i < N:
        buf[i:j] += sig[: j - i]

# ---------- instruments ----------
def epiano(freq, length, vel=1.0):
    n = int((length + 1.5) * SR)
    i = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * freq * i)
            + 0.35 * np.sin(2 * np.pi * freq * 2 * i + 0.3)
            + 0.12 * np.sin(2 * np.pi * freq * 3 * i)
            + 0.05 * np.sin(2 * np.pi * freq * 5.01 * i))
    # gentle tremolo like a Rhodes
    tone *= 1 + 0.05 * np.sin(2 * np.pi * 4.5 * i)
    return tone * env(n, 0.004, 0.9, 0.35, 0.9, length) * 0.22 * vel

def pluck(freq, length, vel=1.0):
    """nylon-guitar-ish pluck via Karplus-Strong."""
    n = int((length + 1.0) * SR)
    period = int(SR / freq)
    buf = np.random.uniform(-1, 1, period)
    out = np.zeros(n)
    for k in range(n):
        out[k] = buf[k % period]
        buf[k % period] = 0.5 * (buf[k % period] + buf[(k + 1) % period]) * 0.996
    return out * env(n, 0.001, 0.4, 0.5, 0.5, length) * 0.9 * vel

def bass(freq, length, vel=1.0):
    n = int((length + 0.4) * SR)
    i = np.arange(n) / SR
    tone = np.sin(2 * np.pi * freq * i) + 0.25 * np.sin(2 * np.pi * freq * 2 * i)
    return np.tanh(1.6 * tone) * env(n, 0.006, 0.3, 0.6, 0.25, length) * 0.3 * vel

def pad(freqs, length):
    n = int((length + 2.0) * SR)
    i = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for det in (-0.4, 0.0, 0.4):
            out += np.sin(2 * np.pi * (f + det) * i + np.random.rand() * 6)
    out /= len(freqs) * 3
    return out * env(n, 1.2, 1.0, 0.8, 1.8, length) * 0.16

def kick(vel=1.0):
    n = int(0.35 * SR)
    i = np.arange(n) / SR
    f = 55 + 90 * np.exp(-i * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-i * 14) * 0.75 * vel

def shaker(vel=1.0):
    n = int(0.09 * SR)
    noise = np.random.uniform(-1, 1, n)
    # crude high-pass: difference
    noise = np.diff(noise, prepend=0)
    return noise * np.exp(-np.arange(n) / SR * 70) * 0.42 * vel

def rim(vel=1.0):
    n = int(0.12 * SR)
    i = np.arange(n) / SR
    return (np.sin(2 * np.pi * 820 * i) * np.exp(-i * 60) + np.random.uniform(-1, 1, n) * np.exp(-i * 90) * 0.9) * 0.36 * vel

# ---------- arrangement ----------
# chords (midi): C  | G/B | Am  | F   (two bars each phrase = 8 bars; loop)
CHORDS = [
    ([60, 64, 67, 71], 48),   # Cmaj7
    ([59, 62, 67, 71], 43),   # G/B
    ([57, 60, 64, 67], 45),   # Am7
    ([57, 60, 65, 69], 41),   # Fmaj7
]
L = np.zeros(N); R = np.zeros(N)
np.random.seed(7)

bars = int(DUR / BAR) + 1
for b in range(bars):
    t0 = b * BAR
    chord, root = CHORDS[(b // 2) % 4]
    intro = b < 2            # first 2 bars: pad + sparse piano only
    outro = t0 > DUR - 4.5

    # pad
    if b % 2 == 0:
        p = pad([note(m) for m in chord], 2 * BAR)
        place(L, p, t0); place(R, p * 0.9, t0 + 0.012)

    # e-piano chord stabs: beat 1 and the "and" of 2, plus arpeggio on beat 3
    vel = 0.7 if intro else 1.0
    for off, v in [(0, 1.0), (1.5 * BEAT, 0.6), (3.0 * BEAT, 0.5)]:
        for k, m in enumerate(chord):
            s = epiano(note(m), 0.9 * BEAT if off else 1.4 * BEAT, v * vel)
            pan = (k - 1.5) / 3
            place(L, s * (1 - 0.35 * pan), t0 + off + k * 0.012)
            place(R, s * (1 + 0.35 * pan), t0 + off + k * 0.012)

    # guitar pluck melody (pentatonic figure over each chord), from bar 2
    if not intro and not outro:
        figure = [chord[2] + 12, chord[1] + 12, chord[3], chord[2] + 12, chord[0] + 12, chord[1] + 12]
        times = [0, 0.5, 1.0, 2.0, 2.5, 3.5]
        for m, bt in zip(figure, times):
            if np.random.rand() < 0.85:
                s = pluck(note(m), 0.6 * BEAT, 0.8)
                place(L, s * 0.7, t0 + bt * BEAT); place(R, s * 1.0, t0 + bt * BEAT + 0.008)

    # bass: root on 1, octave/fifth on the "and" of 3
    if not intro:
        s = bass(note(root), 1.7 * BEAT); place(L, s, t0); place(R, s, t0)
        s = bass(note(root + 7), 0.7 * BEAT, 0.7); place(L, s, t0 + 2.5 * BEAT); place(R, s, t0 + 2.5 * BEAT)
        s = bass(note(root), 0.8 * BEAT, 0.8); place(L, s, t0 + 3.5 * BEAT); place(R, s, t0 + 3.5 * BEAT)

    # drums: soft kick 1 & 3, rim on 2 & 4, shaker 8ths
    if not intro:
        for bt in (0, 2):
            k = kick(); place(L, k, t0 + bt * BEAT); place(R, k, t0 + bt * BEAT)
        for bt in (1, 3):
            r = rim(0.9); place(L, r * 0.8, t0 + bt * BEAT); place(R, r, t0 + bt * BEAT)
        for e in range(8):
            v = 1.0 if e % 2 == 0 else 0.55
            s = shaker(v); place(L, s, t0 + e * BEAT / 2 + 0.004); place(R, s * 0.8, t0 + e * BEAT / 2)
        for e in range(16):
            hn = int(0.03 * SR); hh = np.diff(np.random.uniform(-1, 1, hn), prepend=0) * np.exp(-np.arange(hn) / SR * 160) * (0.22 if e % 4 == 2 else 0.1)
            place(L, hh, t0 + e * BEAT / 4); place(R, hh, t0 + e * BEAT / 4 + 0.002)

# final hit on last chord
tf = (bars - 1) * BAR
for k, m in enumerate([60, 64, 67, 72]):
    s = epiano(note(m), 3.0, 1.0); place(L, s, tf + k * 0.03); place(R, s, tf + k * 0.03)

# master: fade in/out, normalise
mix = np.stack([L, R], 1)
fade_in = np.clip(t_all / 0.8, 0, 1)[:, None]
fade_out = np.clip((DUR - t_all) / 2.5, 0, 1)[:, None]
mix = mix * fade_in * fade_out
mix = np.tanh(mix * 1.3)
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 0.85
pcm = (mix * 32767).astype(np.int16)
with wave.open(sys.argv[1] if len(sys.argv) > 1 else "music_raw.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("wrote", DUR, "s")
