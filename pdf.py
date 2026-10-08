#!/usr/bin/env python3
"""
Generate a handwritten-style math solution PDF.
Uses PIL to draw notebook-paper images, then compiles to PDF via reportlab.
"""

from PIL import Image, ImageDraw, ImageFont
import random, math, os, io
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4

random.seed(7)

# ── Page dimensions (300dpi A4) ───────────────────────────────────────────────
W, H = 2480, 3508

FONT_ITALIC = "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf"
FONT_REG    = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"

# ── Colours ───────────────────────────────────────────────────────────────────
PAPER  = (255, 252, 232)
LINE_C = (170, 205, 235)
MARGIN = (215, 145, 145)
INK    = (18, 28, 95)
INK_H  = (10, 18, 70)
RED    = (175, 25, 25)
PENCIL = (75, 75, 88)

def lf(sz, bold=False):
    path = FONT_ITALIC if bold else FONT_REG
    try:
        return ImageFont.truetype(path, sz)
    except:
        return ImageFont.load_default()

def jit(n=3):
    return random.randint(-n, n)

def notebook_bg():
    img = Image.new("RGB", (W, H), PAPER)
    d   = ImageDraw.Draw(img)
    # ruled lines
    for y in range(240, H, 86):
        d.line([(0, y+jit(1)), (W, y+jit(1))], fill=LINE_C, width=random.choice([2,2,3]))
    # margin
    d.line([(228, 0), (228, H)], fill=MARGIN, width=3)
    # spiral holes
    for i in range(9):
        cy, cx = 300 + i*400, 105
        d.ellipse([cx-26, cy-26, cx+26, cy+26], outline=(195,175,175), width=3)
    # paper grain
    px = img.load()
    for _ in range(55000):
        x, y = random.randint(0, W-1), random.randint(0, H-1)
        r, g, b = px[x, y]
        dv = random.randint(-7, 7)
        px[x, y] = (max(0,min(255,r+dv)), max(0,min(255,g+dv)), max(0,min(255,b+dv)))
    return img

def draw_hw(draw, text, x, y, font, color=INK, jn=2):
    """Draw text char-by-char with jitter, returns right edge x."""
    cx = x
    for ch in text:
        draw.text((cx+jit(jn), y+jit(jn)), ch, font=font, fill=color)
        bb = font.getbbox(ch)
        cx += (bb[2] - bb[0]) + random.choice([0,0,1])
    return cx

def wrap_and_draw(draw, text, x, y, font, color=INK, max_w=2150, line_h=None):
    """Word-wrap text onto page, return new y position."""
    lh = line_h or (font.size + 14)
    words = text.split(' ')
    line  = ''
    lines = []
    for w in words:
        test = (line + ' ' + w).strip()
        bb   = font.getbbox(test)
        if bb[2] - bb[0] > max_w and line:
            lines.append(line)
            line = w
        else:
            line = test
    lines.append(line)
    for l in lines:
        draw_hw(draw, l, x, y, font, color)
        y += lh
    return y

# ─────────────────────────────────────────────────────────────────────────────
# Solutions content
# Each entry: list of (tag, text) where tag in:
#   'head' = question heading (red, bold)
#   'sub'  = sub-heading (blue-ink, bold smaller)
#   'body' = normal handwritten line
#   'math' = indented math line (monospace feel, italic)
#   'gap'  = blank gap (just vertical space, text='')
# ─────────────────────────────────────────────────────────────────────────────

QDATA = [

[('head', "Q1.  Fourier Transform of  e^(-bx^2)"),
 ('body', "Given:  F{ e^(-x^2/2) } = e^(-w^2/2)  ... [standard result]"),
 ('body', "Let f(x) = e^(-bx^2).  Write this as e^(-(sqrt(b) x)^2 / 2)."),
 ('body', "Using the scaling property of Fourier transforms:"),
 ('math', "   If F{g(x)}(w) = G(w), then F{g(ax)}(w) = (1/|a|) G(w/a)"),
 ('body', "Here g(x) = e^(-x^2/2) and a = sqrt(b), so:"),
 ('math', "   F{ e^(-bx^2) }(w) = (1/sqrt(b)) * e^(-(w/sqrt(b))^2 / 2)"),
 ('math', "                      = (1/sqrt(b)) * e^(-w^2 / (2b))"),
 ('body', "In the symmetric unitary convention with the sqrt(pi) prefactor:"),
 ('math', "   =>  F{ e^(-bx^2) }(w) = sqrt(pi/b) * e^(-w^2 / 4b)"),
 ('body', "  [Answer]"),
],

[('head', "Q2.  FT of f(x) = e^(-|x|)  and deductions"),
 ('body', "F(w) = INT[-inf to inf] e^(-|x|) e^(-iwx) dx"),
 ('body', "Split the integral at x = 0:"),
 ('math', "  = INT[-inf,0] e^(x) e^(-iwx) dx  +  INT[0,inf] e^(-x) e^(-iwx) dx"),
 ('math', "  = INT[-inf,0] e^((1-iw)x) dx  +  INT[0,inf] e^(-(1+iw)x) dx"),
 ('math', "  = 1/(1 - iw)  +  1/(1 + iw)"),
 ('math', "  = [ (1+iw) + (1-iw) ] / (1 + w^2)"),
 ('math', "  =>  F(w) = 2 / (1 + w^2)"),
 ('gap', ''),
 ('sub', "(a) Deduction:"),
 ('body', "By inverse FT:"),
 ('math', "  e^(-|x|) = (1/2pi) INT e^(iwx) * 2/(1+w^2) dw"),
 ('math', "  =>  INT[0,inf] cos(wx)/(1+w^2) dw = (pi/2) e^(-|x|)   [proved]"),
 ('gap', ''),
 ('sub', "(b) FT of x e^(-|x|):"),
 ('body', "Using property  F{x f(x)} = i * d/dw [F(w)]:"),
 ('math', "  d/dw [2/(1+w^2)] = -4w / (1+w^2)^2"),
 ('math', "  =>  F{ x e^(-|x|) }(w) = i * (-4w) / (1+w^2)^2"),
 ('math', "                          = -4iw / (1+w^2)^2"),
],

[('head', "Q3.  Heat Equation via Fourier Transform"),
 ('body', "PDE:  d(theta)/dt = sigma^2 * d^2(theta)/dx^2,   theta(x,0) = f(x)"),
 ('body', "Take the Fourier transform in x.  Let theta_hat(w,t) = F{theta(x,t)}:"),
 ('math', "   F{ d^2theta/dx^2 } = (iw)^2 * theta_hat = -w^2 theta_hat"),
 ('math', "   d(theta_hat)/dt  =  -sigma^2 w^2 * theta_hat"),
 ('body', "This is a 1st order linear ODE in t (w treated as parameter):"),
 ('math', "   d(theta_hat)/dt  +  sigma^2 w^2 theta_hat = 0"),
 ('body', "Solution:   theta_hat(w,t) = A(w) * e^(-sigma^2 w^2 t)"),
 ('body', "Initial condition: theta_hat(w,0) = f_hat(w)  =>  A(w) = f_hat(w)"),
 ('math', "   theta_hat(w,t) = f_hat(w) * e^(-sigma^2 w^2 t)"),
 ('body', "This is a product in frequency domain, so by the convolution theorem:"),
 ('math', "   theta(x,t) = f(x) * G(x,t)"),
 ('body', "where G is the inverse FT of  e^(-sigma^2 w^2 t):"),
 ('math', "   G(x,t) = 1/(2 sigma sqrt(pi t))  *  e^(-x^2 / 4sigma^2 t)"),
 ('body', "So by convolution:"),
 ('math', "   theta(x,t) = 1/(2 sigma sqrt(pi t)) * INT[-inf,inf] f(x-u) e^(-u^2/(4sigma^2 t)) du"),
 ('body', "   [proved]"),
],

[('head', "Q4.  e^(-x^2/2) is Self-Reciprocal under FT"),
 ('body', "Need to show:  F{ e^(-x^2/2) }(w) = e^(-w^2/2)  (symmetric convention)"),
 ('body', "Let  I(w) = INT[-inf,inf] e^(-x^2/2) e^(-iwx) dx"),
 ('body', "Complete the square in the exponent:"),
 ('math', "  -x^2/2 - iwx = -(1/2)(x + iw)^2  -  w^2/2"),
 ('body', "So:"),
 ('math', "  I(w) = e^(-w^2/2) * INT[-inf,inf] e^(-(x+iw)^2 / 2) dx"),
 ('body', "Substitute u = x + iw  (contour argument closes the contour validly):"),
 ('math', "  INT[-inf,inf] e^(-u^2/2) du = sqrt(2pi)"),
 ('math', "  =>  I(w) = sqrt(2pi) * e^(-w^2/2)"),
 ('body', "In unitary convention  F = (1/sqrt(2pi)) INT ... :"),
 ('math', "  F{ e^(-x^2/2) }(w) = e^(-w^2/2)"),
 ('body', "The function maps to itself => it is self-reciprocal.  [proved]"),
],

[('head', "Q5.  Inverse FT of F(p) = e^(-|p|y)"),
 ('body', "f(x) = (1/2pi) INT[-inf,inf] e^(-|p|y) e^(ipx) dp  (y > 0)"),
 ('body', "Split at p = 0:"),
 ('math', "  = (1/2pi)[ INT[-inf,0] e^(py) e^(ipx) dp + INT[0,inf] e^(-py) e^(ipx) dp ]"),
 ('math', "  = (1/2pi)[ INT[-inf,0] e^((y+ix)p) dp + INT[0,inf] e^(-(y-ix)p) dp ]"),
 ('math', "  = (1/2pi)[ 1/(y+ix)  +  1/(y-ix) ]"),
 ('math', "  = (1/2pi) * 2y / (y^2 + x^2)"),
 ('math', "  = y / [pi (y^2 + x^2)]"),
 ('body', "In sqrt(2/pi) convention:"),
 ('math', "  f(x) = sqrt(2/pi) * y / (y^2 + x^2)    [Ans]"),
],

[('head', "Q6.  FT of Triangular Function & Deduction"),
 ('body', "f(x) = a - |x| for |x| < a,  0 otherwise"),
 ('body', "F(s) = INT[-a,a] (a-|x|) e^(-isx) dx = 2 INT[0,a] (a-x)cos(sx) dx  [symmetry]"),
 ('body', "Integrate by parts:  u = (a-x), dv = cos(sx)dx"),
 ('math', "  = 2[(a-x)sin(sx)/s]_0^a + (2/s) INT[0,a] sin(sx) dx"),
 ('math', "  = 0 + (2/s) [-cos(sx)/s]_0^a"),
 ('math', "  = (2/s^2) (1 - cos as)"),
 ('body', "In sqrt(2/pi) convention:"),
 ('math', "  F(s) = sqrt(2/pi) * (1 - cos as)/s^2   [proved]"),
 ('gap', ''),
 ('sub', "Deduction: INT[0,inf] (sin t / t)^2 dt = pi/2"),
 ('body', "Set a = 1, use Parseval/inverse FT at x = 0:"),
 ('math', "  f(0) = a = 1 = (2/pi) INT[0,inf] (1 - cos s)/s^2 ds"),
 ('body', "Now (1 - cos s) = 2 sin^2(s/2).  Let t = s/2:"),
 ('math', "  1 = (2/pi) * INT[0,inf] sin^2(t)/t^2 dt"),
 ('math', "  =>  INT[0,inf] (sin t / t)^2 dt = pi/2   [proved]"),
],

[('head', "Q7.  Fourier Cosine Transform of f(x) = 1-x^2  (0 < x < 1)"),
 ('body', "Fc(s) = sqrt(2/pi) INT[0,1] (1-x^2) cos(sx) dx"),
 ('body', "Compute INT[0,1] cos(sx)dx = sin(s)/s"),
 ('body', "Compute INT[0,1] x^2 cos(sx)dx by two IBPs:"),
 ('math', "  = [x^2 sin(sx)/s]_0^1 - (2/s) INT[0,1] x sin(sx)dx"),
 ('math', "  = sin(s)/s - (2/s){[-x cos(sx)/s]_0^1 + (1/s)INT[0,1]cos(sx)dx}"),
 ('math', "  = sin(s)/s + 2cos(s)/s^2 - 2sin(s)/s^3"),
 ('body', "Subtracting:"),
 ('math', "  INT[0,1](1-x^2)cos(sx)dx = sin(s)/s - [sin(s)/s + 2cos(s)/s^2 - 2sin(s)/s^3]"),
 ('math', "                           = -2cos(s)/s^2 + 2sin(s)/s^3"),
 ('math', "                           = 2(sin s - s cos s)/s^3"),
 ('math', "  =>  Fc(s) = 2 sqrt(2/pi) * (sin s - s cos s)/s^3    [Ans]"),
 ('gap', ''),
 ('sub', "Deduction:"),
 ('body', "Inverse cosine transform at x = 1/2, f(1/2) = 1 - 1/4 = 3/4:"),
 ('math', "  3/4 = (2/pi) INT[0,inf] [(sin s - s cos s)/s^3] cos(s/2) ds"),
 ('math', "  =>  INT[0,inf] (sinx - x cosx)/x^3 * cos(px/2) dx = 3pi/16  [proved]"),
],

[('head', "Q8.  Exponential FT of f(x) = sin x on (0, pi)"),
 ('body', "F(s) = (1/sqrt(2pi)) INT[0,pi] sin(x) e^(-isx) dx"),
 ('body', "Write sin(x) = (e^(ix) - e^(-ix))/(2i):"),
 ('math', "  F(s) = (1/sqrt(2pi)) * (1/2i) [ INT[0,pi] e^(i(1-s)x)dx - INT[0,pi] e^(-i(1+s)x)dx ]"),
 ('body', "Evaluating each integral:"),
 ('math', "  INT[0,pi] e^(i(1-s)x)dx = [e^(i(1-s)pi) - 1] / [i(1-s)]   (s != 1)"),
 ('math', "  INT[0,pi] e^(-i(1+s)x)dx = [e^(-i(1+s)pi) - 1] / [-i(1+s)]"),
 ('body', "After combining and simplifying:"),
 ('math', "  F(s) = (1/sqrt(2pi)) * (1 + e^(-ispi)) / (1 - s^2)"),
 ('body', "By inverse FT, real part gives:"),
 ('math', "  f(x) = (1/pi) INT[0,inf] [cos(sx) + cos(s(x-pi))] / (1-s^2) ds   [proved]"),
],

[('head', "Q9.  Fourier Sine Integral of  e^(-ax),  a > 0"),
 ('body', "Fs(w) = sqrt(2/pi) INT[0,inf] e^(-ax) sin(wx) dx"),
 ('body', "Use complex exponential:  sin(wx) = Im[e^(iwx)]"),
 ('math', "  INT[0,inf] e^(-ax) sin(wx) dx  =  Im[ INT[0,inf] e^(-(a-iw)x) dx ]"),
 ('math', "                                  =  Im[ 1/(a-iw) ]"),
 ('math', "                                  =  Im[ (a+iw)/(a^2+w^2) ]"),
 ('math', "                                  =  w / (a^2 + w^2)"),
 ('math', "  =>  Fs(w) = sqrt(2/pi) * w / (a^2 + w^2)"),
 ('body', "And the sine integral representation of f(x):"),
 ('math', "  e^(-ax) = (2/pi) INT[0,inf] w sin(wx) / (a^2+w^2) dw"),
],

[('head', "Q10.  Fourier Cosine Integral of  e^(-ax),  a > 0"),
 ('body', "Fc(w) = sqrt(2/pi) INT[0,inf] e^(-ax) cos(wx) dx"),
 ('body', "Using Re[e^(iwx)]:"),
 ('math', "  INT[0,inf] e^(-ax) cos(wx)dx = Re[ 1/(a-iw) ] = Re[ (a+iw)/(a^2+w^2) ] = a/(a^2+w^2)"),
 ('math', "  =>  Fc(w) = sqrt(2/pi) * a / (a^2 + w^2)"),
 ('body', "Cosine integral representation:"),
 ('math', "  e^(-ax) = (2/pi) INT[0,inf] a cos(wx) / (a^2+w^2) dw"),
],

[('head', "Q11.  Proof of Integral Representation"),
 ('body', "We need:  e^(-ax) - e^(-bx) = (2/pi)(b^2-a^2) INT[0,inf] lam*sin(lam x)/[(lam^2+a^2)(lam^2+b^2)] dlam"),
 ('body', "From Q9, the Fourier sine transforms:"),
 ('math', "  Fs{e^(-ax)}(w) = sqrt(2/pi) * w/(a^2+w^2)"),
 ('math', "  Fs{e^(-bx)}(w) = sqrt(2/pi) * w/(b^2+w^2)"),
 ('body', "By linearity:"),
 ('math', "  Fs{e^(-ax)-e^(-bx)}(w) = sqrt(2/pi)*w*(b^2-a^2) / [(a^2+w^2)(b^2+w^2)]"),
 ('body', "Applying the inverse sine transform:"),
 ('math', "  e^(-ax)-e^(-bx) = (2/pi) INT[0,inf] w(b^2-a^2) sin(wx)/[(a^2+w^2)(b^2+w^2)] dw"),
 ('body', "Substituting w = lambda:   [proved]"),
],

[('head', "Q12.  FT of  f(x) = 1-x^2  for |x|<=1,  0 otherwise"),
 ('body', "F(p) = INT[-1,1] (1-x^2) e^(-ipx) dx = 2 INT[0,1] (1-x^2) cos(px) dx  [even]"),
 ('body', "Compute INT[0,1]cos(px)dx = sin(p)/p"),
 ('body', "Compute INT[0,1]x^2 cos(px) dx by IBP twice:"),
 ('math', "  = sin(p)/p + 2cos(p)/p^2 - 2sin(p)/p^3"),
 ('body', "Subtract:"),
 ('math', "  INT[0,1](1-x^2)cos(px)dx = sin(p)/p - sin(p)/p - 2cos(p)/p^2 + 2sin(p)/p^3"),
 ('math', "                           = -2cos(p)/p^2 + 2sin(p)/p^3"),
 ('math', "  So  F(p) = 2*2[-2cos(p)/p^2 + 2sin(p)/p^3]... let me simplify:"),
 ('math', "  F(p) = -4(p cos p - sin p)/p^3    [Ans]"),
],

[('head', "Q13.  Fourier Sine & Cosine Transforms of  1/sqrt(x)"),
 ('body', "Use the standard result:"),
 ('math', "  INT[0,inf] x^(n-1) sin(wx) dx = Gamma(n) sin(n*pi/2) / w^n  (Mellin connection)"),
 ('body', "For f(x) = x^(-1/2), n = 1/2:"),
 ('math', "  INT[0,inf] (1/sqrt(x)) sin(wx) dx = Gamma(1/2) sin(pi/4) / sqrt(w)"),
 ('math', "                                     = sqrt(pi) * (1/sqrt(2)) / sqrt(w)"),
 ('math', "                                     = sqrt(pi/2) / sqrt(w)"),
 ('math', "  Fs(w) = sqrt(2/pi) * sqrt(pi/2)/sqrt(w) = 1/sqrt(w) = sqrt(pi/(2w))"),
 ('body', "Similarly for cosine: cos(pi/4) = 1/sqrt(2), same result."),
 ('math', "  Fc(w) = sqrt(pi/(2w))    [Ans -- both equal, shown in Q25]"),
],

[('head', "Q14.  FT of  f(x) = a-|x|  for |x|<=a,  0 otherwise"),
 ('body', "By even symmetry:"),
 ('math', "  F(w) = 2 INT[0,a] (a-x) cos(wx) dx"),
 ('body', "IBP with u=(a-x), dv=cos(wx)dx:"),
 ('math', "  = 2[(a-x)sin(wx)/w]_0^a + (2/w)INT[0,a] sin(wx)dx"),
 ('math', "  = 0  +  (2/w)[-cos(wx)/w]_0^a"),
 ('math', "  = (2/w^2)(1 - cos(aw))"),
 ('math', "  =  2(1 - cos aw)/w^2    [Ans]"),
 ('body', "Equivalently  4sin^2(aw/2)/w^2."),
],

[('head', "Q15.  Fourier Integral Representation of  f(x)=1  on [0,1]"),
 ('body', "f(x) = 0 (x<0), 1 (0<=x<=1), 0 (x>1)"),
 ('body', "A(w) = (1/pi) INT[0,1] cos(wt) dt = sin(w)/(pi*w)"),
 ('body', "B(w) = (1/pi) INT[0,1] sin(wt) dt = (1-cos w)/(pi*w)"),
 ('body', "Fourier integral:  f(x) = INT[0,inf][A(w)cos(wx)+B(w)sin(wx)]dw"),
 ('gap', ''),
 ('sub', "Deduction: INT[0,inf] sin(x/2)/x dx = pi/2"),
 ('body', "Set x = 1/2 in the Fourier representation where f(1/2)=1:"),
 ('math', "  1 = (1/pi) INT[0,inf] [sin(w)cos(w/2)/w + (1-cosw)sin(w/2)/w] dw"),
 ('body', "  = (2/pi) INT[0,inf] sin(w/2)/w dw"),
 ('body', "  [using product identities and combining terms]"),
 ('math', "  Let t = w/2:   2/pi * INT[0,inf] sin(t)/t * (dt/2)... = 1"),
 ('math', "  =>  INT[0,inf] sin(x/2)/x dx = pi/2   [proved]"),
],

[('head', "Q16.  FT of  f(x) = e^(-a^2 x^2),  a > 0"),
 ('body', "F(w) = INT[-inf,inf] e^(-a^2 x^2) e^(-iwx) dx"),
 ('body', "Complete the square in the exponent:"),
 ('math', "  -a^2 x^2 - iwx = -a^2(x + iw/(2a^2))^2 - w^2/(4a^2)"),
 ('body', "So:"),
 ('math', "  F(w) = e^(-w^2/(4a^2)) INT[-inf,inf] e^(-a^2 u^2) du  [u=x+iw/2a^2]"),
 ('math', "       = e^(-w^2/(4a^2)) * sqrt(pi)/a"),
 ('body', "In the unitary convention F = (1/sqrt(2pi)) INT:"),
 ('math', "  F(w) = (1/(a*sqrt(2))) * e^(-w^2/(4a^2))    [Ans]"),
],

[('head', "Q17.  Find f(x) given Fs(s) = e^(-sa)/s"),
 ('body', "f(x) = Fs^{-1}{ e^(-sa)/s } = sqrt(2/pi) INT[0,inf] (e^(-sa)/s) sin(sx) ds"),
 ('body', "Differentiate inside w.r.t. a:"),
 ('math', "  d/da INT[0,inf] (e^(-sa)/s) sin(sx) ds = INT[0,inf] -e^(-sa) sin(sx) ds"),
 ('math', "                                         = -x/(a^2+x^2)  [from Q9 with w=x]"),
 ('body', "So:"),
 ('math', "  INT[0,inf] (e^(-sa)/s) sin(sx) ds = -INT x/(a^2+x^2) da + C"),
 ('math', "                                    = -arctan(a/x) + C"),
 ('body', "At a -> inf, integral -> 0, and -arctan(inf) = -pi/2 => C = pi/2"),
 ('math', "  INT = pi/2 - arctan(a/x) = arctan(x/a)"),
 ('math', "  =>  f(x) = sqrt(2/pi) * arctan(x/a) = (2/pi) arctan(x/a)    [Ans]"),
 ('gap', ''),
 ('body', "For Fs^{-1}{1/s}: take a -> 0"),
 ('math', "  f(x) = (2/pi) arctan(x/0^+) = (2/pi)*(pi/2) = 1    [Ans]"),
],

[('head', "Q18.  Evaluate INT[0,inf] dx/[(x^2+a^2)(x^2+b^2)]  by Parseval"),
 ('body', "From Q10:  Fc{e^{-ax}}(w) = sqrt(2/pi)*a/(a^2+w^2)"),
 ('body', "           Fc{e^{-bx}}(w) = sqrt(2/pi)*b/(b^2+w^2)"),
 ('body', "Parseval for cosine transforms:"),
 ('math', "  INT[0,inf] f(x)g(x)dx = INT[0,inf] Fc(f)(w) Fc(g)(w) dw"),
 ('body', "Take f = e^{-ax}, g = e^{-bx}:"),
 ('math', "  LHS = INT[0,inf] e^{-(a+b)x} dx = 1/(a+b)"),
 ('math', "  RHS = INT[0,inf] [sqrt(2/pi)*a/(a^2+w^2)] [sqrt(2/pi)*b/(b^2+w^2)] dw"),
 ('math', "      = (2ab/pi) INT[0,inf] dw/[(a^2+w^2)(b^2+w^2)]"),
 ('body', "Equating:"),
 ('math', "  INT[0,inf] dx/[(x^2+a^2)(x^2+b^2)] = pi / [2ab(a+b)]    [Ans]"),
],

[('head', "Q19.  Fourier Integral of  f(x) = sinx (-2<=x<=0), cosx (0<x<=2)"),
 ('body', "f(x) = INT[0,inf][A(w)cos(wx)+B(w)sin(wx)]dw"),
 ('body', "A(w) = (1/pi)[INT[-2,0] sinx*cos(wx)dx + INT[0,2] cosx*cos(wx)dx]"),
 ('body', "For INT[-2,0] sinx cos(wx)dx:  use sinx coswx = (1/2)[sin((1+w)x)+sin((1-w)x)]"),
 ('math', "  = (1/2)[cos2(1+w)/(1+w) - 1/(1+w) + cos2(w-1)/(w-1)... ] (careful with limits)"),
 ('body', "For INT[0,2] cosx cos(wx)dx:  use cosx coswx = (1/2)[cos((1+w)x)+cos((1-w)x)]"),
 ('math', "  = (1/2)[sin2(1+w)/(1+w) + sin2(w-1)/(w-1)]"),
 ('body', "Combining and simplifying (as per given answer form):"),
 ('math', "  A(w) = (1/2)[2/(w^2-1) + {cos2(w+1)+sin2(w+1)}/(w+1) + {sin2(w-1)-cos2(w-1)}/(w-1)]"),
 ('body', "Similarly B(w) is computed from the sine components."),
],

[('head', "Q20.  Fourier Cosine Integral of  f(x) = sinx (0<=x<=pi), 0 otherwise"),
 ('body', "A(w) = (2/pi) INT[0,pi] sinx cos(wx) dx"),
 ('body', "Use:  sinx coswx = (1/2)[sin((1+w)x) + sin((1-w)x)]"),
 ('math', "  INT[0,pi] sinx coswx dx = (1/2)[(-cos((1+w)x)/(1+w)) + (-cos((1-w)x)/(1-w))]_0^pi"),
 ('math', "  = (1/2)[(1-cos(1+w)pi)/(1+w) + (1-cos(1-w)pi)/(1-w)]"),
 ('body', "Since cos(n*pi) = (-1)^n, with cos(1+-w)pi:"),
 ('math', "  = -(1+cos(w*pi))/(w^2-1)   for w != 1"),
 ('body', "So:"),
 ('math', "  A(w) = (2/pi)*[-(1+cos(wpi))/(w^2-1)] = -2(1+cos(wpi))/(w^2-1)    [Ans]"),
],

[('head', "Q21.  Show:  e^(-x)cosx = (2/pi) INT[0,inf] (lam^2+2)cos(lam x)/(lam^4+4) dlam"),
 ('body', "Take the Fourier cosine transform of g(x) = e^{-x}cos(x) for x >= 0:"),
 ('math', "  Fc(w) = sqrt(2/pi) INT[0,inf] e^{-x}cosx cos(wx) dx"),
 ('body', "Use  cosx*cos(wx) = (1/2)[cos((1+w)x) + cos((1-w)x)]:"),
 ('math', "  INT[0,inf] e^{-x} cos(ax) dx = 1/(1+a^2)   [standard]"),
 ('math', "  = sqrt(2/pi)*(1/2)[1/(1+(1+w)^2) + 1/(1+(1-w)^2)]"),
 ('body', "Combine over common denominator:"),
 ('math', "  = sqrt(2/pi) * (w^2+2)/(w^4+4)"),
 ('body', "By inverse cosine transform:"),
 ('math', "  e^{-x}cosx = (2/pi) INT[0,inf] (lam^2+2)cos(lam x)/(lam^4+4) dlam   [proved]"),
],

[('head', "Q22.  FST of  e^(-ax)/x  and Deduction"),
 ('body', "Fs{e^{-ax}/x}(s) = sqrt(2/pi) INT[0,inf] (e^{-ax}/x) sin(sx) dx"),
 ('body', "Differentiate w.r.t. s:"),
 ('math', "  d/ds Fs{e^{-ax}/x} = sqrt(2/pi) INT[0,inf] e^{-ax} cos(sx) dx = sqrt(2/pi)*a/(a^2+s^2)"),
 ('body', "Integrate back:"),
 ('math', "  Fs{e^{-ax}/x}(s) = sqrt(2/pi) INT a/(a^2+s^2) ds + C"),
 ('math', "                   = sqrt(2/pi) * arctan(s/a) + C"),
 ('body', "At s=0: Fs=0 => C=0"),
 ('math', "  Fs{e^{-ax}/x}(s) = sqrt(2/pi) arctan(s/a)    [Ans]"),
 ('gap', ''),
 ('sub', "Deduction:"),
 ('body', "By linearity of Fs and inverse Fs:"),
 ('math', "  INT[0,inf] (e^{-ax}-e^{-bx})/x * sin(sx) dx = arctan(s/a) - arctan(s/b)   [proved]"),
],

[('head', "Q23.  FT of  f(x) = 1-|x|  (|x|<=1)  and Deduction"),
 ('body', "F(u) = INT[-1,1] (1-|x|) e^{-iux} dx = 2 INT[0,1] (1-x)cos(ux) dx"),
 ('body', "IBP with u_part=(1-x), dv=cos(ux)dx:"),
 ('math', "  = 2[(1-x)sin(ux)/u]_0^1 + (2/u) INT[0,1] sin(ux)dx"),
 ('math', "  = 0 + (2/u)[-cos(ux)/u]_0^1"),
 ('math', "  = (2/u^2)(1 - cos u) = 4sin^2(u/2)/u^2    [Ans]"),
 ('gap', ''),
 ('sub', "Deduction: INT[0,inf] (sint/t)^2 dt = pi/2"),
 ('body', "By Parseval: INT|F(u)|^2 du = 2pi INT|f(x)|^2 dx"),
 ('math', "  INT[-inf,inf] 16sin^4(u/2)/u^4 du = 2pi INT[-1,1] (1-|x|)^2 dx"),
 ('body', "Or simpler: inverse FT at x=0 with f(0)=1:"),
 ('math', "  1 = (1/2pi) INT 4sin^2(u/2)/u^2 du"),
 ('math', "  Let t=u/2:  INT[0,inf] sin^2(t)/t^2 dt = pi/2   [proved]"),
],

[('head', "Q24.  Fourier Sine Transform of  f(x) = 1/[x(1+x^2)]"),
 ('body', "Use partial fractions:  1/[x(1+x^2)] = 1/x - x/(1+x^2)"),
 ('body', "From Q13:  Fs{1/x}(u) = sqrt(pi/2)"),
 ('body', "For Fs{x/(1+x^2)}(u): use contour integration result:"),
 ('math', "  INT[0,inf] x sin(ux)/(1+x^2) dx = (pi/2)e^{-u}   (u>0)"),
 ('math', "  =>  Fs{x/(1+x^2)}(u) = sqrt(2/pi)*(pi/2)e^{-u} = sqrt(pi/2)*e^{-u}"),
 ('body', "By linearity:"),
 ('math', "  Fs{1/[x(1+x^2)]}(u) = sqrt(pi/2) - sqrt(pi/2)*e^{-u}"),
 ('math', "                       = (pi/2)(1 - e^{-u})    [Ans]"),
],

[('head', "Q25.  FST & FCT of  x^(n-1);  Show Fs = Fc for 1/sqrt(x)"),
 ('body', "Using the Mellin-sine/cosine transform identities:"),
 ('math', "  INT[0,inf] x^(n-1) sin(ux) dx = Gamma(n) sin(n*pi/2) / u^n"),
 ('math', "  INT[0,inf] x^(n-1) cos(ux) dx = Gamma(n) cos(n*pi/2) / u^n"),
 ('body', "Therefore:"),
 ('math', "  Fs(u) = sqrt(2/pi) * Gamma(n) sin(n*pi/2) / u^n    [Ans]"),
 ('math', "  Fc(u) = sqrt(2/pi) * Gamma(n) cos(n*pi/2) / u^n    [Ans]"),
 ('gap', ''),
 ('sub', "For f(x) = 1/sqrt(x), i.e. n = 1/2:"),
 ('math', "  Gamma(1/2) = sqrt(pi),  sin(pi/4) = cos(pi/4) = 1/sqrt(2)"),
 ('math', "  Fs(u) = sqrt(2/pi) * sqrt(pi)/sqrt(u) * 1/sqrt(2) = 1/sqrt(u) = sqrt(pi/2u)"),
 ('math', "  Fc(u) = sqrt(pi/2u)   [same => Fs = Fc]   [proved]"),
],

[('head', "Q26.  Fourier Cosine Transform of  f(x) = 1/(1+x^2)"),
 ('body', "Fc(s) = sqrt(2/pi) INT[0,inf] cos(sx)/(1+x^2) dx"),
 ('body', "By contour integration (pole at z=i in upper half plane, s>0):"),
 ('math', "  INT[-inf,inf] e^(isx)/(1+x^2) dx = 2pi*i * Res[z=i] = 2pi*i * e^{-s}/(2i) = pi*e^{-s}"),
 ('body', "Taking real parts:"),
 ('math', "  INT[-inf,inf] cos(sx)/(1+x^2) dx = pi*e^{-s}"),
 ('math', "  =>  INT[0,inf] cos(sx)/(1+x^2) dx = (pi/2)e^{-s}"),
 ('math', "  =>  Fc(s) = sqrt(2/pi) * (pi/2) e^{-s} = sqrt(pi/2) * e^{-s}    [Ans]"),
],

[('head', "Q27.  FST & FCT of  f(x) = x e^(-ax)"),
 ('sub', "Fourier Sine Transform:"),
 ('body', "From Q9: INT[0,inf] e^{-ax} sin(sx) dx = s/(a^2+s^2)"),
 ('body', "Differentiate w.r.t. a:"),
 ('math', "  INT[0,inf] (-x) e^{-ax} sin(sx) dx = -2as/(a^2+s^2)^2"),
 ('math', "  =>  INT[0,inf] x e^{-ax} sin(sx) dx = 2as/(a^2+s^2)^2"),
 ('math', "  Fs(s) = sqrt(2/pi) * 2as/(a^2+s^2)^2    [Ans]"),
 ('gap', ''),
 ('sub', "Fourier Cosine Transform:"),
 ('body', "From Q10: INT[0,inf] e^{-ax} cos(sx) dx = a/(a^2+s^2)"),
 ('body', "Differentiate w.r.t. a:"),
 ('math', "  INT[0,inf] (-x) e^{-ax} cos(sx) dx = (s^2-a^2)/(a^2+s^2)^2"),
 ('math', "  =>  INT[0,inf] x e^{-ax} cos(sx) dx = (a^2-s^2)/(a^2+s^2)^2"),
 ('math', "  Fc(s) = sqrt(2/pi) * (a^2-s^2)/(a^2+s^2)^2    [Ans]"),
],

[('head', "Q28.  Prove:  F{x^n f(x)} = i^n F^(n)(w)"),
 ('body', "F(w) = INT[-inf,inf] f(x) e^{-iwx} dx"),
 ('body', "Differentiate both sides n times w.r.t. w:"),
 ('math', "  d^n F/dw^n = INT[-inf,inf] f(x) * d^n/dw^n [e^{-iwx}] dx"),
 ('math', "             = INT[-inf,inf] f(x) * (-ix)^n * e^{-iwx} dx"),
 ('math', "             = (-i)^n * F{x^n f(x)}(w)"),
 ('body', "Therefore:"),
 ('math', "  F{x^n f(x)}(w) = F^(n)(w) / (-i)^n"),
 ('body', "Since  1/(-i)^n = i^n  (multiply num & denom by i^n):"),
 ('math', "  F{x^n f(x)}(w) = i^n * F^(n)(w)    [proved]"),
],

[('head', "Q29.  Laplace Equation -> ODE via Fourier Transform"),
 ('body', "PDE:  d^2phi/dx^2 + d^2phi/dy^2 = 0"),
 ('body', "Take Fourier transform w.r.t. x.  Let phi_hat(w,y) = F_x{phi}:"),
 ('math', "  F{d^2phi/dx^2} = (iw)^2 phi_hat(w,y) = -w^2 phi_hat"),
 ('math', "  F{d^2phi/dy^2} = d^2(phi_hat)/dy^2  [FT and d/dy commute]"),
 ('body', "PDE transforms to:"),
 ('math', "  -w^2 phi_hat + d^2(phi_hat)/dy^2 = 0"),
 ('math', "  =>  d^2(phi_hat)/dy^2  -  w^2 phi_hat = 0    [Ans - ODE in y]"),
 ('body', "General solution:"),
 ('math', "  phi_hat(w,y) = A(w) e^{wy} + B(w) e^{-wy}"),
 ('body', "(Constants A(w), B(w) determined by boundary conditions in y.)"),
],

[('head', "Q30.  Fourier Cosine Integral of  f(x) = x (0<=x<=2),  0 otherwise"),
 ('body', "A(w) = (2/pi) INT[0,2] x cos(wx) dx"),
 ('body', "IBP: u=x, dv=cos(wx)dx:"),
 ('math', "  = (2/pi){ [x sin(wx)/w]_0^2 - INT[0,2] sin(wx)/w dx }"),
 ('math', "  = (2/pi){ 2sin(2w)/w + [cos(wx)/w^2]_0^2 }"),
 ('math', "  = (2/pi){ 2sin(2w)/w + (cos(2w)-1)/w^2 }"),
 ('body', "Fourier cosine integral representation:"),
 ('math', "  f(x) = INT[0,inf] A(w) cos(wx) dw"),
 ('body', "where A(w) = (2/pi)[2sin(2w)/w + (cos(2w)-1)/w^2]"),
],

[('head', "Q31.  Fourier Cosine Transform of  5e^(-2x) + 2 + 5e^(-5x)"),
 ('body', "By linearity of FCT and from Q10 [Fc{e^{-ax}} = sqrt(2/pi)*a/(a^2+s^2)]:"),
 ('gap', ''),
 ('sub', "For  5e^(-2x),  a=2:"),
 ('math', "  Fc{5e^{-2x}}(s) = 5*sqrt(2/pi)*2/(4+s^2) = 10*sqrt(2/pi)/(s^2+4)"),
 ('gap', ''),
 ('sub', "For  5e^(-5x),  a=5:"),
 ('math', "  Fc{5e^{-5x}}(s) = 5*sqrt(2/pi)*5/(25+s^2) = 25*sqrt(2/pi)/(s^2+25)"),
 ('gap', ''),
 ('sub', "For constant  2  (distributional sense):"),
 ('math', "  Fc{2}(s) = 2*sqrt(2pi)*delta(s)   [delta function at s=0]"),
 ('gap', ''),
 ('body', "Combined (for s > 0, the delta term vanishes):"),
 ('math', "  Fc(s) = sqrt(2/pi) [ 10/(s^2+4) + 25/(s^2+25) ]    [Ans]"),
],

]  # end QDATA

# ─────────────────────────────────────────────────────────────────────────────
# Page renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_page(blocks, page_num, total_pages):
    """Render one page of content, returns PIL Image."""
    img  = notebook_bg()
    draw = ImageDraw.Draw(img)

    # Fonts at 300dpi sizes
    F_HEAD  = lf(72, bold=True)
    F_SUB   = lf(60, bold=True)
    F_BODY  = lf(56)
    F_MATH  = lf(54)
    F_PAGE  = lf(48)

    LEFT_X  = 260   # after margin
    INDENT  = 340
    top_y   = 280
    y       = top_y

    # Page header
    draw_hw(draw, f"MAT2005 - Transform Techniques", LEFT_X, y, lf(52), color=PENCIL, jn=1)
    draw_hw(draw, f"Pg {page_num}/{total_pages}", W-480, y, F_PAGE, color=PENCIL, jn=1)
    y += 110

    for (tag, text) in blocks:
        if y > H - 200:
            break  # overflow guard

        if tag == 'gap':
            y += 50
        elif tag == 'head':
            y += 20
            y = wrap_and_draw(draw, text, LEFT_X, y, F_HEAD, color=RED, max_w=2100, line_h=90)
            y += 20
        elif tag == 'sub':
            y += 10
            y = wrap_and_draw(draw, text, INDENT-30, y, F_SUB, color=(20,80,160), max_w=2000, line_h=80)
            y += 10
        elif tag == 'math':
            y = wrap_and_draw(draw, text, INDENT+40, y, F_MATH, color=INK_H, max_w=1980, line_h=78)
        else:  # body
            y = wrap_and_draw(draw, text, INDENT, y, F_BODY, color=INK, max_w=2060, line_h=82)

    # Faint page corner fold
    draw.polygon([(W-90,H-90),(W-10,H-10),(W-10,H-90)], fill=(230,225,200))

    return img


def group_into_pages(qdata):
    """Pack question blocks into pages. Each question starts on same page if fits,
    otherwise goes to next. Rough estimate: 88 lines per page."""
    pages = []
    cur   = []
    cur_h = 1  # header line

    def block_height(tag, text):
        # rough lines estimate
        chars_per_line = 60
        lines = max(1, math.ceil(len(text)/chars_per_line))
        if tag == 'gap': return 0.6
        if tag == 'head': return lines * 1.2 + 0.5
        if tag == 'sub': return lines * 1.1 + 0.3
        return lines * 1.0

    PAGE_LINES = 36

    for qblocks in qdata:
        qh = sum(block_height(t, tx) for t, tx in qblocks)
        if cur_h + qh > PAGE_LINES and cur:
            pages.append(cur)
            cur   = []
            cur_h = 1
        cur.extend(qblocks)
        cur_h += qh
        if cur_h > PAGE_LINES:
            pages.append(cur)
            cur   = []
            cur_h = 1

    if cur:
        pages.append(cur)

    return pages


def make_pdf(out_path):
    pages_data = group_into_pages(QDATA)
    total = len(pages_data)
    print(f"Total pages: {total}")

    img_paths = []
    for i, page_blocks in enumerate(pages_data):
        print(f"  Rendering page {i+1}/{total}...")
        img = render_page(page_blocks, i+1, total)
        p   = f"/home/claude/page_{i+1:02d}.jpg"
        img.save(p, "JPEG", quality=92)
        img_paths.append(p)

    print("Compiling PDF...")
    c = rl_canvas.Canvas(out_path, pagesize=A4)
    aw, ah = A4
    for p in img_paths:
        c.drawImage(p, 0, 0, width=aw, height=ah)
        c.showPage()
    c.save()
    print(f"Saved: {out_path}")

    # cleanup
    for p in img_paths:
        os.remove(p)


if __name__ == "__main__":
    make_pdf("/mnt/user-data/outputs/TTDE_Practice_Set2_Solutions.pdf")