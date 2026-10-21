# -*- coding: utf-8 -*-
"""Fig. 1: (a) HVI-CIDNet with IG-LCA at LCA3-LCA6, in the visual style of our earlier framework figure (orange
feature stages, LCA modules between the two branches, purple skip connections); (b) the IG-LCA block with the gate
drawn as its four layers (1x1 conv, PReLU, 1x1 conv, 2 sigmoid).

Connections follow the forward pass of net/CIDNet_IGLCA_ABLATION_3456.py (official CIDNet code):
  - the I branch reads the I channel of HVI, the HV branch the whole HVI map (concatenation of HV and I);
  - stages: Conv, then [Down + LCA1], [Down + LCA2], [Down + LCA3], [LCA4], [Up + LCA5], [Up + LCA6], then Up + Conv;
  - skip connections: Conv -> last Up, LCA1 -> the Up before LCA6, LCA2 -> the Up before LCA5;
  - the LCA2 outputs reach the decoder only through the skip connection (marked with a dagger, see the caption);
  - the I-branch LCA5 output is overwritten in the forward pass (dashed arrow ending in a cross);
  - output = PHVIT(concat(HV, I) + input HVI).
Images: figures_src/fig1_assets/{input,output,imap,hvmap}.jpg (an LOL input image, its HVI maps and an enhanced output).
Usage: python analysis/make_fig1_drawio.py <out.drawio>
Export: draw.io --export --format pdf --crop --output IG_LCA_overall_v4.pdf <out.drawio>
"""
import base64
import html
import os
import sys

FONT = "Times New Roman"
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures_src", "fig1_assets")
cells = []
_id = [10]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def esc(t):
    return html.escape(t)


def add(value, style, x, y, w, h):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{esc(value)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def box(x, y, w, h, t="", fill="#ffffff", stroke="#333333", fs=17, bold=False, arc=20, dashed=False, color="#000000",
        sw=1.4):
    st = (f"rounded=1;whiteSpace=wrap;html=1;fontFamily={FONT};fontSize={fs};fillColor={fill};strokeColor={stroke};"
          f"strokeWidth={sw};arcSize={arc};fontColor={color};spacing=0;" + ("fontStyle=1;" if bold else "")
          + ("dashed=1;dashPattern=6 4;" if dashed else ""))
    return add(t, st, x, y, w, h)


def text(x, y, w, h, t, fs=16, color="#000000", align="center", bold=False, italic=False):
    style = (1 if bold else 0) + (2 if italic else 0)
    st = (f"text;html=1;fontFamily={FONT};fontSize={fs};fontColor={color};align={align};verticalAlign=middle;"
          f"fontStyle={style};spacing=0;")
    return add(t, st, x, y, w, h)


def circle(cx, cy, d, t, fs=18):
    st = (f"ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontFamily={FONT};fontSize={fs};fillColor=#ffffff;"
          f"strokeColor=#000000;strokeWidth=1.4;spacing=0;spacingBottom=2;")
    return add(t, st, cx - d / 2, cy - d / 2, d, d)


def dot(cx, cy, d=6):
    return add("", "ellipse;html=1;aspect=fixed;fillColor=#000000;strokeColor=none;", cx - d / 2, cy - d / 2, d, d)


def image(x, y, w, h, name):
    data = base64.b64encode(open(os.path.join(ASSETS, name), "rb").read()).decode()
    st = (f"shape=image;html=1;imageAspect=0;aspect=fixed;image=data:image/jpeg,{data};"
          f"imageBorder=#666666;strokeWidth=1;")
    return add("", st, x, y, w, h)


def trapezoid(x, y, w, h):
    st = ("shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;size=10;direction=south;"
          "fillColor=#f5f5f5;strokeColor=#666666;strokeWidth=1.4;")
    return add("", st, x, y, w, h)


def line(pts, color="#000000", w=1.4, end="block", dashed=False, jump=False):
    i = nid()
    st = (f"edgeStyle=none;html=1;rounded=0;endArrow={end};endFill=1;endSize=5;strokeColor={color};strokeWidth={w};"
          + ("dashed=1;dashPattern=5 3;" if dashed else "") + ("jumpStyle=arc;jumpSize=8;" if jump else ""))
    (x1, y1), (x2, y2), mid = pts[0], pts[-1], pts[1:-1]
    geo = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in mid)
    geo = f'<Array as="points">{geo}</Array>' if mid else ""
    cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/>{geo}'
                 f'</mxGeometry></mxCell>')
    return i


ORANGE, ORANGE_S = "#ffe6cc", "#d79b00"
GREEN, GREEN_S = "#d5e8d4", "#82b366"
GRAY, GRAY_S = "#f5f5f5", "#666666"
PURPLE = "#6600CC"
BLUE, BLUE_S = "#dae8fc", "#6c8ebf"

# ======================= (a) overall architecture =======================
text(0, 0, 520, 24, "(a) HVI-CIDNet with IG-LCA at LCA3–LCA6", fs=19, bold=True, align="left")
YI, YH = 108, 232
YM = (YI + YH) / 2
# background panels (as in the earlier framework figure)
box(74, 66, 142, 208, fill="#e5f1ff", stroke="#9999FF", dashed=True, arc=8, sw=1.2)
box(282, 36, 508, 270, fill="#FFFAEE", stroke="#999900", dashed=True, arc=5, sw=1.2)
# input, HVIT, intensity / HV maps
image(0, YM - 22, 66, 44, "input.jpg")
text(0, YM + 24, 66, 18, "Input", fs=16)
hv = box(84, YM - 15, 46, 30, "HVIT", fill="#cce5ff", stroke="#9999FF", fs=16)
line([(66, YM), (84, YM)])
image(166, YI - 16, 40, 31, "imap.jpg")
image(166, YH - 16, 40, 31, "hvmap.jpg")
text(166, YI - 34, 40, 16, "I", fs=16, italic=True)
text(166, YH + 17, 40, 16, "HV", fs=16)
line([(130, YM), (144, YM), (144, YI), (166, YI)])
line([(130, YM), (144, YM), (144, YH), (166, YH)])
# stages: heights encode resolution; x positions
H = [64, 54, 44, 34, 34, 44, 54, 64]
X0, PITCH, W = 298, 66, 18
xs = [X0 + PITCH * k for k in range(8)]
cx = [x + W / 2 for x in xs]
for yc in (YI, YH):
    for k in range(8):
        box(xs[k], yc - H[k] / 2, W, H[k], fill=ORANGE, stroke=ORANGE_S, arc=30)
    for k in range(7):
        line([(xs[k] + W, yc), (xs[k + 1], yc)])
# branch inputs: I map -> I branch; concat(HV map, I map) -> HV branch
cc = circle(234, YH, 18, "C", fs=13)
line([(206, YI), (xs[0], YI)])
line([(206, YH), (225, YH)])
line([(234, YI), (234, YH - 9)])
dot(234, YI, 5)
line([(243, YH), (xs[0], YH)])
text(246, YH - 19, 32, 16, "HVI", fs=15)
# LCA modules between the branches
labels = ["LCA1", "LCA2", "IG-<br>LCA3", "IG-<br>LCA4", "IG-<br>LCA5", "IG-<br>LCA6"]
PW, PH = 54, 36
for k in range(1, 7):
    ig = k >= 3
    box(cx[k] - PW / 2, YM - PH / 2, PW, PH, labels[k - 1], fill=GREEN if ig else GRAY, stroke=GREEN_S if ig else GRAY_S,
        fs=15, bold=True, arc=25, color="#000000" if ig else "#333333")
    top_i = YI + H[k] / 2
    if k == 2:
        # LCA2 reads the block-2 features of both branches, but its outputs go only to the skip connections
        # (the next encoder stage reads the block-2 output): arrows point INTO LCA2, the skips start at LCA2
        line([(cx[k], top_i), (cx[k], YM - PH / 2)])
        line([(cx[k], YH - H[k] / 2), (cx[k], YM + PH / 2)])
    elif k == 5:
        # the I-branch LCA5 output is overwritten in the forward pass: LCA5 only reads the I-branch feature
        # (arrow INTO LCA5) and only updates the HV branch
        line([(cx[k], top_i), (cx[k], YM - PH / 2)])
        line([(cx[k], YM + PH / 2), (cx[k], YH - H[k] / 2)])
    else:
        line([(cx[k], YM - PH / 2), (cx[k], top_i)])
        line([(cx[k], YM + PH / 2), (cx[k], YH - H[k] / 2)])
# skip connections (purple): Conv -> last Up, LCA1 -> Up before LCA6, LCA2 output -> Up before LCA5
for (a, b, lev) in ((0, 7, 48), (1, 6, 59)):
    line([(cx[a], YI - H[a] / 2), (cx[a], lev), (cx[b] - 4, lev), (cx[b] - 4, YI - H[b] / 2)], color=PURPLE, w=1.3)
    lv = YH + (YI - lev)  # mirrored below the HV row
    line([(cx[a], YH + H[a] / 2), (cx[a], lv), (cx[b] - 4, lv), (cx[b] - 4, YH + H[b] / 2)], color=PURPLE, w=1.3)
xs2 = cx[2] + 19  # the LCA2 skips leave the side of the LCA2 box, beside the block-2 features
lev = 70
line([(xs2, YM - PH / 2), (xs2, lev), (cx[5] - 4, lev), (cx[5] - 4, YI - H[5] / 2)], color=PURPLE, w=1.3, jump=True)
lv = YH + (YI - lev)
line([(xs2, YM + PH / 2), (xs2, lv), (cx[5] - 4, lv), (cx[5] - 4, YH + H[5] / 2)], color=PURPLE, w=1.3, jump=True)
# output head: concat(HV, I) + input HVI -> PHVIT -> output
xe = xs[7] + W
cat = circle(xe + 36, YM, 20, "C", fs=13)
line([(xe, YI), (xe + 36, YI), (xe + 36, YM - 10)])
line([(xe, YH), (xe + 36, YH), (xe + 36, YM + 10)])
pl = circle(xe + 70, YM, 20, "+", fs=17)
line([(xe + 46, YM), (xe + 60, YM)])
text(xe + 52, YM + 24, 36, 16, "HVI", fs=15)
line([(xe + 70, YM + 24), (xe + 70, YM + 10)])
ph = box(xe + 94, YM - 15, 58, 30, "PHVIT", fill="#fff2cc", stroke="#d6b656", fs=15)
line([(xe + 80, YM), (xe + 94, YM)])
image(xe + 166, YM - 22, 66, 44, "output.jpg")
line([(xe + 152, YM), (xe + 166, YM)])
text(xe + 166, YM + 24, 66, 18, "Output", fs=16)
XR = xe + 232

# ======================= (b) IG-LCA block =======================
ox = XR + 40
text(ox, 0, 300, 24, "(b) IG-LCA block", fs=19, bold=True, align="left")
YF, YP = 70, 146                      # rows of F and F'
f = box(ox, YF - 16, 40, 32, "<i>F</i>", fill=GREEN, stroke=GREEN_S, fs=19)
fp = box(ox, YP - 16, 40, 32, "<i>F′</i>", fill="#e1d5e7", stroke="#9673a6", fs=19)
box(ox + 54, YF - 15, 40, 30, "LN", fill=BLUE, stroke=BLUE_S, fs=16)
box(ox + 54, YP - 15, 40, 30, "LN", fill=BLUE, stroke=BLUE_S, fs=16)
line([(ox + 40, YF), (ox + 54, YF)])
line([(ox + 40, YP), (ox + 54, YP)])
cab = box(ox + 134, YF - 26, 70, (YP - YF) + 52, "CAB", fill=BLUE, stroke=BLUE_S, fs=19, arc=12)
text(ox + 137, YF - 9, 20, 18, "Q", fs=14, align="left")
text(ox + 137, YP - 9, 40, 18, "K,V", fs=14, align="left")
d1x, d2x = ox + 106, ox + 118
line([(ox + 94, YF), (ox + 134, YF)])
line([(ox + 94, YP), (ox + 134, YP)])
dot(d1x, YF)
dot(d2x, YP)
YC = (YF + YP) / 2
mul = circle(ox + 250, YC, 24, "×", fs=18)
add_ = circle(ox + 296, YC, 24, "+", fs=18)
line([(ox + 204, YC), (ox + 238, YC)])
text(ox + 206, YC - 20, 30, 18, "Δ<i>F</i>", fs=16)
line([(ox + 262, YC), (ox + 284, YC)])
box(ox + 336, YC - 15, 36, 30, "LN", fill=BLUE, stroke=BLUE_S, fs=16)
line([(ox + 308, YC), (ox + 336, YC)])
box(ox + 386, YC - 16, 44, 32, "IEL", fill=GREEN, stroke=GREEN_S, fs=17)
line([(ox + 372, YC), (ox + 386, YC)])
# the HV-branch block does not (x = IEL(LN(x))) -> dashed bypass into a second sum
add2 = circle(ox + 460, YC, 24, "+", fs=18)
line([(ox + 430, YC), (ox + 448, YC)])
dot(ox + 316, YC, 5)
line([(ox + 316, YC), (ox + 316, YC - 34), (ox + 460, YC - 34), (ox + 460, YC - 12)], dashed=True)
text(ox + 336, YC - 52, 112, 16, "I update only", fs=14, color="#444444")
line([(ox + 472, YC), (ox + 496, YC)])
text(ox + 476, YC - 20, 30, 16, "out", fs=15)
line([(ox + 20, YF - 16), (ox + 20, 32), (ox + 296, 32), (ox + 296, YC - 12)])  # residual
# gate input: LN(F') for the HV update, LN(F) for the I update
YG = 238
MX = ox + 196                         # selector (trapezoid) left edge
mux = trapezoid(MX, YG - 24, 24, 48)
line([(d2x, YP), (d2x, YG - 12), (MX, YG - 12)])
line([(d1x, YF), (d1x, YG + 12), (MX, YG + 12)], jump=True)
text(d2x + 4, YG - 29, MX - d2x - 8, 16, "HV update", fs=14)
text(d1x + 4, YG + 14, MX - d1x - 8, 16, "I update", fs=14)
# gate: two 1x1 convolutions with a PReLU, then 2 sigmoid (dashed frame = g)
GX = ox + 290                         # left edge of the dashed frame
box(GX, YG - 31, 272, 62, fill="#fffaf2", stroke=ORANGE_S, dashed=True, arc=10, sw=1.0)
c1 = box(GX + 12, YG - 19, 64, 38, "1×1 conv<br><i>C</i>→<i>C</i>/4", fill=ORANGE, stroke=ORANGE_S, fs=14)
pr = box(GX + 90, YG - 13, 50, 26, "PReLU", fill=ORANGE, stroke=ORANGE_S, fs=14)
c2 = box(GX + 154, YG - 19, 64, 38, "1×1 conv<br><i>C</i>/4→<i>C</i>", fill=ORANGE, stroke=ORANGE_S, fs=14)
sg = box(GX + 232, YG - 13, 30, 26, "2σ", fill=ORANGE, stroke=ORANGE_S, fs=15)
line([(MX + 24, YG), (GX + 12, YG)])
text(MX + 26, YG - 22, GX - MX - 28, 18, "LN(<i>F</i><sub>I</sub>)", fs=15)
line([(GX + 76, YG), (GX + 90, YG)])
line([(GX + 140, YG), (GX + 154, YG)])
line([(GX + 218, YG), (GX + 232, YG)])
XG = GX + 272 + 22                    # the G line runs outside the frame
line([(GX + 262, YG), (XG, YG), (XG, YC + 50), (ox + 250, YC + 50), (ox + 250, YC + 12)])
text(ox + 254, YC + 28, 20, 18, "<i>G</i>", fs=16, align="left")
text(GX, YG + 34, 272, 16, "<i>g</i>: last conv zero-initialized (<i>G</i> = 1 at start)", fs=14, color="#444444")

xml = ('<mxfile host="drawio"><diagram id="iglca-overall-v4" name="Fig1 v4">'
       '<mxGraphModel dx="1500" dy="320" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
       'fold="1" page="0" pageScale="1" pageWidth="1500" pageHeight="320" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
open(sys.argv[1], "w", encoding="utf-8").write(xml)
print("wrote", sys.argv[1], len(cells), "cells")
