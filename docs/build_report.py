"""
build_report.py -- assemble docs/edge_coupler_report.pdf from the text below and the
numbers / figures in results/.  Run after fdfd_taper_study.py and fdfd_coupler_design.py:

    python docs/build_report.py

Display equations are typeset with matplotlib's mathtext and embedded as images
(cached in docs/.eq/); inline sub/superscripts use explicit size/rise attributes so
that they do not collide with the next line.  The narrative reuses the ECE 434 project
report (introduction, mathematical framework, references) and adds the corrected
formulation, the validation, the optimisation results and the plausibility review.
"""

import os
import re
import json
import hashlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, KeepTogether)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, 'results')
EQ_DIR = os.path.join(HERE, '.eq')
OUT = os.path.join(HERE, 'edge_coupler_report.pdf')
os.makedirs(EQ_DIR, exist_ok=True)
um, nm, deg = 1e-6, 1e-9, np.pi / 180
TEXT_W = 17 * cm

# --- fonts with Greek / µ / arrows (DejaVu ships with matplotlib)
FDIR = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DejaVu', os.path.join(FDIR, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', os.path.join(FDIR, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Oblique', os.path.join(FDIR, 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Mono', os.path.join(FDIR, 'DejaVuSansMono.ttf')))

ss = getSampleStyleSheet()
BODY = ParagraphStyle('body', parent=ss['Normal'], fontName='DejaVu', fontSize=9.5, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8, leading=11, textColor=colors.HexColor('#444444'))
CELL = ParagraphStyle('cell', parent=BODY, fontSize=8.5, leading=10.5, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle('cellb', parent=CELL, fontName='DejaVu-Bold')
H1 = ParagraphStyle('h1', parent=ss['Heading1'], fontName='DejaVu-Bold', fontSize=15, leading=19, spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle('h2', parent=ss['Heading2'], fontName='DejaVu-Bold', fontSize=11.5, leading=15, spaceBefore=9, spaceAfter=4)
TITLE = ParagraphStyle('title', parent=ss['Title'], fontName='DejaVu-Bold', fontSize=19, leading=24, spaceAfter=14)
CAP = ParagraphStyle('cap', parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor('#333333'), spaceBefore=2, spaceAfter=10)


def T(text):
    """Inline markup: give sub/superscripts an explicit size and rise so they stay
    inside the line box (reportlab's defaults let them collide with the next line)."""
    text = re.sub(r'<sub>', '<sub size="6.5" rise="2">', text)
    text = re.sub(r'<sup>', '<sup size="6.5" rise="4">', text)
    return text


def P(text, style=BODY):
    return Paragraph(T(text), style)


def eq(tex, fontsize=12):
    """Typeset a display equation with matplotlib mathtext and return it as a centred image."""
    tex = tex.replace(r'\qquad', r'\qquad ').replace(r'\le ', r'\leq ')   # mathtext spelling / spacing
    name = hashlib.md5(f'{tex}|{fontsize}'.encode()).hexdigest()[:12] + '.png'
    path = os.path.join(EQ_DIR, name)
    if not os.path.exists(path):
        fig = plt.figure(figsize=(9, 1.2))
        fig.patch.set_alpha(0)
        fig.text(0.01, 0.5, tex, fontsize=fontsize, va='center')
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04, transparent=True)
        plt.close(fig)
    w, h = PILImage.open(path).size
    w_pt, h_pt = w / 300 * 72, h / 300 * 72
    if w_pt > TEXT_W:
        h_pt, w_pt = h_pt * TEXT_W / w_pt, TEXT_W
    img = Image(path, width=w_pt, height=h_pt)
    img.hAlign = 'CENTER'
    return KeepTogether([Spacer(1, 2), img, Spacer(1, 6)])


def fig(name, caption, width=16.5 * cm):
    path = os.path.join(RES, name)
    iw, ih = PILImage.open(path).size
    img = Image(path, width=width, height=width * ih / iw)
    return KeepTogether([img, P(caption, CAP)])


def table(rows, col_widths):
    """Table whose cells are wrapping Paragraphs (bold header)."""
    data = [[Paragraph(T(str(c)), CELLB) for c in rows[0]]] + [[Paragraph(T(str(c)), CELL) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=col_widths, hAlign='LEFT')
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#999999')),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                           ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7')),
                           ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
    return KeepTogether(t)


def db(eta):
    return -10 * np.log10(max(eta, 1e-12))


def nearest(dct, key):
    return dct[min(dct, key=lambda k: abs(float(k) - key))]


# ----------------------------------------------------------------------------- data
Tj = json.load(open(os.path.join(RES, 'fdfd_taper_study.json')))
D = json.load(open(os.path.join(RES, 'fdfd_coupler_design.json')))
butt = Tj['reference'][0]['eta']
t_opt = Tj['optimum']
S1 = {float(k): v for k, v in Tj['stage1'].items()}
S2 = {float(k): v for k, v in Tj['stage2'].items()}
S2b = {float(k): v for k, v in Tj['stage2b'].items()}
tip_info_T = {float(k): v for k, v in Tj['tip_info'].items()}
best = D['best']
design = {float(k): v for k, v in D['design'].items()}
tip_info_D = {float(k): v for k, v in D['tip_info'].items()}
align = D['align']


def tol(a):
    y0s = np.array([q['y0'] for q in a['offset']]); ny = np.array([q['eta'] for q in a['offset']]) / a['offset'][0]['eta']
    ths = np.array([q['theta'] for q in a['tilt']]); nt = np.array([q['eta'] for q in a['tilt']]) / a['tilt'][0]['eta']

    def crossing(xs, ys, level=10**-0.1):
        below = np.where(ys < level)[0]
        if len(below) == 0:
            return np.nan
        k = below[0]
        return xs[0] if k == 0 else xs[k-1] + (level - ys[k-1]) * (xs[k] - xs[k-1]) / (ys[k] - ys[k-1])
    return crossing(y0s, ny), crossing(ths, nt)


# ----------------------------------------------------------------------------- document
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                        title='Edge-coupler modelling and optimisation with 2D FDFD', author='Alexander Smirnov')
S = []
S.append(P('Modelling and optimisation of a silicon inverse-taper edge coupler with 2D FDFD', TITLE))

# 1 Introduction (reused from the project report, condensed)
S.append(P('1. Introduction', H1))
S.append(P('Photonic integrated circuits (PICs) are the optical counterpart of electronic integrated circuits and are deployed in optical '
           'communications, computing, sensing, beam steering and quantum information processing [1]. Because PICs are manufactured in '
           'standard semiconductor foundries they inherit cost scaling and performance consistency. The optical interface between the PIC and an '
           'external single-mode fibre, however, remains one of the bottlenecks of the technology: insertion loss limits reach, achievable data '
           'rate and the number of on-chip components.'))
S.append(P('The underlying difficulty is a mode-size mismatch. For single-mode operation at λ ≈ 1550 nm the cross-section of a silicon '
           'waveguide (n ≈ 3.48) in silica cladding (n ≈ 1.44) is sub-micron, e.g. 450 nm × 220 nm, whereas a standard SMF-28 fibre has a mode '
           'field diameter (MFD) of 10.4 µm, reducible to 2–6 µm with micro-lensed fibres [2]. Without a coupling structure the modal overlap is '
           'very small and most of the input power is lost at the interface. Foundries specify about 1.5 dB for edge couplers and 3–4 dB per facet '
           'for grating couplers, while state-of-the-art experimental structures reach 0.05–0.5 dB [1].'))
S.append(P('Coupling schemes fall into three groups [3, 4]: edge coupling through the chip facet, surface-normal coupling with diffraction '
           'gratings, and adiabatic (evanescent) coupling along a tapered region. Edge couplers offer the highest efficiency, broad bandwidth and '
           'low polarisation dependence; their downsides are tight alignment tolerances (about ±2.5 µm for 1 dB with a cleaved fibre [1]), '
           'location at the die periphery, and incompatibility with wafer-level testing. The most common implementation is the <b>inverse taper</b>: '
           'the silicon waveguide width is reduced along the propagation direction to a sub-wavelength tip, the guided mode becomes weakly confined '
           'and expands to match the larger fibre mode, and the taper transforms it adiabatically into the mode of the bulk waveguide.'))
S.append(P('<b>Motivation.</b> The insertion loss of an edge coupler depends on many geometric parameters — tip width, taper length and '
           'shape, source position and misalignment. This work builds a two-dimensional finite-difference frequency-domain (FDFD) model in which '
           'each parameter can be varied independently, validates it against analytical results, and uses it to optimise the coupler for a fixed '
           'platform (220 nm silicon slab) and a fixed source (1 µm Gaussian waist). A separable effective-index argument is then used to '
           'translate the 2D figures into 3D estimates.'))

# 2 Mathematical framework (reused and corrected)
S.append(P('2. Mathematical framework', H1))
S.append(P('2.1 Helmholtz equation and discretisation', H2))
S.append(P('For a single frequency Maxwell’s curl equations combine into the Helmholtz equation; for TE polarisation in the x–y plane '
           '(E<sub>z</sub> only) with a source term f(x, y):'))
S.append(eq(r'$\nabla\times\mathbf{E} = -j\omega\mu\mathbf{H},\qquad \nabla\times\mathbf{H} = j\omega\varepsilon\mathbf{E}'
            r'\qquad\Longrightarrow\qquad \frac{\partial^2 E_z}{\partial x^2} + \frac{\partial^2 E_z}{\partial y^2} + k_0^2\,n^2(x,y)\,E_z = f(x,y)$'))
S.append(P('Second-order central differences on a uniform grid turn this into a sparse linear system A·E<sub>z</sub> = b with the five-point stencil'))
S.append(eq(r'$\frac{E_{i+1,j} - 2E_{i,j} + E_{i-1,j}}{\Delta x^2} + \frac{E_{i,j+1} - 2E_{i,j} + E_{i,j-1}}{\Delta y^2} + k_0^2\,n_{i,j}^2\,E_{i,j} = f_{i,j}$'))
S.append(P('With Δx = 20 nm and Δy = 10–20 nm (λ/(10 n<sub>Si</sub>) = 44 nm) the systems have 10<sup>5</sup>–10<sup>6</sup> unknowns and are solved '
           'with a sparse LU factorisation (SuperLU, COLAMD ordering). The factorisation is cached and reused whenever only the source changes, so '
           'sweeps over source position, offset and tilt cost a fraction of a second per point.'))
S.append(P('2.2 Source', H2))
S.append(P('The fibre is represented by a Gaussian beam launched from a single grid column as a soft (current-sheet) source,'))
S.append(eq(r'$E_{src}(y) = \exp\!\left[-\frac{(y - y_0)^2}{w_0^2}\right]\,\exp\!\left(-j\,k_0\,n\,\sin\theta\;y\right)$'))
S.append(P('The transverse phase factor tilts the beam by θ in the x–y plane (the original code multiplied by a constant '
           'exp(−jk sinθ x<sub>src</sub>) instead, so θ had no effect). The source radiates symmetrically into +x and −x; the −x half is absorbed '
           'by the left PML.'))
S.append(P('2.3 Effective index and mode profile', H2))
S.append(P('The TE modes of the symmetric slab follow Okamoto [6]: the characteristic equation gives the transverse wavenumber k<sub>y</sub>, '
           'and the effective index follows from the dispersion relation'))
S.append(eq(r'$k_y\tan\!\left(\frac{k_y d}{2}\right) = \gamma,\qquad k_y^2 + \gamma^2 = k_0^2\left(n_{core}^2 - n_{clad}^2\right),\qquad'
            r'\beta^2 = k_0^2 n_{core}^2 - k_y^2 = k_0^2 n_{clad}^2 + \gamma^2,\qquad n_{eff} = \beta/k_0$'))
S.append(P('(The original script evaluated √(n<sub>clad</sub><sup>2</sup> + (k<sub>y</sub>/k<sub>0</sub>)<sup>2</sup> + (γ/k<sub>0</sub>)<sup>2</sup> − 1), '
           'which reduces to √(n<sub>core</sub><sup>2</sup> − 1) for every geometry.) The mode profile is cos(k<sub>y</sub>y) in the core and '
           'cos(k<sub>y</sub>d/2)·exp(−γ(|y| − d/2)) outside, normalised to ∫E<sup>2</sup>dy = 1.'))
S.append(P('2.4 Insertion loss', H2))
S.append(P('The coupling efficiency is η = P<sub>mode</sub>/P<sub>in</sub> and IL = −10 log<sub>10</sub> η. With the exp(+jωt) convention implied by '
           'the PML stretch s = 1 − jσ/(ωε<sub>0</sub>), a +x wave is exp(−jβx) and the time-averaged Poynting flux density, the input power and the '
           'guided output power are'))
S.append(eq(r'$S_x = -\frac{1}{2\omega\mu_0}\,\mathrm{Im}\!\left(E_z^{*}\,\frac{\partial E_z}{\partial x}\right),\qquad'
            r'P_{in} = \int S_x\,dy,\qquad a = \int E_{sim}(y)\,E_{mode}^{*}(y)\,dy,\qquad'
            r'P_{mode} = |a|^2\,\frac{\sin(\beta\Delta x)/\Delta x}{2\omega\mu_0}$'))
S.append(P('P<sub>in</sub> is evaluated on a slice three cells downstream of the source (it sees only the +x half of the soft source). '
           'The overlap a uses the unit-normalised mode; in the continuum limit P<sub>mode</sub> = |a|<sup>2</sup>β/(2ωμ<sub>0</sub>), but on the '
           'grid the conserved flux of exp(−jβx) is sin(βΔx)/Δx rather than β (0.9 % lower at βΔx = 0.23), and the same factor is used for '
           'P<sub>mode</sub> so that η ≤ 1 holds exactly. The original code normalised the overlap by the power in the output slice itself, which '
           'measures mode purity rather than insertion loss.'))
S.append(P('2.5 Perfectly matched layer', H2))
S.append(P('Near the edges the equation is written in stretched coordinates with a complex stretch factor and a cubic conductivity profile [10]:'))
S.append(eq(r'$\frac{1}{s_x}\frac{\partial}{\partial x}\!\left(\frac{1}{s_x}\frac{\partial E_z}{\partial x}\right) + '
            r'\frac{1}{s_y}\frac{\partial}{\partial y}\!\left(\frac{1}{s_y}\frac{\partial E_z}{\partial y}\right) + k_0^2 n^2 E_z = f,\qquad'
            r's = 1 - j\,\frac{\sigma}{\omega\varepsilon_0},\qquad \sigma(d) = \sigma_{max}\left(\frac{d}{L_{PML}}\right)^{3}$'))
S.append(P('The stencil coefficients scale as 1/(s<sub>i</sub>s<sub>i+1/2</sub>), so the PML only absorbs when σ<sub>max</sub>/(ωε<sub>0</sub>) is of '
           'order 1–30: the value 1 × 10<sup>13</sup> S/m in the original script gave a stretch of 9 × 10<sup>8</sup>, which decouples the PML cells '
           'from the interior and turns the PML edge into a reflecting wall (the domain became a resonant cavity with a standing-wave ratio of 1 '
           'and an apparent η of 35). The repaired script uses σ<sub>max</sub> = 1 × 10<sup>5</sup> S/m (stretch ≈ 9, round-trip reflection ≈ '
           '10<sup>−6</sup> for a 0.5 µm PML), prints the stretch factor and the standing-wave ratio of the guided mode as diagnostics, and the '
           'results are stable from 10<sup>5</sup> to 10<sup>7</sup> S/m.'))
S.append(P('2.6 Geometry, sub-pixel averaging and taper profiles', H2))
S.append(P('The core is described by a local width d(x): uniform for the bulk waveguide, zero in the cladding in front of the facet, and '
           'rising from d<sub>tip</sub> to d<sub>core</sub> over the taper length. Cells cut by the core boundary receive a fill-fraction-averaged '
           'permittivity (for E<sub>z</sub> the arithmetic mean of ε is exact), so the width varies continuously instead of in 2Δy steps; without '
           'this a 20 nm tip doubled its width in one step and lost ~4 % spuriously. Two width laws are implemented: linear, and an '
           '“adiabatic” law derived from the Love et al. delineation criterion [12] with ρ the half-width, β<sub>1</sub> the local fundamental mode '
           'and β<sub>2</sub> = k<sub>0</sub>n<sub>clad</sub>, integrated to z(ρ) and scaled to the chosen length; the same criterion integrated over '
           'a linear ramp gives the shortest adiabatic length quoted below:'))
S.append(eq(r'$\left|\frac{d\rho}{dz}\right| \le \frac{\rho\,(\beta_1 - \beta_2)}{2\pi},\qquad'
            r'L_{ad} = 2\pi\int_{\rho_{tip}}^{\rho_{core}}\frac{d\rho}{\rho\,(\beta_1(\rho) - \beta_2)}$'))

# 3 Validation
S.append(P('3. Validation of the repaired model', H1))
ref_gap = Tj['reference']
S.append(P(f'<b>Single-mode choice.</b> A 500 nm slab (V = 3.21) guides three TE modes; a symmetric beam excites TE<sub>0</sub> (n<sub>eff</sub> 3.28) '
           f'and the weakly bound TE<sub>2</sub> (n<sub>eff</sub> 1.45), which beat with a period λ/(n<sub>0</sub> − n<sub>2</sub>) = 0.85 µm — the row of '
           f'dots along the core in the original report figures — and the wide beam actually puts more power into TE<sub>2</sub> than into TE<sub>0</sub>. '
           f'All results here therefore use a 220 nm slab (V = 1.41, single-mode, n<sub>eff</sub> = 2.8514), which is also the thickness of the '
           f'standard SOI platform.'))
S.append(P(f'<b>Butt coupling.</b> With the source in the cladding at the facet of the untapered 220 nm waveguide the FDFD gives '
           f'η = {butt:.3f} ({db(butt):.2f} dB), against the analytical overlap of the 1 µm beam with the slab mode of 0.368 (4.35 dB) multiplied by the '
           f'facet factor 4β<sub>in</sub>β<sub>m</sub>/(β<sub>in</sub> + β<sub>m</sub>)<sup>2</sup> = 0.89. Moving the source away from the facet reduces η '
           f'monotonically ({ref_gap[-1]["eta"]:.3f} at {ref_gap[-1]["gap"]/um:.0f} µm), following the exact angular-spectrum propagation of the beam '
           f'(Rayleigh range 2.9 µm). The guided mode at the output is a clean travelling wave (|B|/|F| ≈ 10<sup>−3</sup>), halving Δx changes η by '
           f'0.6 %, and the discrete n<sub>eff</sub> from the phase slope (2.862) converges toward the analytical 2.851.'))
S.append(fig('fdfd_waveguide.png', 'Figure 1. Default case of fdfd_waveguide.py (source inside a uniform 220 nm waveguide): steady-state '
             '|E<sub>z</sub>|<sup>2</sup> and the output cross-section compared with the analytical mode.'))

# 4 Optimisation
S.append(P('4. Optimisation of the inverse taper', H1))
S.append(P('4.1 Tip width, source-to-facet gap, taper length and profile', H2))
rows = [['tip width', 'FDFD η, gap 0, linear 15 µm', 'analytic facet overlap', 'ratio (taper transmission)', 'L<sub>ad</sub> (Love, linear)']]
for d in sorted(S1):
    ti = tip_info_T[min(tip_info_T, key=lambda k: abs(k - d))]
    rows.append([f'{d/nm:.0f} nm', f"{S1[d][0]['eta']:.3f}", f"{ti['eta_facet0']:.3f}", f"{S1[d][0]['eta']/ti['eta_facet0']:.2f}", f"{ti['L_ad']/um:.1f} µm"])
rows.append(['none (220 nm)', f'{butt:.3f}', '0.368 × 0.89', '—', '—'])
S.append(table(rows, [2.6 * cm, 4.2 * cm, 3.4 * cm, 3.8 * cm, 2.9 * cm]))
S.append(Spacer(1, 8))
S.append(P(f'Narrower tips couple better because the tip mode expands (Gaussian-equivalent radius 0.80 µm at 20 nm vs 0.17 µm at 220 nm) and '
           f'its effective index approaches the cladding. Where the 15 µm linear taper is adiabatic (L<sub>ad</sub> ≪ 15 µm) the FDFD equals the analytic '
           f'facet overlap to 1 %; the 20 nm tip, with L<sub>ad</sub> = 15.8 µm, shows taper loss. The gap sweep (Figure 2a, d) has its optimum at '
           f'zero for every geometry and follows the angular-spectrum propagation of the launched beam within 2 %: a flat-phase source cannot gain '
           f'from propagation, so the nonzero optimum gap reported in the original study is not physical. For the 20 nm tip the linear taper rises '
           f'from {S2[min(S2)][0]["eta"]:.3f} (3 µm) to {S2[max(S2)][0]["eta"]:.3f} (25 µm), whereas the Love-profile taper reaches '
           f'{S2b[min(S2b)][0]["eta"]:.3f} at 7 µm and {S2b[max(S2b)][0]["eta"]:.3f} at 25 µm — equal to the facet limit of {t_opt["eta_facet"]:.3f}: '
           f'a linear ramp is ~10× too steep at a thin tip (Δβ ∝ d<sup>2</sup>) and wastefully slow later, so taper <i>shape</i> is as important as length.'))
S.append(fig('fdfd_taper_study.png', 'Figure 2. Taper study (fdfd_taper_study.py): (a) gap sweeps per tip width; (b) tip-width dependence vs the '
             'analytic facet overlap; (c) taper length for linear and adiabatic profiles; (d) gap dependence at the optimum vs beam-propagation theory.'))
S.append(fig('fdfd_taper_optimum.png', f'Figure 3. Field of the taper-study optimum (20 nm tip, adiabatic 25 µm taper, gap 0): η = {t_opt["eta"]:.3f}, '
             f'IL = {db(t_opt["eta"]):.2f} dB. The mode stays ~0.8 µm wide over most of the taper and is compressed into the 220 nm core in the last few µm.'))

S.append(P('4.2 Design study: fabricable tips (40–100 nm), gap = 0', H2))
S.append(P(f'Since the gap cannot improve coupling and 20 nm tips are not manufacturable, the design study fixes the gap at zero and sweeps '
           f'tip widths of 40–100 nm, both profiles and lengths of 3–15 µm (linear 15 µm results are reused from the taper study). The best 2D '
           f'design is a <b>{best["d_tip"]/nm:.0f} nm tip with a {best["profile"]} taper of {best["L"]/um:.0f} µm: η = {best["eta"]:.3f}, '
           f'IL = {best["IL"]:.2f} dB</b>, against {db(butt):.2f} dB for butt coupling.'))
rows = [['tip', 'best profile', 'L (µm)', 'η 2D', 'IL 2D (dB)', 'η vertical (EIM)', 'η 3D estimate', 'IL 3D estimate (dB)']]
for d in sorted(design):
    v = design[d]
    rows.append([f'{d/nm:.0f} nm', v['profile'], f"{v['L']/um:.0f}", f"{v['eta_2d']:.3f}", f"{v['IL_2d']:.2f}", f"{v['eta_z']:.3f}", f"{v['eta_3d']:.3f}", f"{v['IL_3d']:.2f}"])
bt = D['butt']
rows.append(['no taper', '—', '—', f"{bt['eta_2d']:.3f}", f"{db(bt['eta_2d']):.2f}", f"{bt['eta_z']:.3f}", f"{bt['eta_3d']:.3f}", f"{db(bt['eta_3d']):.2f}"])
S.append(table(rows, [1.7 * cm, 2.2 * cm, 1.5 * cm, 1.5 * cm, 2.0 * cm, 2.6 * cm, 2.5 * cm, 2.8 * cm]))
S.append(Spacer(1, 8))
S.append(fig('fdfd_coupler_design.png', 'Figure 4. Design study (fdfd_coupler_design.py): (a) tip width and profile; (b) taper length per tip with the '
             'analytic facet limits; (c) separable 2D → 3D estimate; (d) lateral, (e) vertical and (f) angular misalignment tolerance.'))
S.append(fig('fdfd_coupler_design_field.png', f'Figure 5. Field of the final design ({best["d_tip"]/nm:.0f} nm tip, {best["profile"]} taper, '
             f'{best["L"]/um:.0f} µm, gap 0).'))

S.append(P('4.3 Alignment tolerance', H2))
for label, a in align.items():
    d = a['key'][0]
    y1, t1 = tol(a)
    ti = nearest(tip_info_D, d)
    S.append(P(f'<b>{label.capitalize()} (tip {d/nm:.0f} nm):</b> 1 dB excess loss at a lateral offset of {y1/um:.2f} µm, a vertical offset of '
               f'{ti["z_1dB"]/um:.2f} µm (from the EIM vertical tip mode) and an in-plane tilt of {t1/deg:.0f}°.'))
S.append(P('Position tolerance scales with the spot size (≈ 0.34·√(w<sub>0</sub><sup>2</sup> + w<sub>m</sub><sup>2</sup>) for 1 dB) and angular '
           'tolerance with its inverse, set by the smaller of beam and mode; with a 1 µm beam and a ~1 µm tip mode the position tolerance is more '
           'relaxed and the angular tolerance tighter than for butt coupling to the bare wire (0.33 µm / 33° for 1 dB), exactly as the Fourier '
           'relation between the two requires.'))

S.append(P('4.4 From 2D to 3D', H2))
S.append(P('The 2D model resolves only the in-plane (width) direction. A real wire is 220 nm thick, so the beam is also mismatched vertically. '
           'Following the effective-index method, the lateral mode index n<sub>lat</sub>(d<sub>tip</sub>) of the 2D tip becomes the core index of a '
           '220 nm vertical slab; its TE<sub>0</sub> mode is the vertical profile of the 3D tip mode, and its overlap η<sub>z</sub> with the 1 µm beam '
           'multiplies the 2D efficiency:'))
S.append(eq(r'$\eta_{3D} \approx \eta_{2D}\,(\mathrm{FDFD,\ in\ plane}) \times \eta_{z},\qquad'
            r'\eta_z = \frac{\left|\int E_{beam}(z)\,E_{vert}(z)\,dz\right|^2}{\int E_{beam}^2\,dz\int E_{vert}^2\,dz}$'))
S.append(P('For a 40 nm tip the vertical mode is 1.0 µm wide (η<sub>z</sub> = 0.95) because the low lateral index barely confines it; for a 20 nm tip '
           'it is 3.6 µm wide and η<sub>z</sub> drops to 0.53 — in 3D there is therefore a genuine optimum tip width for a given spot — and for the '
           'untapered 220 nm wire η<sub>z</sub> = 0.47. The estimate ignores substrate leakage through the buried oxide (a real loss channel for '
           'modes a few µm wide), polarisation and the breakdown of the EIM near cut-off, and therefore remains optimistic; measured per-facet '
           'losses of 0.5–1.5 dB for comparable 3D devices bracket the 40–60 nm rows of the table.'))

# 5 Plausibility review
S.append(P('5. Physical plausibility review', H1))
S.append(P('• <b>Tip width:</b> η rises monotonically as the tip narrows, reproduces the analytic facet overlap to 1 % wherever the taper is adiabatic, '
           'and the tip-mode sizes follow the thin-slab law 1/γ = 2/(k<sub>0</sub><sup>2</sup>(n<sub>core</sub><sup>2</sup> − n<sub>clad</sub><sup>2</sup>)d). '
           'The sensitivity (~0.5 dB per 10 nm near 40 nm) is why real couplers are fabrication-limited.'))
S.append(P('• <b>Gap:</b> maximal at zero for every geometry; the decrease matches exact beam propagation (and the 1-D Joyce–DeLoach formula [13]) '
           'to 2–3 %. Any propagation of a flat-phase beam only widens it and curves its wavefront, so a nonzero optimum is not physical for this source.'))
S.append(eq(r'$\eta_{JD}(z) = \frac{2}{\sqrt{\left(\frac{w_1}{w_2} + \frac{w_2}{w_1}\right)^2 + \left(\frac{\lambda z}{\pi n w_1 w_2}\right)^2}}$', 11))
S.append(P('• <b>Taper length and shape:</b> the length scale is set by the Love criterion (L<sub>ad</sub> = 15.8 / 4.9 / 2.6 / 0.8 µm for 20 / 40 / 60 / '
           '120 nm tips); a linear taper saturates slowly because it is too steep at the tip, the criterion-derived profile reaches the facet limit '
           'within 3–25 µm — textbook adiabatic-taper behaviour, and consistent with the original report’s weak L<sub>taper</sub> dependence.'))
S.append(P('• <b>Power bookkeeping:</b> at the optimum P<sub>mode</sub>/P<sub>in</sub> and the total forward flux at the output agree to 0.1 %, so '
           'all surviving power is guided; the deficit is the initial mismatch radiated near the facet; η ≤ 1 everywhere.'))
S.append(P('• <b>Absolute numbers:</b> 0.2–1.1 dB in 2D for a 2 µm MFD spot become 0.4–1.5 dB after the vertical overlap, substrate leakage and '
           'realistic tips — the range quoted for state-of-the-art and foundry couplers. A cleaved SMF-28 (10.4 µm MFD) against the bare wire would '
           'give ~19 dB in the same model, consistent with the ≥10 dB reported for direct coupling [11].'))

# 6 Limitations
S.append(P('6. Limitations and outlook', H1))
S.append(P('The model is scalar and two-dimensional: no vertical confinement, substrate leakage or polarisation conversion; the soft source sits in '
           'the cladding so facet reflection (10.8 % from silica into the bulk mode, ~0 into a tip mode) is not included; the tip is resolved with '
           'four to ten cells across, and the tapers are at most 25 µm, whereas real 3D tapers are 100–300 µm because the mode evolves more slowly. '
           'Natural extensions are a 3D (or semi-vectorial effective-index) mode solver for the vertical direction, a total-field/scattered-field '
           'source [7] to include facet reflections, and a wavelength sweep for the bandwidth.'))

# References (from the project report)
S.append(P('References', H1))
for i, r in enumerate([
    'Ranno, L. et al., “Integrated photonics packaging: challenges and opportunities,” ACS Photonics 9, 3467–3485 (2022).',
    'Corning Inc., SMF-28 Ultra Optical Fiber — Product Information PI-1424-AEN.',
    'Son, G. et al., “High-efficiency broadband light coupling between optical fibers and photonic integrated circuits,” Nanophotonics 7, 1845–1864 (2018).',
    'Marchetti, R. et al., “Coupling strategies for silicon photonics integrated chips,” Photonics Research 7, 201–239 (2019).',
    'Okamoto, K., Fundamentals of Optical Waveguides, Elsevier (2021), Sec. 2.3.2.',
    'Okamoto, K., Fundamentals of Optical Waveguides, Elsevier (2021), Sec. 2.2.4.',
    'Rumpf, R. C., “Simple implementation of arbitrarily shaped total-field/scattered-field regions in FDFD,” PIER B 36, 221–248 (2012).',
    'Jackson, J. D., Classical Electrodynamics, Sec. 6.4.',
    'Okamoto, K., Fundamentals of Optical Waveguides, Elsevier (2021), Sec. 2.3.2.',
    'Berenger, J.-P., “A perfectly matched layer for the absorption of electromagnetic waves,” J. Comput. Phys. 114, 185–200 (1994).',
    'Chrostowski, L. and Hochberg, M., Silicon Photonics Design: From Devices to Systems, Cambridge University Press (2015).',
    'Love, J. D. et al., “Tapered single-mode fibres and devices. Part 1: Adiabaticity criteria,” IEE Proc. J 138, 343–354 (1991).',
    'Joyce, W. B. and DeLoach, B. C., “Alignment of Gaussian beams,” Appl. Opt. 23, 4187–4196 (1984).',
], 1):
    S.append(P(f'[{i}] {r}', SMALL))

# Appendix: beam-parameter study
S.append(PageBreak())
S.append(P('Appendix A. Beam-parameter study (untapered waveguide)', H1))
S.append(P('Before the coupler optimisation the source parameters were swept on the untapered 220 nm waveguide (fdfd_coupler_study.py): the optimum '
           'waist equals the mode radius (0.18–0.20 µm) with η → 1, efficiency falls as 2w<sub>m</sub>/w<sub>0</sub> for wide beams, and the 1 dB '
           'tolerances are 64 nm / 38° for the matched spot and 0.33 µm / 33° for the 1 µm beam — position tolerance grows with spot size, angular '
           'tolerance is set by the smaller spot. Because beam size and slab thickness are fixed by the setup and the platform, these are reference '
           'numbers, not design variables.'))
S.append(fig('fdfd_beam_parameter_study.png', 'Figure A1. Beam-parameter sweeps on the untapered waveguide: waist, lateral offset, tilt and core thickness.'))

doc.build(S)
print('wrote', OUT)
