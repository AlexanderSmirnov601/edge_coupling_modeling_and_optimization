"""
fdfd_taper_study.py -- inverse-taper edge-coupler optimisation with fdfd_waveguide.py

Fixed:      bulk waveguide d_core = 220 nm, Gaussian source w0 = 1 um (y0 = 0, theta = 0),
            lambda = 1550 nm, Si (3.48) / SiO2 (1.44).
Optimised:  tip width d_tip, taper length L_taper, taper width profile (linear / adiabatic),
            source-to-facet gap.

Layout along x:
   | PML | cladding, source plane somewhere here | facet: d_tip -> taper -> d_core | 3 um bulk | PML |
The gap is swept by moving the source plane, which only changes the right-hand
side, so all gap points of one geometry reuse its LU factorization.

Independent physics used to judge the FDFD results (no FDFD involved):
  * local TE0 mode of a uniform slab of width d_tip (size, n_eff) + thin-slab asymptote
  * exact angular-spectrum propagation of the launched beam across the gap in uniform
    cladding, overlapped with the tip mode  -> expected facet efficiency vs gap
  * Joyce-DeLoach Gaussian-to-Gaussian coupling formula for the same quantity
  * Love et al. adiabaticity criterion -> shortest adiabatic taper length L_ad(d_tip)

Usage:
    python fdfd_taper_study.py            # run all sweeps (~15 min), save JSON + figures
    python fdfd_taper_study.py --replot   # regenerate the figure from the saved JSON

Outputs (results/): fdfd_taper_study.png, fdfd_taper_optimum.png, fdfd_taper_study.json
"""

import os
import sys
import time
import json
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import fdfd_waveguide as fw

OUT_PNG  = os.path.join(fw.RESULTS, 'fdfd_taper_study.png')
OUT_OPT  = os.path.join(fw.RESULTS, 'fdfd_taper_optimum.png')
OUT_JSON = os.path.join(fw.RESULTS, 'fdfd_taper_study.json')
um, nm = 1e-6, 1e-9

# Palette roles (fixed): blue = FDFD, orange = analytical, aqua = closed-form / alt. profile, gray = reference
C_SIM, C_AN, C_SHEET, C_REF = '#2a78d6', '#eb6834', '#1baf7a', '#8a8883'
BLUES = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c', '#042c5c']   # sequential, light -> dark
INK2 = '#52514e'

fw.LU_OPTIONS = {}   # SuperLU default (COLAMD) -- benchmarked 5x faster than MMD_AT_PLUS_A on this grid

# ----------------------------------------------------------------------------- setup
GAP_MAX = 4.0 * um
X_FACET = 1.0 * um + GAP_MAX          # facet position; the source moves in [X_FACET - GAP_MAX, X_FACET]
BULK    = 3.0 * um                    # uniform waveguide after the taper
base = dict(fw.params, dy=10 * nm, y_offset=0.5, Ly=5.0 * um,
            w0=1.0 * um, y0=0.0, theta=0.0, d_core=220 * nm)

GAPS       = np.array([0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]) * um
TIPS       = np.array([20, 40, 60, 80, 120]) * nm
L_REF      = 15 * um
LENGTHS    = np.array([3, 7, 15, 25]) * um      # linear profile
LENGTHS_AD = np.array([7, 15, 25]) * um         # adiabatic (Love) profile


def geom(d_tip, L_taper, profile='linear'):
    p = dict(base)
    p.update(x_facet=X_FACET, d_tip=d_tip, L_taper=L_taper, taper_profile=profile,
             Lx=X_FACET + L_taper + BULK + base['L_pml'])
    return p


def run_gaps(p, gaps, label):
    rows, t = [], time.perf_counter()
    for gap in gaps:
        r = fw.simulate(dict(p, x_src=X_FACET - gap), verbose=False)
        rows.append(dict(gap=gap, eta=r['eta'], IL=r['IL_dB'], P_in=r['P_in'], P_out=r['P_out'],
                         P_total=r['P_total'], swr=r['swr'], n_eff_sim=r['n_eff_sim']))
        print(f"  {label}: gap = {gap/um:4.2f} um  eta = {r['eta']:.4f}  IL = {r['IL_dB']:5.2f} dB"
              f"  |B|/|F| = {r['swr']:.4f}  P_total/P_in = {r['P_total']/r['P_in']:.3f}", flush=True)
    print(f"  [{time.perf_counter()-t:.0f} s incl. factorization, {r['g'].N_total} unknowns]", flush=True)
    return rows


def best_of(rows):
    return max(rows, key=lambda r: r['eta'])


# ----------------------------------------------------------------------------- analytics (no FDFD)
k0 = 2 * np.pi / base['lambda0']
n_core, n_clad = base['n_core'], base['n_clad']
NA2 = n_core**2 - n_clad**2
k_clad = k0 * n_clad
y_grid = (np.arange(int(round(base['Ly'] / base['dy']))) + base['y_offset']) * base['dy'] - base['Ly'] / 2


def local_mode(d):
    """Fundamental TE mode of a uniform slab of width d on the study's y grid."""
    g = SimpleNamespace(k0=k0, d_core=d, n_core=n_core, n_clad=n_clad, y=y_grid)
    return fw.slab_mode(g)


def overlap(E1, E2, y):
    return abs(integrate.trapezoid(E1 * np.conj(E2), y))**2 / (
        integrate.trapezoid(abs(E1)**2, y) * integrate.trapezoid(abs(E2)**2, y))


def propagate(E0, y, z, launched=True):
    """Exact 2D angular-spectrum propagation of E0(y) by distance z in uniform
    cladding.  launched=True applies the 1/kx weighting with which a current
    sheet launches each plane-wave component (relative to ky = 0)."""
    N, dy = len(y), y[1] - y[0]
    ky = 2 * np.pi * np.fft.fftfreq(N, dy)
    kx = np.where(ky**2 <= k_clad**2, np.sqrt(np.maximum(k_clad**2 - ky**2, 0.0)) + 0j,
                  -1j * np.sqrt(np.maximum(ky**2 - k_clad**2, 0.0)))
    spec = np.fft.fft(E0)
    if launched:
        spec = spec * np.where(np.abs(kx) > 0, k_clad / np.where(np.abs(kx) > 0, kx, 1.0), 0.0)
    return np.fft.ifft(spec * np.exp(-1j * kx * z))


def facet_efficiency(d_tip, gap, launched=True):
    """Expected efficiency of the beam -> tip-mode transfer at the facet after the
    launched beam has travelled 'gap' through cladding (no taper loss included)."""
    m = local_mode(d_tip)
    E0 = np.exp(-y_grid**2 / base['w0']**2)
    E = propagate(E0, y_grid, gap, launched)
    return overlap(E, m.E_mode, y_grid)


def joyce_deloach(w1, w2, z):
    """Gaussian-beam (waist w1) to Gaussian-mode (w2) coupling after distance z,
    one transverse dimension (slab): the square root of the usual circular-beam
    form 4/[(w1/w2 + w2/w1)^2 + (lambda z / (pi n w1 w2))^2]."""
    return 2.0 / np.sqrt((w1 / w2 + w2 / w1)**2 + (base['lambda0'] * z / (np.pi * n_clad * w1 * w2))**2)


def gaussian_equivalent_radius(E_mode):
    ws = np.linspace(0.05, 3.0, 600) * um
    ov = [overlap(np.exp(-y_grid**2 / w**2), E_mode, y_grid) for w in ws]
    return ws[int(np.argmax(ov))], max(ov)


def adiabatic_length(d_tip, d_end=base['d_core']):
    """Love et al. delineation criterion integrated over a linear-in-width taper:
    |d rho/dz| <= rho * (beta_1 - beta_2) / (2 pi), beta_2 = k0 n_clad  ->
    L_ad = 2 pi * int d rho / (rho * k0 (n_eff(2 rho) - n_clad))."""
    rho = np.linspace(d_tip / 2, d_end / 2, 400)
    dbeta = np.array([k0 * (fw.slab_n_eff(2 * r, k0, n_core, n_clad) - n_clad) for r in rho])
    return integrate.trapezoid(2 * np.pi / (rho * dbeta), rho)


def tip_table(tips):
    info = {}
    for d in tips:
        m = local_mode(d)
        w_eq, ov_eq = gaussian_equivalent_radius(m.E_mode)
        gamma_thin = k0**2 * NA2 * d / 2
        info[float(d)] = dict(n_eff=m.n_eff, w_1e=m.w_1e, w_2m=m.w_2m, w_eq=w_eq, w_thin=1 / gamma_thin,
                              eta_facet0=facet_efficiency(d, 0.0), L_ad=adiabatic_length(d))
    return info


# ----------------------------------------------------------------------------- the study
def run_study():
    T0 = time.perf_counter()
    print("Stage 0: butt coupling (bulk waveguide starts at the facet, no taper)", flush=True)
    ref_rows = run_gaps(geom(base['d_core'], 1 * nm), GAPS, "butt 220 nm")

    print(f"\nStage 1: tip width (L_taper = {L_REF/um:.0f} um, linear) x gap", flush=True)
    S1 = {float(d): run_gaps(geom(d, L_REF), GAPS, f"tip {d/nm:3.0f} nm") for d in TIPS}
    d_best = max(S1, key=lambda d: best_of(S1[d])['eta'])
    print(f"  -> best tip {d_best/nm:.0f} nm at gap {best_of(S1[d_best])['gap']/um:.2f} um: "
          f"eta = {best_of(S1[d_best])['eta']:.4f}", flush=True)

    print(f"\nStage 2: linear taper length (d_tip = {d_best/nm:.0f} nm) x gap", flush=True)
    S2 = {float(L_REF): S1[d_best]}
    for L in LENGTHS:
        if float(L) not in S2:
            S2[float(L)] = run_gaps(geom(d_best, L), GAPS, f"linear L {L/um:4.1f} um")
    L_lin = max(S2, key=lambda L: best_of(S2[L])['eta'])
    print(f"  -> best linear length {L_lin/um:.0f} um at gap {best_of(S2[L_lin])['gap']/um:.2f} um: "
          f"eta = {best_of(S2[L_lin])['eta']:.4f}", flush=True)

    print(f"\nStage 2b: adiabatic (Love-profile) taper length (d_tip = {d_best/nm:.0f} nm) x gap", flush=True)
    S2b = {float(L): run_gaps(geom(d_best, L, 'adiabatic'), GAPS, f"adiabatic L {L/um:4.1f} um") for L in LENGTHS_AD}
    L_ad_best = max(S2b, key=lambda L: best_of(S2b[L])['eta'])
    print(f"  -> best adiabatic-profile length {L_ad_best/um:.0f} um at gap {best_of(S2b[L_ad_best])['gap']/um:.2f} um: "
          f"eta = {best_of(S2b[L_ad_best])['eta']:.4f}", flush=True)

    if best_of(S2b[L_ad_best])['eta'] > best_of(S2[L_lin])['eta']:
        profile, L_best, rows_best = 'adiabatic', L_ad_best, S2b[L_ad_best]
    else:
        profile, L_best, rows_best = 'linear', L_lin, S2[L_lin]
    opt = best_of(rows_best)

    print("\nStage 3: field plot of the optimum", flush=True)
    r_opt = fw.simulate(dict(geom(d_best, L_best, profile), x_src=X_FACET - opt['gap']), verbose=True)
    fw.make_plots(r_opt, OUT_OPT, show=False)
    plt.close('all')

    print("\nAnalytics", flush=True)
    tips_fine = np.array([20, 30, 40, 50, 60, 80, 100, 120, 160, 220]) * nm
    tip_info = tip_table(tips_fine)
    for d, ti in tip_info.items():
        print(f"  tip {d/nm:4.0f} nm: n_eff = {ti['n_eff']:.4f}  1/e half-width = {ti['w_1e']/um:.3f} um  "
              f"Gaussian-equivalent radius = {ti['w_eq']/um:.3f} um  thin-slab 1/gamma = {ti['w_thin']/um:.3f} um  "
              f"facet overlap(gap=0) = {ti['eta_facet0']:.3f}  L_ad = {ti['L_ad']/um:6.1f} um", flush=True)

    eta_facet_best = facet_efficiency(d_best, opt['gap'])
    data = dict(base=base, gaps=list(GAPS), tips=list(TIPS), lengths=list(LENGTHS), lengths_ad=list(LENGTHS_AD),
                reference=ref_rows,
                stage1={str(k): v for k, v in S1.items()},
                stage2={str(k): v for k, v in S2.items()},
                stage2b={str(k): v for k, v in S2b.items()},
                optimum=dict(d_tip=d_best, profile=profile, L_taper=L_best, gap=opt['gap'], eta=opt['eta'],
                             eta_facet=eta_facet_best, T_taper=opt['eta'] / eta_facet_best,
                             linear_best=dict(L=L_lin, eta=best_of(S2[L_lin])['eta']),
                             adiabatic_best=dict(L=L_ad_best, eta=best_of(S2b[L_ad_best])['eta'])),
                tip_info={str(k): v for k, v in tip_info.items()},
                minutes=(time.perf_counter() - T0) / 60)
    json.dump(data, open(OUT_JSON, 'w'), indent=1, default=float)
    return data


# ----------------------------------------------------------------------------- summary + figure
def _best_length(stage):
    """(L, eta) of the best taper length in a {str(L): rows} stage dict (None if empty)."""
    if not stage:
        return None
    L = max(stage, key=lambda k: best_of(stage[k])['eta'])
    return dict(L=float(L), eta=best_of(stage[L])['eta'])


def summary(data):
    o = data['optimum']
    ref_best = max(r['eta'] for r in data['reference'])
    lin = o.get('linear_best') or _best_length(data.get('stage2', {}))
    ad  = o.get('adiabatic_best') or _best_length(data.get('stage2b', {}))
    print("\n=== Optimum (w0 = 1 um, d_core = 220 nm fixed) ===")
    print(f"  d_tip = {o['d_tip']/nm:.0f} nm, {o.get('profile', 'linear')} taper profile, "
          f"L_taper = {o['L_taper']/um:.0f} um, gap = {o['gap']/um:.2f} um")
    print(f"  eta = {o['eta']:.4f}  ->  IL = {-10*np.log10(o['eta']):.2f} dB   "
          f"(butt coupling at its best gap: eta = {ref_best:.4f}, IL = {-10*np.log10(ref_best):.2f} dB)")
    if lin:
        print(f"  linear-taper best:    L = {lin['L']/um:.0f} um, eta = {lin['eta']:.4f}")
    if ad:
        print(f"  adiabatic-taper best: L = {ad['L']/um:.0f} um, eta = {ad['eta']:.4f}")
    print(f"  analytic facet efficiency at that gap (beam propagated, overlapped with tip mode): {o['eta_facet']:.4f}")
    print(f"  implied taper transmission eta / eta_facet = {o['T_taper']:.3f}   ({-10*np.log10(o['T_taper']):.2f} dB)")
    ti = data['tip_info'][str(float(o['d_tip']))]
    print(f"  tip mode: Gaussian-equivalent radius {ti['w_eq']/um:.3f} um vs beam w0 = {base['w0']/um:.2f} um; "
          f"L_ad(d_tip, linear) = {ti['L_ad']/um:.1f} um")
    if 'minutes' in data:
        print(f"Total simulation time: {data['minutes']:.1f} min")


def make_figure(data):
    o = data['optimum']
    d_best, L_best, profile = o['d_tip'], o['L_taper'], o.get('profile', 'linear')
    S1  = {float(k): v for k, v in data['stage1'].items()}
    S2  = {float(k): v for k, v in data['stage2'].items()}
    S2b = {float(k): v for k, v in data.get('stage2b', {}).items()}
    tip_info = {float(k): v for k, v in data['tip_info'].items()}
    tips, ref_rows = np.array(data['tips']), data['reference']
    rows_best = (S2b if profile == 'adiabatic' else S2)[float(L_best)]

    gaps_fine = np.linspace(0, GAP_MAX, 61)
    facet_best = np.array([facet_efficiency(d_best, z) for z in gaps_fine])
    facet_ideal = np.array([facet_efficiency(d_best, z, launched=False) for z in gaps_fine])
    w_eq = tip_info[float(d_best)]['w_eq']
    jd = joyce_deloach(base['w0'], w_eq, gaps_fine) * tip_info[float(d_best)]['eta_facet0'] / joyce_deloach(base['w0'], w_eq, 0.0)
    tips_fine = np.array(sorted(tip_info))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) eta vs gap for each tip width (+ butt reference)
    ax = axes[0, 0]
    ax.plot([r['gap']/um for r in ref_rows], [r['eta'] for r in ref_rows], 's--', color=C_REF, lw=1.2, ms=4,
            label='no taper (butt, 220 nm)')
    for d, col in zip(tips, BLUES[1:]):
        rows = S1[float(d)]
        ax.plot([r['gap']/um for r in rows], [r['eta'] for r in rows], 'o-', color=col, lw=1.5, ms=4,
                label=f'FDFD tip {d/nm:.0f} nm')
    ax.set_xlabel('source-to-facet gap (µm)'); ax.set_ylabel('coupling efficiency η')
    ax.set_title(f'(a) Gap sweep per tip width  (linear taper, $L_{{taper}}$ = {L_REF/um:.0f} µm)', fontsize=11)
    ax.set_xlim(0, GAP_MAX/um); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.25)

    # (b) eta vs tip width: FDFD vs analytic facet overlap at gap 0
    ax = axes[0, 1]
    ax.plot(tips_fine/nm, [tip_info[d]['eta_facet0'] for d in tips_fine], '-', color=C_AN, lw=2,
            label='analytic facet overlap, gap = 0 (no taper loss)')
    ax.plot(tips/nm, [S1[float(d)][0]['eta'] for d in tips], 's-', color=C_SIM, lw=1.2, ms=5, mfc='white',
            label='FDFD, gap = 0')
    ax.plot(tips/nm, [best_of(S1[float(d)])['eta'] for d in tips], 'o-', color=C_SIM, lw=1.5, ms=6,
            label='FDFD, best gap')
    for k, d in enumerate(tips):
        ti = tip_info[float(d)]
        ax.annotate(f"$w_{{eq}}$ = {ti['w_eq']/um:.2f} µm, $L_{{ad}}$ = {ti['L_ad']/um:.0f} µm",
                    xy=(d/nm, best_of(S1[float(d)])['eta']), xytext=(14, -4 - 12 * (k % 2)),
                    textcoords='offset points', ha='left', fontsize=7.5, color=INK2)
    ax.set_xlabel('tip width $d_{tip}$ (nm)'); ax.set_ylabel('coupling efficiency η')
    ax.set_title(f'(b) Tip width  (linear taper, $L_{{taper}}$ = {L_REF/um:.0f} µm)', fontsize=11)
    ax.set_xlim(0, 230); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.25)

    # (c) eta vs taper length at the best tip, linear vs adiabatic profile
    ax = axes[1, 0]
    Ls, Lsb = sorted(S2), sorted(S2b)
    ax.plot([L/um for L in Ls], [best_of(S2[L])['eta'] for L in Ls], 'o-', color=C_SIM, lw=1.5, ms=6,
            label='FDFD, linear profile (best gap)')
    ax.plot([L/um for L in Ls], [S2[L][0]['eta'] for L in Ls], 's-', color=C_SIM, lw=1.0, ms=5, mfc='white',
            label='FDFD, linear profile, gap = 0')
    ax.plot([L/um for L in Lsb], [best_of(S2b[L])['eta'] for L in Lsb], 'D-', color=C_SHEET, lw=1.5, ms=6,
            label='FDFD, adiabatic (Love) width profile (best gap)')
    ax.axhline(tip_info[float(d_best)]['eta_facet0'], color=C_AN, lw=1.5, ls='--',
               label='analytic facet efficiency (lossless-taper limit)')
    L_ad = tip_info[float(d_best)]['L_ad']
    ax.axvline(L_ad/um, color=C_REF, lw=1, ls=':')
    ax.text(L_ad/um + 0.4, 0.30, f"$L_{{ad}}$ = {L_ad/um:.1f} µm (Love criterion, linear)", rotation=90,
            fontsize=8, color=INK2, va='bottom')
    ax.set_xlabel('taper length $L_{taper}$ (µm)'); ax.set_ylabel('coupling efficiency η')
    ax.set_title(f'(c) Taper length and profile  ($d_{{tip}}$ = {d_best/nm:.0f} nm)', fontsize=11)
    ax.set_xlim(0, max(LENGTHS)/um * 1.05); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.25)

    # (d) eta vs gap at the optimum: FDFD vs analytic propagation vs Joyce-DeLoach
    ax = axes[1, 1]
    ax.plot(gaps_fine/um, facet_best, '-', color=C_AN, lw=2, label='angular-spectrum propagation × tip-mode overlap')
    ax.plot(gaps_fine/um, facet_ideal, '--', color=C_AN, lw=1.2, label='same, ideal Gaussian (no 1/$k_x$ launch factor)')
    ax.plot(gaps_fine/um, jd, ':', color=C_SHEET, lw=2, label=f'Joyce–DeLoach (1-D), $w_{{eq}}$ = {w_eq/um:.2f} µm')
    ax.plot([r['gap']/um for r in rows_best], [r['eta'] for r in rows_best], 'o-', color=C_SIM, lw=1.5, ms=6,
            label=f'FDFD (tip {d_best/nm:.0f} nm, {profile} taper, L = {L_best/um:.0f} µm)')
    ax.plot(gaps_fine/um, facet_best * o['T_taper'], '-.', color=C_SIM, lw=1.0,
            label=f"analytic × taper transmission {o['T_taper']:.2f}")
    ax.set_xlabel('source-to-facet gap (µm)'); ax.set_ylabel('coupling efficiency η')
    ax.set_title('(d) Gap dependence at the optimum: FDFD vs. beam-propagation theory', fontsize=11)
    ax.set_xlim(0, GAP_MAX/um); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(True, alpha=0.25)

    fig.suptitle('Inverse-taper edge coupler, 1 µm Gaussian → 220 nm Si slab: FDFD optimisation vs. physics', fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Figure saved to {OUT_PNG}")
    return fig


if __name__ == '__main__':
    if '--replot' in sys.argv:
        data = json.load(open(OUT_JSON))
    else:
        data = run_study()
    summary(data)
    make_figure(data)
    plt.show()
