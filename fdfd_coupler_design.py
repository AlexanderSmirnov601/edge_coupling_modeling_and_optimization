"""
fdfd_coupler_design.py -- final inverse-taper edge-coupler design study (gap = 0)

Fixed by the platform / setup:  220 nm Si slab, 1 um Gaussian beam (1/e field radius),
                                lambda = 1550 nm, Si (3.48) / SiO2 (1.44), source at the facet.
Design variables:               tip width 40-100 nm, taper width profile (linear / adiabatic),
                                taper length 3-15 um.
Then, for the best coupler:     alignment tolerance (lateral offset, in-plane tilt) by FDFD,
                                and an effective-index (separable) 2D -> 3D estimate including
                                the vertical overlap and vertical alignment tolerance.

Linear-taper results at 15 um for 40/60/80 nm tips are reused from results/fdfd_taper_study.json
(gap-0 rows); the 2D physics at gap 0 does not depend on the domain length used there.

Outputs (results/): fdfd_coupler_design.png, fdfd_coupler_design_field.png, fdfd_coupler_design.json
Usage:  python fdfd_coupler_design.py            (~10 min)
        python fdfd_coupler_design.py --replot   (figure from the saved JSON)
"""

import os
import sys
import json
import time
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import fdfd_waveguide as fw
from fdfd_taper_study import (base, y_grid, k0, n_core, n_clad, local_mode, overlap,
                              facet_efficiency, adiabatic_length, best_of)

OUT_PNG   = os.path.join(fw.RESULTS, 'fdfd_coupler_design.png')
OUT_FIELD = os.path.join(fw.RESULTS, 'fdfd_coupler_design_field.png')
OUT_JSON  = os.path.join(fw.RESULTS, 'fdfd_coupler_design.json')
IN_TAPER  = os.path.join(fw.RESULTS, 'fdfd_taper_study.json')
um, nm, deg = 1e-6, 1e-9, np.pi / 180

C_SIM, C_AN, C_SHEET, C_REF = '#2a78d6', '#eb6834', '#1baf7a', '#8a8883'
BLUES = ['#9ecae1', '#6baed6', '#3182bd', '#08519c']
INK2 = '#52514e'
ONE_DB = 10**(-0.1)

# ----------------------------------------------------------------------------- design space
TIPS       = np.array([40, 60, 80, 100]) * nm
LENGTHS_AD = np.array([3, 7, 15]) * um
L_LIN      = 15 * um
X_FACET    = 1.5 * um                  # source sits on the facet (gap = 0)
BULK       = 3.0 * um
Y0_LIST    = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]) * um
TH_LIST    = np.array([0, 5, 10, 15, 20, 25, 30, 40]) * deg


def geom(d_tip, L_taper, profile):
    p = dict(base)
    p.update(x_facet=X_FACET, d_tip=d_tip, L_taper=L_taper, taper_profile=profile,
             Lx=X_FACET + L_taper + BULK + base['L_pml'], x_src=X_FACET)
    return p


def record(r):
    return dict(eta=r['eta'], IL=r['IL_dB'], swr=r['swr'], P_total_over_P_in=r['P_total'] / r['P_in'],
                n_eff_sim=r['n_eff_sim'])


def nearest(dct, key):
    return dct[min(dct, key=lambda k: abs(float(k) - key))]


# ----------------------------------------------------------------------------- 2D -> 3D (effective-index, separable) estimate
Z_GRID = (np.arange(2000) + 0.5) * 10 * nm - 10 * um     # wide vertical grid: weakly bound vertical modes need it


def vertical_mode(d_tip, thickness=base['d_core']):
    """Vertical (thickness-direction) TE0 mode at the facet of a w x 220 nm wire tip:
    effective-index method -- the lateral 2D mode index of a slab of width d_tip
    becomes the core index of a 220 nm thick vertical slab."""
    n_lat = fw.slab_n_eff(d_tip, k0, n_core, n_clad)
    g = SimpleNamespace(k0=k0, d_core=thickness, n_core=n_lat, n_clad=n_clad, y=Z_GRID)
    return n_lat, fw.slab_mode(g)


def vertical_overlap(d_tip, z0=0.0):
    n_lat, m = vertical_mode(d_tip)
    E = np.exp(-(Z_GRID - z0)**2 / base['w0']**2)
    return overlap(E, m.E_mode, Z_GRID), n_lat, m


def crossing(xs, ys, level):
    below = np.where(np.asarray(ys) < level)[0]
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return xs[0]
    return xs[k-1] + (level - ys[k-1]) * (xs[k] - xs[k-1]) / (ys[k] - ys[k-1])


# ----------------------------------------------------------------------------- study
def run_study():
    T0 = time.perf_counter()
    prev = json.load(open(IN_TAPER)) if os.path.exists(IN_TAPER) else None

    # Checkpoint of the expensive stages (1 and 2), so that a failure in the cheap
    # analytics/plotting stages never costs the simulations again.
    CKPT = os.path.join(fw.RESULTS, 'fdfd_coupler_design_checkpoint.json')
    ck = json.load(open(CKPT)) if os.path.exists(CKPT) else {}

    def save_ck():
        json.dump(ck, open(CKPT, 'w'), default=float)

    # Stage 1: tip width x profile x length at gap 0
    print("Stage 1: tip width x taper profile x length (gap = 0)", flush=True)
    S = {}   # (tip, profile, L) -> record
    if 'sweep' in ck:
        S = {(float(a), b, float(c)): v for (a, b, c), v in ((k.split('|'), v) for k, v in ck['sweep'].items())}
        print("  [stage 1 loaded from checkpoint]", flush=True)
    for d in (TIPS if 'sweep' not in ck else []):
        for L in LENGTHS_AD:
            t = time.perf_counter()
            r = fw.simulate(geom(d, L, 'adiabatic'), verbose=False)
            S[(float(d), 'adiabatic', float(L))] = record(r)
            print(f"  tip {d/nm:3.0f} nm  adiabatic L = {L/um:4.1f} um: eta = {r['eta']:.4f}  IL = {r['IL_dB']:5.2f} dB"
                  f"  |B|/|F| = {r['swr']:.4f}  P_total/P_in = {r['P_total']/r['P_in']:.3f}   "
                  f"[{time.perf_counter()-t:.0f} s, {r['g'].N_total} unknowns]", flush=True)
        reused = None
        if prev is not None:
            rows = nearest(prev['stage1'], float(d)) if any(abs(float(k) - d) < 1e-12 for k in prev['stage1']) else None
            if rows is not None:
                g0 = rows[0]
                reused = dict(eta=g0['eta'], IL=g0['IL'], swr=g0['swr'], P_total_over_P_in=g0['P_total'] / g0['P_in'],
                              n_eff_sim=g0['n_eff_sim'])
        if reused is not None:
            S[(float(d), 'linear', float(L_LIN))] = reused
            print(f"  tip {d/nm:3.0f} nm  linear    L = {L_LIN/um:4.1f} um: eta = {reused['eta']:.4f}  IL = {reused['IL']:5.2f} dB   [reused from fdfd_taper_study.json]", flush=True)
        else:
            t = time.perf_counter()
            r = fw.simulate(geom(d, L_LIN, 'linear'), verbose=False)
            S[(float(d), 'linear', float(L_LIN))] = record(r)
            print(f"  tip {d/nm:3.0f} nm  linear    L = {L_LIN/um:4.1f} um: eta = {r['eta']:.4f}  IL = {r['IL_dB']:5.2f} dB"
                  f"  |B|/|F| = {r['swr']:.4f}   [{time.perf_counter()-t:.0f} s]", flush=True)

    ck['sweep'] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in S.items()}
    save_ck()
    best_key = max(S, key=lambda k: S[k]['eta'])
    d_best, prof_best, L_best = best_key
    print(f"  -> best: tip {d_best/nm:.0f} nm, {prof_best} taper, L = {L_best/um:.0f} um: eta = {S[best_key]['eta']:.4f} "
          f"(IL = {S[best_key]['IL']:.2f} dB)", flush=True)

    # Stage 2: alignment tolerance by FDFD (source moved / tilted; LU reused per geometry)
    print("\nStage 2: alignment tolerance (lateral offset y0, in-plane tilt theta)", flush=True)
    align = ck.get('align', {})
    if align:
        print("  [stage 2 loaded from checkpoint]", flush=True)
    wide_key = max((k for k in S if k[0] == float(TIPS.max()) and k[1] == 'adiabatic'), key=lambda k: S[k]['eta'])
    for key, label in [(best_key, 'best'), (wide_key, 'widest tip')]:
        if label in align:
            continue
        d, prof, L = key
        p = geom(d, L, prof)
        rows_y, rows_t = [], []
        for y0 in Y0_LIST:
            r = fw.simulate(dict(p, y0=y0), verbose=False)
            rows_y.append(dict(y0=y0, eta=r['eta']))
        for th in TH_LIST:
            r = fw.simulate(dict(p, theta=th), verbose=False)
            rows_t.append(dict(theta=th, eta=r['eta']))
        align[label] = dict(key=list(key), offset=rows_y, tilt=rows_t)
        ny = np.array([q['eta'] for q in rows_y]) / rows_y[0]['eta']
        nt = np.array([q['eta'] for q in rows_t]) / rows_t[0]['eta']
        print(f"  {label} (tip {d/nm:.0f} nm, {prof}, L = {L/um:.0f} um): 1-dB lateral offset = "
              f"{crossing(Y0_LIST, ny, ONE_DB)/um:.3f} um, 1-dB tilt = {crossing(TH_LIST, nt, ONE_DB)/deg:.1f} deg", flush=True)

    ck['align'] = align
    save_ck()

    # Stage 3: field plot of the best design
    print("\nStage 3: field plot of the best design", flush=True)
    r_opt = fw.simulate(geom(d_best, L_best, prof_best), verbose=True)
    fw.make_plots(r_opt, OUT_FIELD, show=False)
    plt.close('all')

    # Stage 4: analytics and the 2D -> 3D estimate
    print("\nStage 4: analytics / 2D -> 3D estimate", flush=True)
    tips_fine = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 160, 220]) * nm
    tip_info = {}
    for d in tips_fine:
        m = local_mode(d)
        eta_z, n_lat, mz = vertical_overlap(d)
        z_fine = np.linspace(0, 0.8, 81) * um
        eta_z_curve = [vertical_overlap(d, z0)[0] for z0 in z_fine]
        tip_info[float(d)] = dict(n_eff_lat=m.n_eff, w_1e_lat=m.w_1e, eta_facet0=facet_efficiency(d, 0.0),
                                  L_ad=adiabatic_length(d), n_lat=n_lat, w_1e_vert=mz.w_1e, eta_z=eta_z,
                                  z_1dB=crossing(z_fine, np.array(eta_z_curve) / eta_z, ONE_DB))
        print(f"  tip {d/nm:4.0f} nm: lateral n_eff = {m.n_eff:.3f}, w_1e = {m.w_1e/um:.2f} um, facet overlap = {tip_info[float(d)]['eta_facet0']:.3f}, "
              f"L_ad = {tip_info[float(d)]['L_ad']/um:5.1f} um | vertical (EIM): core n = {n_lat:.3f}, w_1e = {mz.w_1e/um:.2f} um, "
              f"eta_z = {eta_z:.3f}, 1-dB vertical offset = {tip_info[float(d)]['z_1dB']/um:.2f} um", flush=True)

    # butt-coupling reference (no taper) from the taper study, gap 0
    ref2d = prev['reference'][0]['eta'] if prev is not None else None
    eta_z_bulk = vertical_overlap(base['d_core'])[0]

    design = {}
    for d in TIPS:
        keys = [k for k in S if k[0] == float(d)]
        kb = max(keys, key=lambda k: S[k]['eta'])
        eta2d = S[kb]['eta']
        eta_z = tip_info[float(d)]['eta_z']
        design[float(d)] = dict(profile=kb[1], L=kb[2], eta_2d=eta2d, IL_2d=-10*np.log10(eta2d), eta_z=eta_z,
                                eta_3d=eta2d * eta_z, IL_3d=-10*np.log10(eta2d * eta_z))
    data = dict(tips=list(TIPS), lengths_ad=list(LENGTHS_AD), L_lin=L_LIN, y0_list=list(Y0_LIST), theta_list=list(TH_LIST),
                sweep={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in S.items()},
                best=dict(d_tip=d_best, profile=prof_best, L=L_best, **S[best_key]),
                align=align, tip_info={str(k): v for k, v in tip_info.items()}, design={str(k): v for k, v in design.items()},
                butt=dict(eta_2d=ref2d, eta_z=eta_z_bulk, eta_3d=(ref2d * eta_z_bulk) if ref2d else None),
                minutes=(time.perf_counter() - T0) / 60)
    json.dump(data, open(OUT_JSON, 'w'), indent=1, default=float)
    return data


# ----------------------------------------------------------------------------- summary + figure
def summary(data):
    b = data['best']
    print("\n=== Final design (w0 = 1 um, 220 nm slab, gap = 0) ===")
    print(f"  tip {b['d_tip']/nm:.0f} nm, {b['profile']} taper, L = {b['L']/um:.0f} um: "
          f"2D eta = {b['eta']:.4f} (IL {b['IL']:.2f} dB)")
    print("  tip   profile   L(um)   eta_2D   IL_2D(dB)   eta_z(EIM)   eta_3D est.   IL_3D est.(dB)")
    for k, v in sorted(data['design'].items(), key=lambda kv: float(kv[0])):
        print(f"  {float(k)/nm:4.0f}  {v['profile']:9s} {v['L']/um:5.0f}   {v['eta_2d']:.3f}    {v['IL_2d']:5.2f}       "
              f"{v['eta_z']:.3f}        {v['eta_3d']:.3f}          {v['IL_3d']:5.2f}")
    bt = data['butt']
    if bt['eta_2d']:
        print(f"  butt (no taper):        2D eta = {bt['eta_2d']:.3f} ({-10*np.log10(bt['eta_2d']):.2f} dB), "
              f"eta_z = {bt['eta_z']:.3f}, 3D est. = {bt['eta_3d']:.3f} ({-10*np.log10(bt['eta_3d']):.2f} dB)")
    for label, a in data['align'].items():
        d, prof, L = a['key']
        ny = np.array([q['eta'] for q in a['offset']]) / a['offset'][0]['eta']
        nt = np.array([q['eta'] for q in a['tilt']]) / a['tilt'][0]['eta']
        y0s = np.array([q['y0'] for q in a['offset']]); ths = np.array([q['theta'] for q in a['tilt']])
        ti = nearest(data['tip_info'], d)
        print(f"  alignment, {label} (tip {d/nm:.0f} nm): 1-dB lateral {crossing(y0s, ny, ONE_DB)/um:.2f} um, "
              f"1-dB vertical (EIM) {ti['z_1dB']/um:.2f} um, 1-dB tilt {crossing(ths, nt, ONE_DB)/deg:.0f} deg")
    if 'minutes' in data:
        print(f"Total simulation time: {data['minutes']:.1f} min")


def make_figure(data):
    tips = np.array(data['tips'])
    S = {tuple([float(a), b, float(c)]) for a, b, c in (k.split('|') for k in data['sweep'])}
    sweep = {(float(a), b, float(c)): v for (a, b, c), v in ((k.split('|'), v) for k, v in data['sweep'].items())}
    tip_info = {float(k): v for k, v in data['tip_info'].items()}
    design = {float(k): v for k, v in data['design'].items()}
    b = data['best']
    tips_fine = np.array(sorted(tip_info))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # (a) tip width x profile
    ax = axes[0, 0]
    ax.plot(tips_fine/nm, [tip_info[d]['eta_facet0'] for d in tips_fine], '-', color=C_AN, lw=2,
            label='analytic facet limit (lossless taper)')
    for L, mk in zip(data['lengths_ad'], ['^', 's', 'D']):
        ax.plot(tips/nm, [sweep[(float(d), 'adiabatic', float(L))]['eta'] for d in tips], mk + '-', color=C_SIM,
                lw=1.2, ms=6, label=f'FDFD adiabatic profile, L = {L/um:.0f} µm', alpha=0.55 + 0.15 * list(data['lengths_ad']).index(L))
    ax.plot(tips/nm, [sweep[(float(d), 'linear', float(data['L_lin']))]['eta'] for d in tips], 'o--', color=C_SHEET,
            lw=1.2, ms=6, label=f"FDFD linear profile, L = {data['L_lin']/um:.0f} µm")
    ax.set_xlabel('tip width $d_{tip}$ (nm)'); ax.set_ylabel('coupling efficiency η (2D)')
    ax.set_title('(a) Tip width and taper profile (gap = 0)', fontsize=11)
    ax.set_xlim(0, 230); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    # (b) taper length per tip (adiabatic)
    ax = axes[0, 1]
    for d, col in zip(tips, BLUES):
        Ls = sorted(L for (dd, pr, L) in sweep if dd == float(d) and pr == 'adiabatic')
        ax.plot([L/um for L in Ls], [sweep[(float(d), 'adiabatic', L)]['eta'] for L in Ls], 'o-', color=col, lw=1.5, ms=6,
                label=f'tip {d/nm:.0f} nm  ($L_{{ad}}$ = {tip_info[float(d)]["L_ad"]/um:.1f} µm)')
        ax.axhline(tip_info[float(d)]['eta_facet0'], color=col, lw=0.8, ls=':')
    ax.set_xlabel('taper length $L_{taper}$ (µm)'); ax.set_ylabel('coupling efficiency η (2D)')
    ax.set_title('(b) Taper length, adiabatic profile (dotted: facet limit)', fontsize=11)
    ax.set_xlim(0, 16); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.25)

    # (c) 2D -> 3D estimate
    ax = axes[0, 2]
    xs = np.arange(len(tips) + 1)
    labels = [f'{d/nm:.0f} nm' for d in tips] + ['no taper']
    e2 = [design[float(d)]['eta_2d'] for d in tips] + [data['butt']['eta_2d'] or 0]
    ez = [design[float(d)]['eta_z'] for d in tips] + [data['butt']['eta_z']]
    e3 = [design[float(d)]['eta_3d'] for d in tips] + [data['butt']['eta_3d'] or 0]
    w = 0.27
    ax.bar(xs - w, e2, w, color=C_SIM, label='η lateral (2D FDFD, best taper)')
    ax.bar(xs, ez, w, color=C_AN, label='η vertical (EIM tip mode × beam)')
    ax.bar(xs + w, e3, w, color=C_SHEET, label='η 3D estimate = product')
    for x, v in zip(xs, e3):
        ax.text(x + w, v + 0.02, f'{-10*np.log10(max(v,1e-9)):.1f} dB', ha='center', fontsize=8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel('coupling efficiency'); ax.set_ylim(0, 1.05)
    ax.set_title('(c) Separable 2D → 3D estimate (220 nm thick wire)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, axis='y', alpha=0.25)

    # (d) lateral offset, (f) tilt from FDFD; (e) vertical offset from the EIM vertical mode
    ax = axes[1, 0]
    for (label, a), col in zip(data['align'].items(), (C_SIM, C_SHEET)):
        d = a['key'][0]
        y0s = np.array([q['y0'] for q in a['offset']]); ny = np.array([q['eta'] for q in a['offset']]) / a['offset'][0]['eta']
        ax.plot(y0s/um, ny, 'o-', color=col, lw=1.5, ms=5, label=f'FDFD, tip {d/nm:.0f} nm ({label})')
        m = local_mode(d)
        yf = np.linspace(0, 0.6, 100) * um
        ax.plot(yf/um, [overlap(np.exp(-(y_grid - s)**2 / base['w0']**2), m.E_mode, y_grid) / tip_info[float(d)]['eta_facet0'] for s in yf],
                '--', color=col, lw=1.0, label=f'overlap with tip mode, {d/nm:.0f} nm')
    ax.axhline(ONE_DB, color=C_REF, lw=0.8); ax.text(0.59, ONE_DB + 0.015, '−1 dB', ha='right', fontsize=8, color=INK2)
    ax.set_xlabel('lateral offset $y_0$ (µm)'); ax.set_ylabel('η / η(0)')
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1.05); ax.set_title('(d) Lateral (in-plane) misalignment', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    zf = np.linspace(0, 0.8, 81) * um
    for (label, a), col in zip(data['align'].items(), (C_SIM, C_SHEET)):
        d = a['key'][0]
        ez0 = vertical_overlap(d)[0]
        ax.plot(zf/um, [vertical_overlap(d, z)[0] / ez0 for z in zf], '-', color=col, lw=1.5,
                label=f'EIM vertical tip mode, tip {d/nm:.0f} nm ({label})')
    ez0 = vertical_overlap(base['d_core'])[0]
    ax.plot(zf/um, [vertical_overlap(base['d_core'], z)[0] / ez0 for z in zf], '--', color=C_REF, lw=1.2, label='no taper (220 nm)')
    ax.axhline(ONE_DB, color=C_REF, lw=0.8); ax.text(0.79, ONE_DB + 0.015, '−1 dB', ha='right', fontsize=8, color=INK2)
    ax.set_xlabel('vertical offset $z_0$ (µm)'); ax.set_ylabel('η / η(0)')
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 1.05); ax.set_title('(e) Vertical misalignment (analytic, EIM)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    ax = axes[1, 2]
    for (label, a), col in zip(data['align'].items(), (C_SIM, C_SHEET)):
        d = a['key'][0]
        ths = np.array([q['theta'] for q in a['tilt']]); nt = np.array([q['eta'] for q in a['tilt']]) / a['tilt'][0]['eta']
        ax.plot(ths/deg, nt, 'o-', color=col, lw=1.5, ms=5, label=f'FDFD, tip {d/nm:.0f} nm ({label})')
    ax.axhline(ONE_DB, color=C_REF, lw=0.8); ax.text(39, ONE_DB + 0.015, '−1 dB', ha='right', fontsize=8, color=INK2)
    ax.set_xlabel('in-plane tilt θ (deg)'); ax.set_ylabel('η / η(0)')
    ax.set_xlim(0, 40); ax.set_ylim(0, 1.05); ax.set_title('(f) Angular misalignment (in-plane)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.suptitle(f"Inverse-taper edge coupler design, 1 µm beam → 220 nm Si slab, gap = 0:  best = tip {b['d_tip']/nm:.0f} nm, "
                 f"{b['profile']} taper, L = {b['L']/um:.0f} µm (2D IL {b['IL']:.2f} dB)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
