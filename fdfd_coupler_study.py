"""
fdfd_coupler_study.py -- parameter study for the Gaussian-beam -> Si-slab coupler
defined in fdfd_waveguide.py.

For every point three numbers are compared:
  * FDFD            eta = P_out / P_in   (Poynting input power, guided-mode output power)
  * analytical      butt-coupling overlap |<E_src|E_mode>|^2 / (<E_src|E_src><E_mode|E_mode>)
  * soft-source     1/beta-weighted overlap (what a current-sheet source is expected to give)

Sweeps
  1. beam waist w0           -> optimum waist vs. the mode size
  2. lateral offset y0       -> alignment tolerance, at the optimum waist and at w0 = 1 um
  3. tilt angle theta        -> angular tolerance, at the optimum waist and at w0 = 1 um
  4. core thickness d_core   -> mode-size (inverse-taper) effect for the 1 um beam
  5. grid convergence        -> dx = 10 nm vs 20 nm for the default case
Sweeps 1-3 reuse one LU factorization (only the right-hand side changes), so
they cost ~0.1-0.2 s per point; 4 and 5 re-factorize.

Output: C:\\Users\\user\\fdfd_coupler_study.png and a printed summary.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import os
from fdfd_waveguide import params, simulate, RESULTS

OUT_PNG = os.path.join(RESULTS, 'fdfd_beam_parameter_study.png')
um  = 1e-6
deg = np.pi / 180

# Palette: fixed roles (blue = FDFD, orange = analytical overlap, aqua = soft-source
# estimate, gray = reference formulas); never re-assigned between panels.
C_SIM, C_AN, C_SHEET, C_REF = '#2a78d6', '#eb6834', '#1baf7a', '#8a8883'
ONE_DB = 10**(-0.1)   # eta/eta0 at 1 dB excess loss

# Wider transverse domain than the default script so that waists up to 1.5 um
# and tilted beams stay clear of the transverse PML.
base = dict(params)
base['Ly'] = 6e-6

T0 = time.perf_counter()


def run(**kw):
    p = dict(base)
    p.update(kw)
    return simulate(p, verbose=False)


def overlap(E_src, E_mode, y):
    num = abs(integrate.trapezoid(E_src * np.conj(E_mode), y))**2
    den = integrate.trapezoid(abs(E_src)**2, y) * integrate.trapezoid(abs(E_mode)**2, y)
    return num / den


def soft_source(eta_an, n_eff, n_clad, theta=0.0):
    n_rad = n_clad * np.cos(theta)
    return (eta_an / n_eff) / (eta_an / n_eff + (1 - eta_an) / n_rad)


def crossing(xs, ys, level):
    """First x at which the (decreasing) curve ys drops below level; NaN if never."""
    below = np.where(ys < level)[0]
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return xs[0]
    return xs[k-1] + (level - ys[k-1]) * (xs[k] - xs[k-1]) / (ys[k] - ys[k-1])


# =============================================================================
# Sweep 1: beam waist
# =============================================================================
print("Sweep 1: beam waist w0  (y0 = 0, theta = 0)")
w0_list = np.array([0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30,
                    0.35, 0.40, 0.50, 0.60, 0.80, 1.00, 1.25, 1.50]) * um
S1 = []
for w0 in w0_list:
    r = run(w0=w0)
    S1.append((r['eta'], r['eta_analytic'], r['eta_sheet'], r['swr']))
    print(f"  w0 = {w0/um:5.3f} um: eta_FDFD = {r['eta']:.4f}   overlap = {r['eta_analytic']:.4f}"
          f"   soft-source = {r['eta_sheet']:.4f}   |B|/|F| = {r['swr']:.4f}")
S1 = np.array(S1)
mode, g = r['mode'], r['g']
print(f"  mode: n_eff = {mode.n_eff:.4f}, 1/e half-width = {mode.w_1e/um:.3f} um, "
      f"2nd-moment radius w_m = {mode.w_2m/um:.3f} um")

# Fine analytical curves (no simulation needed) and refined simulated optimum
w_fine   = np.linspace(0.05, 1.6, 400) * um
ov_fine  = np.array([overlap(np.exp(-g.y**2 / w**2), mode.E_mode, g.y) for w in w_fine])
ss_fine  = soft_source(ov_fine, mode.n_eff, g.n_clad)
i_best   = int(S1[:, 0].argmax())
w_ref    = np.linspace(w0_list[max(i_best-1, 0)], w0_list[min(i_best+1, len(w0_list)-1)], 11)
eta_ref  = np.array([run(w0=w)['eta'] for w in w_ref])
w0_opt_sim, eta_opt_sim = w_ref[eta_ref.argmax()], eta_ref.max()
w0_opt_an,  eta_opt_an  = w_fine[ov_fine.argmax()], ov_fine.max()
w0_opt_ss,  eta_opt_ss  = w_fine[ss_fine.argmax()], ss_fine.max()
print(f"  optimum waist: FDFD {w0_opt_sim/um:.3f} um (eta = {eta_opt_sim:.3f});  "
      f"overlap {w0_opt_an/um:.3f} um (eta = {eta_opt_an:.3f});  soft-source {w0_opt_ss/um:.3f} um (eta = {eta_opt_ss:.3f})")

# =============================================================================
# Sweep 2: lateral offset
# =============================================================================
print("\nSweep 2: lateral offset y0")
y0_list = np.arange(0, 0.501, 0.05) * um
y0_fine = np.linspace(0, 0.5, 200) * um
waists  = [(w0_opt_sim, 'optimum'), (1.0 * um, '1.00 um')]
S2 = {}
for w0, lab in waists:
    rows = []
    for y0 in y0_list:
        r = run(w0=w0, y0=y0)
        rows.append((r['eta'], r['eta_analytic'], r['eta_sheet']))
    S2[lab] = np.array(rows)
    norm = S2[lab] / S2[lab][0]
    gauss = np.exp(-2 * y0_list**2 / (w0**2 + mode.w_2m**2))
    print(f"  w0 = {w0/um:.3f} um ({lab}):  1-dB offset  FDFD {crossing(y0_list, norm[:,0], ONE_DB)/um:.3f} um,"
          f"  overlap {crossing(y0_list, norm[:,1], ONE_DB)/um:.3f} um,"
          f"  soft-source {crossing(y0_list, norm[:,2], ONE_DB)/um:.3f} um,"
          f"  Gaussian formula {crossing(y0_list, gauss, ONE_DB)/um:.3f} um")

# =============================================================================
# Sweep 3: tilt angle
# =============================================================================
print("\nSweep 3: tilt angle theta")
th_list = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50]) * deg
th_fine = np.linspace(0, 50, 200) * deg
S3 = {}
for w0, lab in waists:
    rows = []
    for th in th_list:
        r = run(w0=w0, theta=th)
        rows.append((r['eta'], r['eta_analytic'], r['eta_sheet']))
    S3[lab] = np.array(rows)
    norm = S3[lab] / S3[lab][0]
    q = g.k0 * g.n_clad * np.sin(th_list)
    gauss = np.exp(-q**2 * w0**2 * mode.w_2m**2 / (2 * (w0**2 + mode.w_2m**2)))
    print(f"  w0 = {w0/um:.3f} um ({lab}):  1-dB angle  FDFD {crossing(th_list, norm[:,0], ONE_DB)/deg:.1f} deg,"
          f"  overlap {crossing(th_list, norm[:,1], ONE_DB)/deg:.1f} deg,"
          f"  soft-source {crossing(th_list, norm[:,2], ONE_DB)/deg:.1f} deg,"
          f"  Gaussian formula {crossing(th_list, gauss, ONE_DB)/deg:.1f} deg")

# =============================================================================
# Sweep 4: core thickness (new factorization per point)
# =============================================================================
print("\nSweep 4: core thickness d_core  (w0 = 1.00 um)")
d_list = np.array([0.10, 0.14, 0.18, 0.22, 0.26, 0.30]) * um
S4 = []
for d in d_list:
    r = run(d_core=d, w0=1.0 * um)
    S4.append((r['eta'], r['eta_analytic'], r['mode'].n_eff, r['mode'].w_2m, r['swr']))
    print(f"  d = {d/um:.2f} um: n_eff = {r['mode'].n_eff:.4f}  w_m = {r['mode'].w_2m/um:.3f} um  "
          f"eta_FDFD = {r['eta']:.4f}  overlap = {r['eta_analytic']:.4f}  |B|/|F| = {r['swr']:.4f}")
S4 = np.array(S4)

# =============================================================================
# Sweep 5: grid convergence for the default case (script's own domain)
# =============================================================================
print("\nSweep 5: grid convergence (default case, Ly = 4 um, w0 = 1 um); dx refined along propagation")
S5 = []
for dx in (20e-9, 10e-9):
    p = dict(params)
    p['dx'] = dx
    t = time.perf_counter()
    r = simulate(p, verbose=False)
    S5.append((dx, r['eta'], r['eta_analytic'], r['n_eff_sim'], r['mode'].n_eff, r['swr']))
    print(f"  dx = {dx*1e9:4.0f} nm: eta_FDFD = {r['eta']:.4f}  overlap = {r['eta_analytic']:.4f}  "
          f"n_eff(discrete) = {r['n_eff_sim']:.4f} vs {r['mode'].n_eff:.4f}  |B|/|F| = {r['swr']:.4f}  "
          f"[{r['g'].N_total} unknowns, {time.perf_counter()-t:.1f} s]")

# =============================================================================
# Figure
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

# (a) waist
ax = axes[0, 0]
ax.plot(w_fine/um, ov_fine, color=C_AN, lw=2, label='analytical overlap')
ax.plot(w_fine/um, ss_fine, color=C_SHEET, lw=2, ls='--', label='soft-source estimate (1/β-weighted)')
ax.plot(w0_list/um, S1[:, 0], 'o', color=C_SIM, ms=6, label='FDFD  $P_{out}/P_{in}$')
ax.plot(w_ref/um, eta_ref, '.', color=C_SIM, ms=4)
ax.axvline(mode.w_2m/um, color=C_REF, lw=1, ls=':')
ax.text(mode.w_2m/um + 0.02, 0.03, f'mode radius $w_m$ = {mode.w_2m/um:.2f} µm', rotation=90,
        va='bottom', ha='left', fontsize=8.5, color='#52514e')
ax.annotate(f'FDFD optimum: $w_0$ = {w0_opt_sim/um:.2f} µm, η = {eta_opt_sim:.3f}\n'
            f'overlap optimum: $w_0$ = {w0_opt_an/um:.2f} µm, η = {eta_opt_an:.3f}',
            xy=(w0_opt_sim/um, eta_opt_sim), xytext=(0.30, 0.62), textcoords='axes fraction',
            fontsize=8.5, color='#52514e', arrowprops=dict(arrowstyle='-', color=C_REF, lw=0.8))
ax.set_xlabel('beam waist $w_0$ (µm)')
ax.set_ylabel('coupling efficiency η')
ax.set_xlim(0, 1.6); ax.set_ylim(0, 1)
ax.set_title('(a) Waist optimisation  ($y_0$ = 0, θ = 0)', fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, alpha=0.25)

# (b) offset
ax = axes[0, 1]
for (w0, lab), col in zip(waists, (C_SIM, C_AN)):
    S = S2[lab]
    name = f'$w_0$ = {w0/um:.2f} µm'
    ax.plot(y0_list/um, S[:, 0]/S[0, 0], 'o-', color=col, lw=1.5, ms=5, label=f'FDFD, {name}')
    ax.plot(y0_list/um, S[:, 1]/S[0, 1], '--', color=col, lw=1.5, label=f'overlap, {name}')
    ax.plot(y0_list/um, S[:, 2]/S[0, 2], '-.', color=col, lw=1.0, label=f'soft-source est., {name}')
    ax.plot(y0_fine/um, np.exp(-2*y0_fine**2/(w0**2 + mode.w_2m**2)), ':', color=C_REF, lw=1.3,
            label='Gaussian–Gaussian formula' if col == C_SIM else None)
ax.axhline(ONE_DB, color=C_REF, lw=0.8)
ax.text(0.49, ONE_DB + 0.015, '−1 dB', ha='right', fontsize=8, color='#52514e')
ax.set_xlabel('lateral offset $y_0$ (µm)')
ax.set_ylabel('η / η(0)')
ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.05)
ax.set_title('(b) Lateral misalignment tolerance', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.25)

# (c) tilt
ax = axes[1, 0]
for (w0, lab), col in zip(waists, (C_SIM, C_AN)):
    S = S3[lab]
    name = f'$w_0$ = {w0/um:.2f} µm'
    ax.plot(th_list/deg, S[:, 0]/S[0, 0], 'o-', color=col, lw=1.5, ms=5, label=f'FDFD, {name}')
    ax.plot(th_list/deg, S[:, 1]/S[0, 1], '--', color=col, lw=1.5, label=f'overlap, {name}')
    ax.plot(th_list/deg, S[:, 2]/S[0, 2], '-.', color=col, lw=1.0, label=f'soft-source est., {name}')
    q = g.k0 * g.n_clad * np.sin(th_fine)
    ax.plot(th_fine/deg, np.exp(-q**2 * w0**2 * mode.w_2m**2 / (2*(w0**2 + mode.w_2m**2))), ':',
            color=C_REF, lw=1.3, label='Gaussian–Gaussian formula' if col == C_SIM else None)
ax.axhline(ONE_DB, color=C_REF, lw=0.8)
ax.text(49, ONE_DB + 0.015, '−1 dB', ha='right', fontsize=8, color='#52514e')
ax.set_xlabel('tilt angle θ (deg)')
ax.set_ylabel('η / η(0)')
ax.set_xlim(0, 50); ax.set_ylim(0, 1.05)
ax.set_title('(c) Angular misalignment tolerance', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.25)

# (d) core thickness
ax = axes[1, 1]
ax.plot(d_list/um, S4[:, 1], 's--', color=C_AN, lw=1.5, ms=5, label='analytical overlap')
ax.plot(d_list/um, S4[:, 0], 'o-', color=C_SIM, lw=1.5, ms=6, label='FDFD  $P_{out}/P_{in}$')
for d, eta, w2m, neff in zip(d_list, S4[:, 0], S4[:, 3], S4[:, 2]):
    ax.annotate(f'$w_m$={w2m/um:.2f} µm\n$n_{{eff}}$={neff:.2f}', xy=(d/um, eta), xytext=(0, -28),
                textcoords='offset points', ha='center', fontsize=7.5, color='#52514e')
ax.set_xlabel('core thickness $d_{core}$ (µm)')
ax.set_ylabel('coupling efficiency η')
ax.set_xlim(0.08, 0.32); ax.set_ylim(0, max(0.6, S4[:, 1].max() + 0.1))
ax.set_title('(d) Core thickness at $w_0$ = 1.00 µm  (mode-size effect)', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.25)

fig.suptitle('Gaussian beam → 220 nm Si slab coupler: FDFD vs. analytical expectations', fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT_PNG, dpi=150)
print(f"\nFigure saved to {OUT_PNG}")
print(f"Total study time: {time.perf_counter() - T0:.1f} s")
plt.show()
