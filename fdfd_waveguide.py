
"""
2D FDFD simulation: Gaussian beam coupling into a slab waveguide (TE polarization).
Solves the scalar Helmholtz equation with PML boundary conditions.

Run the file directly for the default case (prints the power budget and saves
the plots).  For parameter studies import it and call simulate(params_dict):
the sparse LU factorization is cached and reused automatically as long as only
the source parameters (w0, y0, theta) change -- see fdfd_coupler_study.py.

Time convention: exp(+j*omega*t).  The PML stretch s = 1 - j*sigma/(omega*eps0)
and the source phase exp(-j*k*y*sin(theta)) both imply this convention, so a
+x-propagating wave is exp(-j*beta*x) and the time-averaged Poynting flux is
    S_x = -1/(2*omega*mu0) * Im( conj(Ez) * dEz/dx )
(positive for power flowing in the +x direction).
"""

import os
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.integrate as integrate
from scipy.constants import mu_0 as mu0
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# =============================================================================
# USER-DEFINED PARAMETERS
# =============================================================================
params = {
    # Wavelength
    'lambda0': 1.55e-6,          # m

    # Refractive indices
    'n_core': 3.48,              # Silicon core
    'n_clad': 1.44,              # SiO2 cladding

    # Waveguide geometry
    'd_core': 0.22e-6,           # Core thickness (220 nm)

    # Domain size
    'Lx': 10e-6,                 # Domain length (propagation direction)
    'Ly': 4e-6,                  # Domain width (transverse direction)

    # Grid resolution (will be checked against lambda/10/n_max)
    'dx': 0.02e-6,               # 20 nm
    'dy': 0.02e-6,               # 20 nm

    # Gaussian beam
    'w0': 1.0e-6,                # Beam waist (1/e field radius)
    'y0': 0.0,                   # Beam center (y)
    'theta': 0.0,                # Injection angle in the x-y plane (radians)

    # PML
    'L_pml': 0.5e-6,             # PML thickness
    'sigma_max': 1e5,            # Max PML conductivity [S/m].  sigma_max/(omega*eps0) ~ 9 here,
                                 # i.e. round-trip reflection ~1e-6 for this thickness/grading.
                                 # (1e13 gave a stretch of ~1e9: the PML cells decouple from the
                                 # interior and the PML edge becomes a reflecting wall.)
    'pml_order': 3,              # Polynomial grading order

    # --- Optional extras (defaults reproduce the plain butt-coupling case) ---
    'y_offset': 0.0,             # transverse grid offset in cells: 0 -> node at y=0 (core = odd number of
                                 # nodes); 0.5 -> no node at y=0 (core = even number of nodes, so widths
                                 # are even multiples of dy)
    'x_src': None,               # source plane position [m]; None -> first cell inside the left PML
    # Inverse-taper edge coupler: pure cladding for x < x_facet, core width rising
    # linearly from d_tip at x_facet to d_core at x_facet + L_taper, then uniform.
    'x_facet': None,             # None -> uniform waveguide over the whole domain (no taper)
    'd_tip': None,
    'L_taper': None,
    'taper_profile': 'linear',   # 'linear' width ramp, or 'adiabatic': width law from the Love
                                 # delineation criterion d(rho)/dz = C*rho*(beta(rho) - k0*n_clad),
                                 # scaled to total length L_taper (slow at the tip, fast later)
}

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS, exist_ok=True)
OUTPUT_FILE = os.path.join(RESULTS, 'fdfd_waveguide.png')


# =============================================================================
# SETUP: derived constants, grid, refractive index, PML stretch factors
# =============================================================================
def sigma_profile(coords, n_pml, d_cell, order, s_max):
    """Polynomial-graded PML conductivity along a 1D array of coordinates."""
    N = len(coords)
    sig = np.zeros(N)
    # Left PML
    for i in range(n_pml):
        depth = (n_pml - i) / n_pml
        sig[i] = s_max * depth**order
    # Right PML
    for i in range(n_pml):
        depth = (i + 1) / n_pml
        sig[N - n_pml + i] = s_max * depth**order
    return sig


def s_factor(sig, omega, eps0):
    """Complex coordinate stretching: s = 1 - j*sigma/(omega*eps0)."""
    return 1.0 - 1j * sig / (omega * eps0)


def slab_n_eff(d, k0, n_core, n_clad):
    """n_eff of the fundamental TE mode of a symmetric slab of width d
    (characteristic equation ky*tan(ky*d/2) = gamma; root below the first tan pole)."""
    NA2 = n_core**2 - n_clad**2

    def f(ky):
        return ky * np.tan(ky * d / 2) - np.sqrt(max(NA2 * k0**2 - ky**2, 0.0))

    hi = min(np.sqrt(NA2) * k0, np.pi / d) * 0.9999
    return np.sqrt(n_core**2 - (brentq(f, 1e3, hi) / k0)**2)


def setup(p):
    """Derived constants, grid, refractive-index map and PML stretch factors.
    Returns a namespace g that the other building blocks take as input."""
    g = SimpleNamespace(**p)
    g.k0    = 2 * np.pi / g.lambda0
    g.omega = 3e8 * g.k0          # angular frequency
    g.eps0  = 8.854e-12

    # Check resolution
    n_max  = max(g.n_core, g.n_clad)
    dx_max = g.lambda0 / (10 * n_max)
    assert g.dx <= dx_max, f"dx={g.dx*1e9:.1f}nm exceeds lambda/(10*n_max)={dx_max*1e9:.1f}nm"
    assert g.dy <= dx_max, f"dy={g.dy*1e9:.1f}nm exceeds lambda/(10*n_max)={dx_max*1e9:.1f}nm"

    # Grid
    g.Nx = int(round(g.Lx / g.dx))
    g.Ny = int(round(g.Ly / g.dy))
    g.N_total = g.Nx * g.Ny
    g.x  = np.arange(g.Nx) * g.dx
    g.y  = (np.arange(g.Ny) + g.y_offset) * g.dy - g.Ly / 2   # centered at y=0

    # Refractive index distribution.  d_x is the local core width along x:
    # uniform d_core, or (inverse taper) 0 before the facet, linear ramp
    # d_tip -> d_core over L_taper, then d_core.
    if g.x_facet is None:
        g.d_x = np.full(g.Nx, g.d_core)
    elif g.taper_profile == 'linear':
        ramp  = np.clip((g.x - g.x_facet) / g.L_taper, 0.0, 1.0)
        g.d_x = np.where(g.x < g.x_facet, 0.0, g.d_tip + (g.d_core - g.d_tip) * ramp)
    elif g.taper_profile == 'adiabatic':
        # Love et al. (1991) delineation criterion, d rho/dz = C rho (beta_1 - beta_2)
        # with beta_2 = k0 n_clad, integrated into z(rho) and scaled to L_taper.
        rho_t = np.linspace(g.d_tip / 2, g.d_core / 2, 400)
        dbeta = np.array([g.k0 * (slab_n_eff(2 * r, g.k0, g.n_core, g.n_clad) - g.n_clad) for r in rho_t])
        z_t = integrate.cumulative_trapezoid(1.0 / (rho_t * dbeta), rho_t, initial=0.0)
        z_t *= g.L_taper / z_t[-1]
        rho_x = np.interp(g.x - g.x_facet, z_t, rho_t)      # clamps to rho_tip / rho_bulk outside
        g.d_x = np.where(g.x < g.x_facet, 0.0, 2 * rho_x)
    else:
        raise ValueError(f"unknown taper_profile {g.taper_profile!r}")
    # Sub-pixel averaging: each node j owns the cell [y_j - dy/2, y_j + dy/2];
    # a cell cut by the core boundary gets eps = n^2 averaged by fill fraction.
    # For Ez (TE) the field is continuous across the boundary, so the
    # arithmetic mean of eps is the correct effective permittivity.  Cells that
    # are fully inside/outside are exact, so uniform waveguides whose edges
    # fall on cell boundaries are unchanged; tapered widths become continuous
    # instead of changing in 2*dy steps (at a 20 nm tip such a staircase
    # doubles the core in one step and causes a large spurious loss).
    half = g.d_x[:, None] / 2
    y_lo, y_hi = g.y[None, :] - g.dy / 2, g.y[None, :] + g.dy / 2
    fill = np.clip((np.minimum(y_hi, half) - np.maximum(y_lo, -half)) / g.dy, 0.0, 1.0)
    g.n2 = np.where(fill >= 1.0, g.n_core**2,
                    np.where(fill <= 0.0, g.n_clad**2,
                             g.n_clad**2 + fill * (g.n_core**2 - g.n_clad**2)))
    g.n = np.sqrt(g.n2)
    g.core_mask = (np.abs(g.y) <= g.d_core / 2)   # bulk-waveguide core (used for the output mode)

    # PML coordinate stretching factors
    g.n_pml_x = int(round(g.L_pml / g.dx))
    g.n_pml_y = int(round(g.L_pml / g.dy))
    sig_x = sigma_profile(g.x, g.n_pml_x, g.dx, g.pml_order, g.sigma_max)
    sig_y = sigma_profile(g.y, g.n_pml_y, g.dy, g.pml_order, g.sigma_max)
    g.sx = s_factor(sig_x, g.omega, g.eps0)   # shape (Nx,)
    g.sy = s_factor(sig_y, g.omega, g.eps0)   # shape (Ny,)
    # Half-cell stretched factors (average of adjacent cells)
    g.sx_half = 0.5 * (g.sx[:-1] + g.sx[1:])   # shape (Nx-1,) at i+1/2
    g.sy_half = 0.5 * (g.sy[:-1] + g.sy[1:])   # shape (Ny-1,) at j+1/2

    # PML sanity numbers.  The stencil coefficients scale as 1/(s_i*s_{i+1/2}):
    # if sigma/(omega*eps0) >> 1 the PML cells decouple from the interior
    # instead of absorbing and the PML edge acts as a reflecting (Neumann)
    # wall.  An absorbing PML needs sigma_max/(omega*eps0) of order 1-30; the
    # suggested value gives round-trip power reflection R_target for a normally
    # incident cladding wave with this thickness and grading order.
    g.s_prime_max = g.sigma_max / (g.omega * g.eps0)
    g.R_target = 1e-6
    g.sigma_max_suggested = (-np.log(g.R_target) * (g.pml_order + 1)
                             / (2 * g.k0 * g.n_clad * g.L_pml) * (g.omega * g.eps0))

    # Slices used by the source, the input-power probe and the output overlap
    if g.x_src is None:
        g.i_src = g.n_pml_x + 1      # soft source just inside the left PML
    else:
        g.i_src = int(round(g.x_src / g.dx))
        assert g.n_pml_x + 1 <= g.i_src < g.Nx - g.n_pml_x - 6, "x_src must lie between the PMLs"
    g.i_in  = g.i_src + 3            # Poynting probe for the input power
    g.i_out = g.Nx - g.n_pml_x - 2   # output cross-section, just before the right PML
    # Start of the window used to fit the guided-mode amplitude a(x) (must lie
    # in the uniform output waveguide): after the taper, or shortly after the source.
    if g.x_facet is None:
        g.i_fit = g.i_in + 40
    else:
        g.i_fit = int(np.searchsorted(g.x, g.x_facet + g.L_taper + 0.5e-6))
    assert g.i_out - g.i_fit >= 50, "need >= 50 cells of uniform output waveguide for the mode fit"
    return g


# =============================================================================
# MATRIX ASSEMBLY  A * Ez = b   (vectorized 5-point stencil)
# =============================================================================
def assemble(g):
    """Sparse CSC Helmholtz operator with PML stretching and Dirichlet frame.
    Indexing: flat index = i*Ny + j  where i in [0,Nx), j in [0,Ny)."""
    Nx, Ny, dx, dy = g.Nx, g.Ny, g.dx, g.dy
    sx, sy, sx_half, sy_half = g.sx, g.sy, g.sx_half, g.sy_half
    n2 = g.n2    # (Nx, Ny), sub-pixel averaged eps

    # x-direction second derivative with PML:  (1/sx) * d/dx [ (1/sx) * dEz/dx ]
    # Discretized as:
    #   [1/(sx_i * dx)] * [ (Ez_{i+1,j} - Ez_{i,j}) / (sx_{i+1/2} * dx)
    #                     - (Ez_{i,j} - Ez_{i-1,j}) / (sx_{i-1/2} * dx) ]
    # where the outer 1/sx factor uses sx at node i.
    # Coupling coefficients to the (i+1) / (i-1) neighbours depend on i only.
    # They are left at zero on the boundary rows i = 0 and i = Nx-1, which are
    # Dirichlet rows (see below).
    c_xp = np.zeros(Nx, dtype=complex)
    c_xm = np.zeros(Nx, dtype=complex)
    c_xp[1:Nx-1] = 1.0 / (sx[1:Nx-1] * sx_half[1:Nx-1] * dx**2)   # -> (i+1, j)
    c_xm[1:Nx-1] = 1.0 / (sx[1:Nx-1] * sx_half[0:Nx-2] * dx**2)   # -> (i-1, j)

    # y-direction: same structure, coefficients depend on j only.
    c_yp = np.zeros(Ny, dtype=complex)
    c_ym = np.zeros(Ny, dtype=complex)
    c_yp[1:Ny-1] = 1.0 / (sy[1:Ny-1] * sy_half[1:Ny-1] * dy**2)   # -> (i, j+1)
    c_ym[1:Ny-1] = 1.0 / (sy[1:Ny-1] * sy_half[0:Ny-2] * dy**2)   # -> (i, j-1)

    # Diagonal entry: -(c_xp + c_xm) - (c_yp + c_ym) + k0^2 n^2,  shape (Nx, Ny)
    diag = (-(c_xp + c_xm))[:, None] + (-(c_yp + c_ym))[None, :] + g.k0**2 * n2

    # Dirichlet boundary rows: Ez = 0 on the outer frame (absorbed by the PML
    # anyway).  Diagonal only, no off-diagonal couplings.  The diagonal is
    # scaled to 1/dx^2 (same equation, Ez = 0) so that these rows have the
    # same magnitude as the interior stencil rows; with a unit diagonal a
    # backward-stable LU only honours them to ~eps*||A||*||x||, which can leave
    # O(1) garbage on the frame.
    interior = np.zeros((Nx, Ny), dtype=bool)
    interior[1:Nx-1, 1:Ny-1] = True
    diag[~interior] = 1.0 / dx**2

    P = np.arange(g.N_total).reshape(Nx, Ny)   # P[i, j] == i*Ny + j
    p_int = P[interior]                        # flat indices of interior nodes
    ii, jj = np.nonzero(interior)              # their (i, j) grid indices

    rows = np.concatenate([p_int, p_int, p_int, p_int, P.ravel()])
    cols = np.concatenate([p_int + Ny, p_int - Ny, p_int + 1, p_int - 1, P.ravel()])
    vals = np.concatenate([c_xp[ii], c_xm[ii], c_yp[jj], c_ym[jj], diag.ravel()])

    # Single sparse call; CSC is what SuperLU wants, so no internal conversion later.
    return sp.coo_matrix((vals, (rows, cols)), shape=(g.N_total, g.N_total)).tocsc()


# =============================================================================
# SOURCE VECTOR  b  -- Gaussian beam injected at x = x_src
# =============================================================================
def make_source(g):
    """Soft (current-sheet) Gaussian source on the slice x = x[i_src].
    The tilt enters as the transverse phase exp(-j*k0*n_clad*sin(theta)*y)
    of a plane wave travelling at angle theta to the x axis."""
    y = g.y
    E_src = (np.exp(-(y - g.y0)**2 / g.w0**2)
             * np.exp(-1j * g.k0 * g.n_clad * np.sin(g.theta) * y))
    # The source enters as a current source added to b at the interior nodes of
    # the slice (the soft source radiates into both +x and -x).
    b = np.zeros(g.N_total, dtype=complex)
    j_int = np.arange(1, g.Ny - 1)
    b[g.i_src * g.Ny + j_int] = -E_src[j_int] * g.k0**2
    return b, E_src


# =============================================================================
# ANALYTICAL MODE PROFILE  E_mode(y)
# =============================================================================
def slab_mode(g):
    """Fundamental TE mode of the symmetric slab.
    Characteristic equation: ky * tan(ky * d/2) = gamma, with
    ky^2 + gamma^2 = (n_core^2 - n_clad^2) * k0^2.
    Returns ky, gamma, n_eff, beta and the profile E_mode(y) normalized so that
    trapezoid(E_mode^2, y) = 1."""
    k0, d = g.k0, g.d_core
    NA2 = g.n_core**2 - g.n_clad**2

    def char_eq(ky):
        """ky*tan(ky*d/2) - gamma, vectorized over ky (inf where gamma^2 <= 0)."""
        ky = np.asarray(ky, dtype=float)
        g2 = NA2 * k0**2 - ky**2
        gamma = np.sqrt(np.maximum(g2, 0.0))
        f = np.where(g2 <= 0, np.inf, ky * np.tan(ky * d / 2) - gamma)
        return f if f.ndim else float(f)   # brentq wants a plain float

    # Scan for the first sign change, then refine (fundamental mode)
    ky_vals = np.linspace(1e4, np.sqrt(NA2) * k0 * 0.9999, 10000)
    f_vals  = char_eq(ky_vals)
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    if len(sign_changes) == 0:
        raise RuntimeError("Could not find guided mode -- check d_core, n_core, n_clad, lambda0")
    ky = brentq(char_eq, ky_vals[sign_changes[0]], ky_vals[sign_changes[0] + 1])
    gamma = np.sqrt(NA2 * k0**2 - ky**2)

    # Profile: cos inside the core, evanescent tails outside
    half  = d / 2
    A_cos = 1.0
    B_exp = A_cos * np.cos(ky * half)   # continuity at interface
    ay = np.abs(g.y)
    E = np.where(ay <= half,
                 A_cos * np.cos(ky * g.y),
                 B_exp * np.exp(-gamma * (ay - half)))
    E = E / np.sqrt(integrate.trapezoid(E**2, g.y))

    # Effective index from the dispersion relation  beta^2 = k0^2 n_core^2 - ky^2
    #                                                      = k0^2 n_clad^2 + gamma^2
    n_eff = np.sqrt(g.n_core**2 - (ky / k0)**2)
    n_eff_check = np.sqrt(g.n_clad**2 + (gamma / k0)**2)
    assert np.isclose(n_eff, n_eff_check, rtol=1e-9), (n_eff, n_eff_check)

    # Mode size measures (for comparing with the beam waist)
    sel  = (g.y >= 0) & (E <= E.max() / np.e)
    w_1e = g.y[sel].min() if sel.any() else np.inf                # 1/e field half-width (inf: not reached on the grid)
    # 2nd-moment radius: for a Gaussian exp(-y^2/w^2), <y^2> = w^2/4  ->  w = 2*sqrt(<y^2>)
    w_2m = 2 * np.sqrt(integrate.trapezoid(g.y**2 * E**2, g.y))

    return SimpleNamespace(ky=ky, gamma=gamma, n_eff=n_eff, n_eff_check=n_eff_check,
                           beta=k0 * n_eff, E_mode=E, w_1e=w_1e, w_2m=w_2m)


# =============================================================================
# POWER BUDGET: INPUT POWER, OUTPUT MODE POWER, INSERTION LOSS
# =============================================================================
def poynting_x(g, Ez, i):
    """Time-averaged x-directed Poynting flux density S_x(y) through slice x_i
    (exp(+j*omega*t) convention, central difference for dEz/dx)."""
    dEz_dx = (Ez[i + 1, :] - Ez[i - 1, :]) / (2 * g.dx)
    return -0.5 / (g.omega * mu0) * np.imag(np.conj(Ez[i, :]) * dEz_dx)


def analyze(g, Ez, E_src, mode):
    """Input power, guided-mode output power, coupling efficiency, insertion
    loss, plus the analytical overlap and two consistency diagnostics."""
    y, E_mode, beta = g.y, mode.E_mode, mode.beta

    # Input power: Poynting probe a few cells downstream of the source.  The
    # soft source radiates symmetrically into +x and -x; this probe only sees
    # the +x half, so no extra factor of 1/2 is applied.
    P_in = integrate.trapezoid(poynting_x(g, Ez, g.i_in), y)

    # Guided-mode amplitude a(x) = <E_mode|Ez(x,.)> along the whole domain and
    # at the output slice x = Lx - L_pml (mode is unit-normalized).
    a_x    = integrate.trapezoid(Ez * np.conj(E_mode)[None, :], y, axis=1)
    E_out  = Ez[g.i_out, :]
    a_mode = a_x[g.i_out]

    # Validation 3: is the guided mode at the output a clean travelling wave?
    # Fit a(x) between source and output to  F exp(-j beta x) + B exp(+j beta x).
    # |B|/|F| ~ 0 for a travelling mode; ~1 means reflections / standing wave,
    # in which case the projected P_out also counts backward-travelling power.
    win   = slice(g.i_fit, g.i_out + 1)
    basis = np.stack([np.exp(-1j * beta * g.x[win]), np.exp(1j * beta * g.x[win])], axis=1)
    F_amp, B_amp = np.linalg.lstsq(basis, a_x[win], rcond=None)[0]
    swr = np.abs(B_amp) / np.abs(F_amp)

    # Validation 4: effective index of the *discrete* guided mode from the
    # phase slope of a(x) (meaningful when |B| << |F|); converges to the
    # analytical n_eff as the grid is refined.
    phase = np.unwrap(np.angle(a_x[win]))
    n_eff_sim = -np.polyfit(g.x[win], phase, 1)[0] / g.k0

    # Output guided-mode power.  Continuum: P = |a|^2 beta / (2 omega mu0).  On
    # the 5-point grid the exactly conserved x-flux of exp(-j beta x) is
    # Im(conj(E_i) E_{i+1})/dx = sin(beta dx)/dx rather than beta (0.9% lower
    # at beta*dx = 0.23), and the Poynting probe measures that discrete flux.
    # The same factor is used here (with the discrete beta when the mode is a
    # clean travelling wave), otherwise eta can exceed 1 by the difference.
    beta_d   = g.k0 * n_eff_sim if swr < 0.1 else beta
    flux_fac = np.sin(beta_d * g.dx) / g.dx
    P_out   = np.abs(a_mode)**2 * flux_fac / (2 * g.omega * mu0)       # guided-mode power
    P_total = integrate.trapezoid(poynting_x(g, Ez, g.i_out), y)     # all +x power at the output

    eta   = P_out / P_in
    IL_dB = -10 * np.log10(max(eta, 1e-20))

    # Validation 1: analytical butt-coupling overlap of the launched beam with
    # the guided mode.
    num = np.abs(integrate.trapezoid(E_src * np.conj(E_mode), y))**2
    den = integrate.trapezoid(np.abs(E_src)**2, y) * integrate.trapezoid(np.abs(E_mode)**2, y)
    eta_analytic   = num / den
    IL_analytic_dB = -10 * np.log10(max(eta_analytic, 1e-20))

    # Validation 2: a soft (current-sheet) source excites every mode with a
    # power proportional to |overlap|^2 / beta, so the high-beta guided mode
    # gets less of the launched power than the plain field overlap suggests.
    # Weighting the overlap by 1/beta (radiation taken at beta ~ k0*n_clad*cos(theta))
    # estimates what P_out/P_in should give for this source type.
    n_rad = g.n_clad * np.cos(g.theta)
    eta_sheet   = (eta_analytic / mode.n_eff) / (eta_analytic / mode.n_eff + (1 - eta_analytic) / n_rad)
    IL_sheet_dB = -10 * np.log10(max(eta_sheet, 1e-20))

    return dict(P_in=P_in, P_out=P_out, P_total=P_total, a_mode=a_mode, E_out=E_out,
                eta=eta, IL_dB=IL_dB,
                eta_analytic=eta_analytic, IL_analytic_dB=IL_analytic_dB,
                eta_sheet=eta_sheet, IL_sheet_dB=IL_sheet_dB, swr=swr,
                n_eff_sim=n_eff_sim, beta_d=beta_d)


# =============================================================================
# DRIVER
# =============================================================================
_SOURCE_KEYS = ('w0', 'y0', 'theta', 'x_src')   # parameters that only change b, not A
_LU_CACHE = {'key': None, 'lu': None}           # last factorization (one entry)
LU_OPTIONS = {}                                 # extra keyword arguments for scipy.sparse.linalg.splu


def _operator_key(p):
    def norm(v):
        return v if (v is None or isinstance(v, str)) else float(v)
    return tuple(sorted((k, norm(v)) for k, v in p.items() if k not in _SOURCE_KEYS))


def simulate(p=None, verbose=True):
    """Full pipeline for one parameter set.  Returns a dict with the power
    budget (P_in, P_out, P_total, eta, IL_dB, eta_analytic, eta_sheet, swr, ...)
    plus g (setup namespace), Ez, E_src and mode.  The LU factorization is
    reused when only w0 / y0 / theta differ from the previous call."""
    p = dict(params if p is None else p)
    say = print if verbose else (lambda *a, **k: None)

    g = setup(p)
    say(f"Grid: {g.Nx} x {g.Ny} = {g.N_total} unknowns")
    say(f"Domain: {g.Lx*1e6:.1f} um x {g.Ly*1e6:.1f} um")
    say(f"PML: max stretch sigma_max/(omega*eps0) = {g.s_prime_max:.2e}   "
        f"(absorbing range ~1-30; sigma_max ~ {g.sigma_max_suggested:.1e} S/m gives R_pml ~ {g.R_target:.0e})")
    if g.s_prime_max > 1e2:
        say("WARNING: PML stretch is far too large -- the PML edge acts as a reflecting wall and the\n"
            "         domain becomes a resonant cavity.  P_in, P_out and eta are then NOT meaningful.")

    key = _operator_key(p)
    if _LU_CACHE['key'] == key:
        lu = _LU_CACHE['lu']
        say("Reusing cached LU factorization (only the source changed).")
    else:
        t0 = time.perf_counter()
        A = assemble(g)
        say(f"Matrix assembled in {time.perf_counter() - t0:.2f} s  (nnz = {A.nnz})")
        say("Factorizing (sparse LU on the CSC matrix)...")
        t0 = time.perf_counter()
        lu = spla.splu(A, **LU_OPTIONS)
        say(f"Done in {time.perf_counter() - t0:.2f} s.")
        _LU_CACHE.update(key=key, lu=lu)

    b, E_src = make_source(g)
    t0 = time.perf_counter()
    Ez = lu.solve(b).reshape(g.Nx, g.Ny)
    say(f"Solve: {time.perf_counter() - t0:.2f} s.")

    mode = slab_mode(g)
    say(f"\nMode parameters:")
    say(f"  ky    = {mode.ky:.4e} rad/m")
    say(f"  gamma = {mode.gamma:.4e} rad/m")
    say(f"  n_eff = {mode.n_eff:.4f}   (check via n_clad, gamma: {mode.n_eff_check:.4f})")
    say(f"  mode 1/e half-width = {mode.w_1e*1e6:.3f} um, 2nd-moment radius = {mode.w_2m*1e6:.3f} um")

    res = analyze(g, Ez, E_src, mode)
    say(f"\n=== Results ===")
    say(f"P_in   (Poynting probe at x = {g.x[g.i_in]*1e6:.2f} um)     = {res['P_in']:.4e} W/m")
    say(f"P_out  (guided mode at x = {g.x[g.i_out]*1e6:.2f} um)       = {res['P_out']:.4e} W/m")
    say(f"P_total (Poynting flux at x = {g.x[g.i_out]*1e6:.2f} um)    = {res['P_total']:.4e} W/m   [diagnostic: P_out <= P_total <= P_in]")
    say(f"eta (simulated)  = P_out / P_in                = {res['eta']:.4f}")
    say(f"eta (analytical) = butt-coupling overlap       = {res['eta_analytic']:.4f}   (IL = {res['IL_analytic_dB']:.2f} dB)")
    say(f"eta (soft-source estimate, 1/beta-weighted)    = {res['eta_sheet']:.4f}   (IL = {res['IL_sheet_dB']:.2f} dB)")
    say(f"n_eff                                          = {mode.n_eff:.4f}")
    say(f"Insertion Loss IL = -10 log10(eta)             = {res['IL_dB']:.2f} dB")
    say(f"\nDiagnostics:")
    say(f"  guided-mode standing-wave ratio |B|/|F| at output = {res['swr']:.4f}   (<~0.05 for a clean travelling mode)")
    say(f"  n_eff of the discrete mode (phase slope of a(x))  = {res['n_eff_sim']:.4f}   (analytical {mode.n_eff:.4f})")
    if res['swr'] > 0.1:
        say("  WARNING: the guided mode is largely a standing wave (reflections from the boundaries);\n"
            "           P_out then includes backward-travelling power and eta / IL are not meaningful.")

    res.update(g=g, Ez=Ez, E_src=E_src, mode=mode, params=p)
    return res


# =============================================================================
# PLOTS
# =============================================================================
def make_plots(res, output_file=OUTPUT_FILE, show=True):
    g, Ez, mode = res['g'], res['Ez'], res['mode']
    x, y, d_core = g.x, g.y, g.d_core
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 2D intensity ---
    ax = axes[0]
    intensity = np.abs(Ez)**2
    extent = [y[0]*1e6, y[-1]*1e6, x[0]*1e6, x[-1]*1e6]
    im = ax.imshow(intensity, origin='lower', aspect='auto', extent=extent,
                   cmap='inferno')
    plt.colorbar(im, ax=ax, label=r'$|E_z|^2$')
    ax.set_xlabel('y (um)')
    ax.set_ylabel('x (um)')
    ax.set_title('2D Steady-State Intensity $|E_z(x,y)|^2$')
    # Mark core boundaries (follows the taper profile if there is one)
    edge = np.where(g.d_x > 0, g.d_x / 2, np.nan) * 1e6
    ax.plot(-edge, x*1e6, color='cyan', lw=0.8, ls='--', label='core edge')
    ax.plot( edge, x*1e6, color='cyan', lw=0.8, ls='--')
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.legend(fontsize=8)

    # --- 1D cross-section comparison ---
    ax2 = axes[1]
    E_sim_norm = np.abs(res['E_out'])
    if E_sim_norm.max() > 0:
        E_sim_norm = E_sim_norm / E_sim_norm.max()
    E_mode_norm = np.abs(mode.E_mode)
    if E_mode_norm.max() > 0:
        E_mode_norm = E_mode_norm / E_mode_norm.max()

    ax2.plot(y*1e6, E_sim_norm,  label='$|E_{sim}(y)|$ at output', lw=1.5)
    ax2.plot(y*1e6, E_mode_norm, label='$|E_{mode}(y)|$ analytical', lw=1.5, ls='--')
    ax2.axvspan(-d_core/2*1e6, d_core/2*1e6, alpha=0.15, color='green', label='core')
    ax2.set_xlabel('y (um)')
    ax2.set_ylabel('Normalized field')
    ax2.set_title(f"Field Cross-Section at Output  (IL = {res['IL_dB']:.2f} dB, $\\eta$ = {res['eta']:.3f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    results = simulate(params)
    make_plots(results, OUTPUT_FILE)
    print(f"\nPlot saved to {OUTPUT_FILE}")
