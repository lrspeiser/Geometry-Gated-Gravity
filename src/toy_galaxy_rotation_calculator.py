
import numpy as np

# ---------- Physical constants ----------
G_AST = 4.30091e-6   # kpc * (km/s)^2 / Msun
c_kms = 299_792.458  # km/s

def symmetrize_fourfold(pos):
    x, y = pos[:,0], pos[:,1]
    return np.vstack([
        np.column_stack([ x,  y]),
        np.column_stack([ x, -y]),
        np.column_stack([-x,  y]),
        np.column_stack([-x, -y]),
    ])

def sample_exponential_disk(N_base, Rd_kpc, Rmax_kpc, rng):
    R = np.zeros(N_base)
    k = 0
    while k < N_base:
        r = rng.exponential(Rd_kpc) + rng.exponential(Rd_kpc)
        if r <= Rmax_kpc:
            R[k] = r
            k += 1
    theta = rng.uniform(0.0, 2.0*np.pi, size=N_base)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    pos = np.column_stack([x, y])
    return symmetrize_fourfold(pos)

def sample_plummer_bulge(N_base, a_kpc, Rmax_kpc, rng):
    R = np.zeros(N_base)
    k = 0
    while k < N_base:
        r = rng.uniform(0.0, Rmax_kpc)
        w = r / (r*r + a_kpc*a_kpc)**1.5
        w_ref = a_kpc / (a_kpc*a_kpc + a_kpc*a_kpc)**1.5 + 1e-12
        if rng.uniform(0.0, 1.0) < (w / (w_ref + 1e-18)):
            R[k] = r
            k += 1
    theta = rng.uniform(0.0, 2.0*np.pi, size=N_base)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    pos = np.column_stack([x, y])
    return symmetrize_fourfold(pos)

def make_galaxy(config_name: str, seed: int, spread_scale: float):
    rng = np.random.default_rng(seed)
    if config_name == "MW-like disk":
        N_disk_total = 2000
        N_bulge_total = 400
        N_disk_base = N_disk_total // 4
        N_bulge_base = N_bulge_total // 4
        Rd = 3.0 * spread_scale
        Rmax = 20.0 * spread_scale
        a_bulge = 0.6 * spread_scale
        M_disk = 6.0e10
        M_bulge = 1.2e10
        pos_disk = sample_exponential_disk(N_disk_base, Rd, Rmax, rng)
        pos_bulge = sample_plummer_bulge(N_bulge_base, a_bulge, 3.0*spread_scale, rng)
        pos = np.vstack([pos_disk, pos_bulge])
        masses = np.concatenate([np.full(len(pos_disk), M_disk/N_disk_total),
                                 np.full(len(pos_bulge), M_bulge/N_bulge_total)])
    elif config_name == "Central-dominated":
        N_bulge_total = 1600
        N_outer_total = 400
        N_bulge_base = N_bulge_total // 4
        N_outer_base = N_outer_total // 4
        a_bulge = 0.8 * spread_scale
        Rmax_outer = 15.0 * spread_scale
        M_bulge = 8.0e10
        M_outer = 1.5e10
        pos_bulge = sample_plummer_bulge(N_bulge_base, a_bulge, 4.0*spread_scale, rng)
        pos_outer = sample_exponential_disk(N_outer_base, 4.0*spread_scale, Rmax_outer, rng)
        pos = np.vstack([pos_bulge, pos_outer])
        masses = np.concatenate([np.full(len(pos_bulge), M_bulge/N_bulge_total),
                                 np.full(len(pos_outer), M_outer/N_outer_total)])
    elif config_name == "Dwarf disk":
        N_disk_total = 1000
        N_disk_base = N_disk_total // 4
        Rd = 1.0 * spread_scale
        Rmax = 6.0 * spread_scale
        M_disk = 3.0e9
        pos = sample_exponential_disk(N_disk_base, Rd, Rmax, rng)
        masses = np.full(len(pos), M_disk/N_disk_total)
    else:
        raise ValueError("Unknown config")
    return pos, masses

def net_acceleration_at_point(point, pos, masses, G_eff, k_GR=0.0, atten_extra=0.0, soften_kpc=0.1):
    r_vec = pos - point
    r2 = np.sum(r_vec*r_vec, axis=1)
    eps2 = (float(soften_kpc)**2)
    inv_r3 = 1.0 / np.power(r2 + eps2, 1.5)
    newton_terms = (r_vec * (masses * inv_r3)[:, None]).sum(axis=0)
    a_newton_vec = G_eff * newton_terms
    r = np.sqrt(r2 + eps2)
    per_body_scale = (1.0 - k_GR * (G_eff / max(1e-30, G_AST)) * G_AST * masses / (r * c_kms**2))
    per_body_scale = np.clip(per_body_scale, 0.0, 1.0)
    per_body_scale *= (1.0 - np.clip(atten_extra, 0.0, 0.999))
    gr_terms = (r_vec * ((masses * inv_r3) * per_body_scale)[:, None]).sum(axis=0)
    a_gr_vec = G_eff * gr_terms
    return a_gr_vec, a_newton_vec

def circular_speed_from_accel(a_vec, R_vec_to_COM):
    R = np.linalg.norm(R_vec_to_COM)
    if R <= 0:
        return 0.0
    u_in = -R_vec_to_COM / R
    a_rad = float(np.dot(a_vec, u_in))
    a_rad = max(0.0, a_rad)
    return np.sqrt(R * a_rad)

def main():
    import argparse, os, sys
    import matplotlib

    # CLI options for reproducibility and headless usage
    parser = argparse.ArgumentParser(description="Interactive toy galaxy rotation calculator with optional headless save.")
    parser.add_argument('--backend', type=str, default=None, help='Optional Matplotlib backend override (e.g., MacOSX, TkAgg, QtAgg, Agg).')
    parser.add_argument('--save', type=str, default=None, help='If set, save a PNG to this path and exit (headless mode).')

    config_names = ["MW-like disk", "Central-dominated", "Dwarf disk"]
    parser.add_argument('--config', type=str, choices=config_names, default="MW-like disk", help='Initial configuration.')
    parser.add_argument('--seed', type=int, default=42, help='Initial RNG seed.')
    parser.add_argument('--spread', type=float, default=1.0, help='Initial spread scale.')
    parser.add_argument('--outer', type=float, default=0.95, help='Initial outer percentile (0.80–0.995).')
    parser.add_argument('--soften', type=float, default=0.2, help='Initial softening (kpc).')
    parser.add_argument('--gscale', type=float, default=1.0, help='Initial G scale.')
    parser.add_argument('--grk', type=float, default=0.0, help='Initial GR k (toy).')
    parser.add_argument('--atten', type=float, default=0.0, help='Initial extra attenuation (0–0.8).')
    parser.add_argument('--boost', type=float, default=1.0, help='Initial boost fraction.')
    parser.add_argument('--vobs', type=float, default=220.0, help='Initial observed v (km/s).')

    args = parser.parse_args()

    if args.backend:
        try:
            matplotlib.use(args.backend, force=True)
        except Exception as e:
            print(f"Warning: could not set backend {args.backend}: {e}", file=sys.stderr)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons

    def backend_is_interactive() -> bool:
        b = matplotlib.get_backend().lower()
        # Treat common non-interactive backends as headless
        non_interactive = ('agg', 'pdf', 'ps', 'svg', 'cairo', 'template')
        return not any(b.startswith(x) for x in non_interactive)

    # ----- Initialize galaxy -----
    pos, masses = make_galaxy(args.config, args.seed, args.spread)
    COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
    R_all = np.linalg.norm(pos - COM, axis=1)
    R_edge = np.quantile(R_all, float(args.outer))
    test_point = COM + np.array([R_edge, 0.0])

    # ----- Figure and axes -----
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_axes([0.08, 0.24, 0.84, 0.72])
    ax.set_aspect('equal', 'box')
    galaxy_scatter = ax.scatter(pos[:,0], pos[:,1], s=6, alpha=0.6)
    test_star_plot, = ax.plot([test_point[0]], [test_point[1]], marker='*', markersize=10)
    com_plot, = ax.plot([COM[0]], [COM[1]], marker='x', markersize=8)
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_title("Toy Galaxy Rotation Calculator (symmetrized)")

    text_box = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

    ax_sG   = fig.add_axes([0.10, 0.18, 0.70, 0.02])
    ax_kGR  = fig.add_axes([0.10, 0.15, 0.70, 0.02])
    ax_att  = fig.add_axes([0.10, 0.12, 0.70, 0.02])
    ax_boost= fig.add_axes([0.10, 0.09, 0.70, 0.02])
    ax_vobs = fig.add_axes([0.10, 0.06, 0.70, 0.02])
    ax_spread = fig.add_axes([0.10, 0.03, 0.70, 0.02])
    ax_soft = fig.add_axes([0.10, 0.21, 0.70, 0.02])

    # Sliders with CLI-provided initial values
    s_Gscale = Slider(ax_sG,   'G scale', 0.1, 5.0, valinit=float(args.gscale), valstep=0.01)
    s_kGR    = Slider(ax_kGR,  'GR k (toy)', 0.0, 20000.0, valinit=float(args.grk), valstep=10.0)
    s_att    = Slider(ax_att,  'Extra atten (0-1)', 0.0, 0.8, valinit=float(args.atten), valstep=0.01)
    s_boost  = Slider(ax_boost,'Boost fraction', 0.0, 1.5, valinit=float(args.boost), valstep=0.01)
    s_vobs   = Slider(ax_vobs, 'Observed v (km/s)', 0.0, 400.0, valinit=float(args.vobs), valstep=1.0)
    s_spread = Slider(ax_spread, 'Spread ×', 0.5, 3.0, valinit=float(args.spread), valstep=0.01)
    s_soften = Slider(ax_soft, 'Softening (kpc)', 0.01, 1.0, valinit=float(args.soften), valstep=0.01)

    ax_radio = fig.add_axes([0.82, 0.08, 0.14, 0.13])
    radio = RadioButtons(ax_radio, config_names, active=config_names.index(args.config))

    ax_outer = fig.add_axes([0.10, 0.24, 0.70, 0.02])
    s_outer = Slider(ax_outer, 'Outer percentile', 0.80, 0.995, valinit=float(args.outer), valstep=0.001)

    ax_seed = fig.add_axes([0.82, 0.03, 0.14, 0.03])
    s_seed = Slider(ax_seed, 'Seed', 0, 9999, valinit=int(args.seed), valstep=1)

    def recalc_display():
        # Physics-only recalculation (no resampling)
        nonlocal pos, masses, COM, R_all, R_edge, test_point
        G_eff_base = G_AST * float(s_Gscale.val)
        a_gr_vec, a_newton_vec = net_acceleration_at_point(
            test_point, pos, masses, G_eff_base,
            k_GR=float(s_kGR.val),
            atten_extra=float(s_att.val),
            soften_kpc=float(s_soften.val)
        )
        a0 = np.linalg.norm(a_newton_vec)
        agr = np.linalg.norm(a_gr_vec)
        drop_frac = 0.0
        if a0 > 0:
            drop_frac = max(0.0, 1.0 - (agr / a0))

        f_boost = float(s_boost.val)
        boost_factor = 1.0
        if f_boost > 0 and drop_frac > 0 and (1.0 - f_boost * drop_frac) > 1e-6:
            boost_factor = 1.0 / (1.0 - f_boost * drop_frac)

        a_final_vec = a_gr_vec * boost_factor
        v_newton = circular_speed_from_accel(a_newton_vec, test_point - COM)
        v_gr_noboost = circular_speed_from_accel(a_gr_vec, test_point - COM)
        v_final = circular_speed_from_accel(a_final_vec, test_point - COM)
        v_obs = float(s_vobs.val)
        dv = v_final - v_obs

        total_mass = masses.sum()
        lines = [
            f"Config: {radio.value_selected}  |  N lumps: {len(masses)}  |  Total luminous mass: {total_mass:,.2e} Msun",
            f"Outer-edge radius (percentile {s_outer.val:.3f}): R = {np.linalg.norm(test_point - COM):.2f} kpc",
            f"G scale: {s_Gscale.val:.2f}   |   GR k (toy): {s_kGR.val:.0f}   |   Extra atten: {s_att.val:.2f}   |   Softening: {s_soften.val:.2f} kpc",
            f"Drop from attenuation: {100*drop_frac:.4f}%   |   Auto-boost factor: ×{(1.0 / (1.0 - float(s_boost.val) * drop_frac) if (1.0 - float(s_boost.val) * drop_frac) > 1e-6 else np.inf):.5f}",
            f"Speeds (km/s):  Newtonian={v_newton:.2f}   GR,no boost={v_gr_noboost:.2f}   Final (with boost)={v_final:.2f}",
            f"Observed target: {v_obs:.1f} km/s   |   Final − Observed = {dv:+.2f} km/s"
        ]
        text_box.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def rebuild_galaxy_and_draw():
        # Structural changes: resample positions and update geometry
        nonlocal pos, masses, COM, R_all, R_edge, test_point
        cfg = radio.value_selected
        seed = int(s_seed.val)
        spread = float(s_spread.val)
        pos, masses = make_galaxy(cfg, seed, spread)
        COM = (pos * masses[:, None]).sum(axis=0) / masses.sum()
        R_all = np.linalg.norm(pos - COM, axis=1)
        R_edge = np.quantile(R_all, float(s_outer.val))
        test_point = COM + np.array([R_edge, 0.0])
        galaxy_scatter.set_offsets(pos)
        test_star_plot.set_data([test_point[0]], [test_point[1]])
        com_plot.set_data([COM[0]], [COM[1]])
        R_plot = np.max(np.linalg.norm(pos - COM, axis=1)) * 1.15 + 1e-3
        ax.set_xlim(COM[0] - R_plot, COM[0] + R_plot)
        ax.set_ylim(COM[1] - R_plot, COM[1] + R_plot)
        recalc_display()

    # Wire callbacks: separate physics from structure
    s_Gscale.on_changed(lambda _v: recalc_display())
    s_kGR.on_changed(lambda _v: recalc_display())
    s_att.on_changed(lambda _v: recalc_display())
    s_boost.on_changed(lambda _v: recalc_display())
    s_vobs.on_changed(lambda _v: recalc_display())
    s_soften.on_changed(lambda _v: recalc_display())

    s_spread.on_changed(lambda _v: rebuild_galaxy_and_draw())
    s_outer.on_changed(lambda _v: rebuild_galaxy_and_draw())
    s_seed.on_changed(lambda _v: rebuild_galaxy_and_draw())
    radio.on_clicked(lambda _label: rebuild_galaxy_and_draw())

    # Initial draw
    rebuild_galaxy_and_draw()

    # Headless save or interactive show
    if args.save:
        recalc_display()
        out_path = os.path.abspath(args.save)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved preview image to {out_path}")
        return

    if not backend_is_interactive():
        out_path = os.path.abspath('rotation_preview.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Non-interactive backend detected; saved preview image to {out_path}")
        return

    plt.show()

if __name__ == "__main__":
    main()
