import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FullLobule import FullLobule
from LobuleQuadrant import LobuleQuadrant
from LobuleVisualizer import LobuleVisualizer
from config import Config

config = Config()

# ══════════════════════════════════════════════════════════════════════════════
# ── Main ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # lobule = FullLobule()
    # viz = LobuleVisualizer(lobule)

    # for i in range(10000):
    #     print(f"Initial flux step {i+1}/10000", end="\r")
    #     lobule.compute_flux()

    #     if (i + 1) % 5000 == 0:
    #         viz.quadrants_side_by_side()
    # viz.history()
    # quit()

    q = LobuleQuadrant("top-left")
    q = LobuleQuadrant("top-left")
    n = q.physio_grid.shape[0]
    dx = config.LOBULE_SIZE / n
    print(f"LOBULE_SIZE = {config.LOBULE_SIZE}")
    print(f"n = {n}")
    print(f"dx = {dx:.3e}")
    print(f"DT = {config.DT:.3e}")
    print(f"D_SIN = {config.D_SIN:.3e}")
    print(
        f"Convective CFL = {(np.abs(q.vx).max() + np.abs(q.vy).max()) * config.DT / dx:.3f}"
    )
    uptake_rate = config.F_UNBOUND * config.CL_INFLUX / config.V_SINUSOID
    print(f"Uptake rate = {uptake_rate:.3e} s⁻¹")
    print(f"Uptake * DT = {uptake_rate * config.DT:.3f}")
    viz = LobuleVisualizer(q)
    total = []
    print(f"Inlet concentration: {q.C[q.inlet_pos].sum():.4e} µM")
    for step in range(200000):
        print(f"Step {step+1}/200000", end="\r")
        C_before = q.C.sum()
        q.compute_flux()
        C_after = q.C.sum()

        if (step + 1) % 10000 == 0 or step == 0:
            viz.concentration(step)
            q.audit_mass(step_num=step)
            inlet_uM = q.C[q.inlet_pos] / 1000
            res_uM = q.c_reservoir / 1000

            print(f"Reservoir (Blood) Concentration: {res_uM:.2f} µM")
            print(f"Inlet Pixel Concentration:       {inlet_uM:.2f} µM")
            wait = input("Press Enter to continue...")

        total.append(q.C.sum())

    plt.plot(total)
    plt.title("top-left mass over time")
    plt.show()
    quit()

    TOTAL_STEPS = 10000
    PLOT_EVERY = 1000
    N_FRAMES = TOTAL_STEPS // PLOT_EVERY

    dirs = ["top-left", "top-right", "bottom-left", "bottom-right"]
    positions = {
        "top-left": (0, 0),
        "top-right": (0, 1),
        "bottom-left": (1, 0),
        "bottom-right": (1, 1),
    }

    # ── Print initial state ───────────────────────────────────────────────────
    for d in lobule.DIRS:
        q = lobule.quadrants[d]
        print(
            f"{d}: vx_mean={q.vx.mean():.3e} vy_mean={q.vy.mean():.3e} "
            f"C_sum={q.C.sum():.3e} C_max={q.C.max():.3e}"
        )

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Quadrants at t = 0.000s", fontsize=13)

    ims, cbars = {}, {}
    for d in dirs:
        ax = axes[positions[d]]
        im = ax.imshow(
            lobule.quadrants[d].C,
            cmap="viridis",
            origin="upper",
            vmin=0,
            vmax=config.INLET_CONC,
        )
        ax.set_title(d)
        ax.axis("off")
        cbars[d] = plt.colorbar(im, ax=ax, label="Concentration (µM)")
        ims[d] = im

    plt.tight_layout()

    step_counter = [0]

    # ── Animation update ──────────────────────────────────────────────────────
    def update(_):
        for _ in range(PLOT_EVERY):
            C = lobule.compute_flux()
            step_counter[0] += 1

        s = step_counter[0]
        print(f"Step {s:>7}  t={s * config.DT:.3f}s  mass={np.sum(C):.4e}")

        fig.suptitle(
            f"Quadrants at t = {s * config.DT:.3f}s  " f"mass={np.sum(C):.4e}",
            fontsize=13,
        )

        for d in dirs:
            data = lobule.quadrants[d].C
            ims[d].set_data(data)
            ims[d].set_clim(vmin=0, vmax=max(data.max(), 1e-12))

        return list(ims.values())

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        interval=50,
        blit=False,
        repeat=False,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Saving video...")
    ani.save("lobule.mp4", writer="ffmpeg", fps=20, dpi=150)
    print("Saved lobule.mp4")
    plt.close(fig)

    viz.history()
