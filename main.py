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
    lobule = FullLobule()
    viz = LobuleVisualizer(lobule)

    q_tl = lobule.quadrants["top-left"]
    q_tr = lobule.quadrants["top-right"]

    # top-right should be a left-right mirror of top-left
    print(np.array_equal(q_tl.physio_grid, np.fliplr(q_tr.physio_grid)))
    print(np.allclose(q_tl.P, np.fliplr(q_tr.P)))
    print(np.allclose(q_tl.vx, -np.fliplr(q_tr.vx)))
    print(np.allclose(q_tl.vy, np.fliplr(q_tr.vy)))

    # viz.quiver_quadrants()

    for steps in range(1000):
        C = lobule.compute_flux()
        if steps % 100 == 0:
            print(f"Step {steps:>4}  t={steps * config.DT:.3f}s  mass={np.sum(C):.4e}")
            viz.quadrants_side_by_side()

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
