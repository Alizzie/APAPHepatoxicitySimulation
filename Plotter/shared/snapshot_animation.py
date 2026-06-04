import matplotlib.pyplot as plt

frames_to_save = [0, 10, 25, 50]  # adjust indices to match your times
for idx in frames_to_save:
    plt.figure(figsize=(5, 5))
    plt.imshow(
        quadrant.concentration_history[idx] / config.V_PIXEL,
        cmap="viridis",
        origin="upper",
    )
    plt.colorbar(label="Concentration (µM)")
    plt.title(f"t = {quadrant.time_history[idx*20]:.1f} s")
    plt.tight_layout()
    plt.savefig(f"images/transport_snapshot_t{idx}.png", dpi=300)
    plt.close()
