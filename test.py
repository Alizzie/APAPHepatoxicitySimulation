from LobuleQuadrant import LobuleQuadrant
from MetabolismModel import MetabolismModel
from LobuleVisualizer import LobuleVisualizer

N_STEPS = 300000

# ── Initialise transport model ────────────────────────────────────────────────
lobule = LobuleQuadrant(direction="top-left")
viz = LobuleVisualizer(lobule)

# ── Initialise metabolism model ───────────────────────────────────────────────
metab = MetabolismModel(
    physio_grid=lobule.physio_grid,
    hep_labels=lobule.hep_labels,
    inlet_pos=lobule.inlet_pos,
    outlet_pos=lobule.outlet_pos,
)

log_txt = open("log.txt", "w", encoding="utf-8")
print(
    f"Inlet vx={lobule.vx[lobule.inlet_pos]:.3e}, vy={lobule.vy[lobule.inlet_pos]:.3e}"
)
print(
    f"Outlet vx={lobule.vx[lobule.outlet_pos]:.3e}, vy={lobule.vy[lobule.outlet_pos]:.3e}"
)

viz.quiver()

# ── Simulation loop ───────────────────────────────────────────────────────────
for step in range(N_STEPS):
    print(f"Step {step+1}/{N_STEPS}", end="\r")

    # 1. Transport Step
    # Advects blood, diffuses, and performs hepatocyte <-> sinusoid exchange
    C_full = lobule.compute_flux()

    # # 2. Handshake: Transport -> Metabolism
    # # Tell the metabolism model how much APAP is currently in the hepatocytes
    # metab.P = C_full * lobule.hep_mask

    # # 3. Metabolism Step
    # # Hepatocytes consume APAP (P) and generate metabolites (NAPQI, GSH, etc.)
    # P_new = metab.step()

    # # 4. Handshake: Metabolism -> Transport
    # # Reassemble the grid: untouched sinusoid drug + newly reduced hepatocyte drug
    # lobule.C = (C_full * lobule.sin_mask) + (P_new * lobule.hep_mask)

    # 5. Record
    lobule.record()
    # metab.record()

    # 6. Periodic audit & Visualization
    if step % 1000 == 0:
        viz.concentration(step=step)
        # viz.metabolism_state(metab, step=step)
        lobule.audit_mass(step)
        # means = metab.get_zone_means()
        # print(f"\n--- METABOLISM ZONE STATS (Step {step}) ---")
        # for z in (1, 2, 3):
        #     log_txt.write(
        #         f"Step {step}: Zone {z}: APAP={means[z]['P']:.2f} "
        #         f"P Zone history: {metab.zone_P_history[z][-5:]}"
        #         f"NAPQI={means[z]['NAPQI']:.4f} "
        #         f"NAPQI Zone history: {metab.zone_N_history[z][-5:]}"
        #         f"GSH={means[z]['GSH']:.1f} "
        #         f"GSH Zone history: {metab.zone_G_history[z][-5:]}"
        #         f"Adducts={means[z]['Ci']:.6f}\n"
        #         f"Toxicity history: {metab.zone_toxicity_history[z][-5:]}\n"
        #         f"Reservoir concentration: {lobule.c_reservoir:.2f} µM\n"
        #         f"Inlet pixel concentration: {lobule.C[lobule.inlet_pos].sum():.2f} µM\n"
        #         f"Mass entered: {lobule.total_mass_history[-1]:.2f} µmol\n"
        #         f"Mass in lobule: {lobule.C.sum():.2f} µmol\n"
        #         "-----------------------------\n"
        #     )

        #     print(
        #         f"Zone {z} | APAP: {means[z]['P']:.2f} | "
        #         f"NAPQI: {means[z]['NAPQI']:.4e} | "
        #         f"GSH: {means[z]['GSH']:.1f} | "
        #         f"Adducts: {means[z]['Ci']:.6e}"
        #     )

        print("-------------------------------------------\n")

        wait = input("Press Enter to continue...")
log_txt.close()
