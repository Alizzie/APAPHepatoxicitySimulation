"""
dashboard.py — Real-time liver lobule simulation dashboard
Run with:  panel serve dashboard.py --autoreload --show
"""

import numpy as np
import panel as pn
import holoviews as hv
from holoviews.streams import Pipe
import param

from config import Config
from LobuleQuadrant import LobuleQuadrant
from MetabolismModel import MetabolismModel

config = Config()

pn.extension("bokeh")
hv.extension("bokeh")

BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#21262d"
TEXT = "#e6edf3"
DEAD_COL = "#8B4513"

PLOT_H = 300
SLIDER_W = 260

CI_DEATH_THRESHOLD = 1.1  # µM


# ══════════════════════════════════════════════════════════════════════════════
class SimController(param.Parameterized):

    dose = param.Number(
        default=26450, bounds=(1000, 100000), step=500, label="Dose (µmol)"
    )
    dt = param.Number(
        default=0.001, bounds=(0.0001, 0.01), step=0.0001, label="Timestep (s)"
    )
    steps_frame = param.Integer(default=200, bounds=(1, 2000), label="Steps / frame")

    def __init__(self, **params):
        super().__init__(**params)
        self._running = False
        self._step = 0
        self._cb = None
        self._build_model()
        self._build_pipes()
        self._build_layout()

    # ── Model ─────────────────────────────────────────────────────────────────
    def _build_model(self):
        dose = self.dose / config.V_BLOOD / 4
        self.lobule = LobuleQuadrant(dose=dose, exchange_on=False)
        self.metab = MetabolismModel(
            physio_grid=self.lobule.physio_grid,
            hep_labels=self.lobule.hep_labels,
            inlet_pos=self.lobule.inlet_pos,
            outlet_pos=self.lobule.outlet_pos,
        )
        self.zone_borders_plot = self._generate_border_plot()

    # ── Pipes — each sends a dict {"data": arr, "vmax": float} ───────────────
    def _build_pipes(self):
        n = self.lobule.physio_grid.shape[0]
        blank = np.zeros((n, n))
        blank3 = np.zeros((n, n, 2))
        self.pipe_conc = Pipe(data={"data": blank, "vmax": 1.0})
        self.pipe_hep = Pipe(data={"data": blank3, "vmax": 1.0})
        self.pipe_napqi = Pipe(data={"data": blank, "vmax": 1.0})
        self.pipe_ci = Pipe(data={"data": blank, "vmax": 1.0})

    # ── Plot helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _img_opts(title, cmap, vmax, h=PLOT_H):
        return dict(
            title=title,
            cmap=cmap,
            clim=(0, vmax),
            colorbar=True,
            responsive=True,
            height=h,
            xaxis=None,
            yaxis=None,
            bgcolor=BG,
            toolbar=None,
            default_tools=[],
        )

    def _make_image(self, pipe, title, cmap):
        def _cb(data):
            arr = data["data"]
            vmax = max(float(data["vmax"]), 1e-9)
            img = hv.Image(arr, bounds=(0, 0, 1, 1)).opts(
                title=title,
                cmap=cmap,
                clim=(0, vmax),
                colorbar=True,
                aspect="square",
                frame_width=PLOT_H,
                frame_height=PLOT_H,
                xaxis=None,
                yaxis=None,
                bgcolor=BG,
                toolbar=None,
                default_tools=[],
            )

            return (img * self.zone_borders_plot).opts(toolbar=None, default_tools=[])

        return hv.DynamicMap(_cb, streams=[pipe])

    def _make_hep_overlay(self, pipe):
        def _cb(data):
            arr = data["data"]
            vmax = max(float(data["vmax"]), 1e-9)
            conc = arr[:, :, 0]
            dead = arr[:, :, 1]
            base = hv.Image(np.where(conc > 0, conc, np.nan), bounds=(0, 0, 1, 1)).opts(
                title="Hepatocyte drug + dead cells (brown)",
                cmap="Blues",
                clim=(0, vmax),
                colorbar=True,
                aspect="square",
                frame_width=PLOT_H,
                frame_height=PLOT_H,
                xaxis=None,
                yaxis=None,
                bgcolor=BG,
                toolbar=None,
                default_tools=[],
            )
            overlay = hv.Image(
                np.where(dead > 0, 1.0, np.nan), bounds=(0, 0, 1, 1)
            ).opts(
                cmap=[DEAD_COL],
                clim=(0, 1),
                colorbar=False,
                frame_width=PLOT_H,
                frame_height=PLOT_H,
                xaxis=None,
                yaxis=None,
                bgcolor=BG,
                toolbar=None,
                default_tools=[],
                alpha=0.85,
            )
            return (base * overlay * self.zone_borders_plot).opts(
                toolbar=None, default_tools=[]
            )

        return hv.DynamicMap(_cb, streams=[pipe])

    def _generate_border_plot(self):
        """Dynamically calculates exact zone boundaries directly from the zone_map"""
        zmap = self.metab.zone_map
        n = zmap.shape[0]

        # Calculate Manhattan distance (r + c) from the top-left (0,0) for all pixels
        rows, cols = np.indices(zmap.shape)
        dist = rows + cols

        # Find the exact pixel boundary (max distance) for Zone 1 and Zone 2.
        # Add 0.5 to draw the line perfectly in the space *between* the pixels.
        t1 = np.max(dist[zmap == 1]) + 0.5
        t2 = np.max(dist[zmap == 2]) + 0.5

        def get_intersections(t):
            # Maps the exact pixel threshold 't' to the normalized 0.0-1.0 Cartesian bounds
            pts = []
            if t <= n:
                pts.append((0, 1 - t / n))  # Intersection with left edge
                pts.append((t / n, 1))  # Intersection with top edge
            else:
                pts.append(((t - n) / n, 0))  # Intersection with bottom edge
                pts.append((1, 1 - (t - n) / n))  # Intersection with right edge
            return pts

        return hv.Path([get_intersections(t1), get_intersections(t2)]).opts(
            color="white", line_dash="dashed", line_width=2, alpha=0.5
        )

    def _panel_col(self, plot):
        return pn.Column(
            pn.pane.HoloViews(plot),
            styles={
                "background": PANEL_BG,
                "border": f"1px solid {BORDER}",
                "border-radius": "6px",
                "padding": "6px",
            },
        )

    # ── Config summary ────────────────────────────────────────────────────────
    @staticmethod
    def _config_summary():
        cfg = Config()
        rows = [
            ("K₄₅₀ Z1", f"{cfg.K_450_ZONE1:.2e} s⁻¹"),
            ("K₄₅₀ Z2", f"{cfg.K_450_ZONE2:.2e} s⁻¹"),
            ("K₄₅₀ Z3", f"{cfg.K_450_ZONE3:.2e} s⁻¹"),
            ("K_G", f"{cfg.K_G:.2e} s⁻¹"),
            ("K_S", f"{cfg.K_S:.2e} µM⁻¹s⁻¹"),
            ("K_GSH", f"{cfg.K_GSH:.2e} µM⁻¹s⁻¹"),
            ("K_PSH", f"{cfg.K_PSH:.2e} s⁻¹"),
            ("K_N", f"{cfg.K_N:.2e} s⁻¹"),
            ("CL influx", f"{cfg.CL_INFLUX:.2e} m³/s"),
            ("CL efflux", f"{cfg.CL_EFFLUX:.2e} m³/s"),
            ("F unbound", f"{cfg.F_UNBOUND}"),
            ("GSH₀", f"{cfg.G_INIT:.1f} µM"),
            ("SO₄₀", f"{cfg.S_INIT:.1f} µM"),
            ("U_x", f"{cfg.U_X:.2e} m/s"),
            ("D_sin", f"{cfg.D_SIN:.2e} m²/s"),
            ("V_blood", f"{cfg.V_BLOOD} L"),
            ("Ci death", f"{CI_DEATH_THRESHOLD} µM"),
        ]
        lines = ["| Parameter | Value |", "|---|---|"]
        for name, val in rows:
            lines.append(f"| **{name}** | {val} |")
        return "\n".join(lines)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        sliders = pn.Param(
            self,
            parameters=["dose", "dt", "steps_frame"],
            widgets={
                "dose": pn.widgets.FloatSlider,
                "dt": pn.widgets.FloatSlider,
                "steps_frame": pn.widgets.IntSlider,
            },
            width=SLIDER_W,
            show_name=False,
        )
        # Force slider labels white via CSS
        pn.config.raw_css.append(
            """
            .bk-slider-title { color: #e6edf3 !important; }
            .noUi-tooltip    { color: #e6edf3 !important; background: #161b22 !important; }
            .bk-input-group label { color: #e6edf3 !important; }
            .panel-widget-box .widget-label { color: #e6edf3 !important; }
        """
        )

        self._btn_play = pn.widgets.Button(
            name="▶  Play", button_type="success", width=120
        )
        self._btn_pause = pn.widgets.Button(
            name="⏸  Pause", button_type="warning", width=120
        )
        self._btn_reset = pn.widgets.Button(
            name="↺  Reset", button_type="danger", width=240
        )
        self._step_md = pn.pane.Markdown(
            "**Step:** 0 &nbsp; **t:** 0.000 s",
            styles={"color": TEXT},
        )
        self._stats_md = pn.pane.Markdown(
            self._stats_text({}),
            styles={
                "color": TEXT,
                "font-size": "12px",
                "background": PANEL_BG,
                "padding": "8px",
                "border": f"1px solid {BORDER}",
                "border-radius": "6px",
            },
            width=SLIDER_W,
        )
        config_md = pn.pane.Markdown(
            self._config_summary(),
            styles={
                "color": TEXT,
                "font-size": "11px",
                "background": PANEL_BG,
                "padding": "8px",
                "border": f"1px solid {BORDER}",
                "border-radius": "6px",
            },
            width=SLIDER_W,
        )

        self._btn_play.on_click(self._on_play)
        self._btn_pause.on_click(self._on_pause)
        self._btn_reset.on_click(self._on_reset)

        sidebar = pn.Column(
            pn.pane.Markdown("## ⚙️ Parameters", styles={"color": TEXT}),
            sliders,
            pn.layout.Divider(),
            pn.pane.Markdown("## Controls", styles={"color": TEXT}),
            pn.Row(self._btn_play, self._btn_pause),
            self._btn_reset,
            self._step_md,
            pn.layout.Divider(),
            pn.pane.Markdown("## 📊 Zone Statistics", styles={"color": TEXT}),
            self._stats_md,
            pn.layout.Divider(),
            pn.pane.Markdown("## 🔬 Fixed Parameters", styles={"color": TEXT}),
            config_md,
            width=SLIDER_W + 20,
            scroll=True,
            styles={
                "background": PANEL_BG,
                "padding": "10px",
                "border-right": f"2px solid {BORDER}",
            },
        )

        row1 = pn.Row(
            self._panel_col(
                self._make_image(self.pipe_conc, "Drug concentration (µM)", "inferno")
            ),
            self._panel_col(self._make_hep_overlay(self.pipe_hep)),
            sizing_mode="stretch_width",
        )
        row2 = pn.Row(
            self._panel_col(self._make_image(self.pipe_napqi, "NAPQI (µM)", "Oranges")),
            self._panel_col(self._make_image(self.pipe_ci, "Adducts Ci (µM)", "Reds")),
            sizing_mode="stretch_width",
        )

        main = pn.Column(
            pn.pane.Markdown(
                "# 🔬 Liver Lobule Simulation Dashboard", styles={"color": TEXT}
            ),
            row1,
            row2,
            sizing_mode="stretch_width",
            styles={"background": BG, "padding": "10px"},
        )

        self.layout = pn.Row(
            sidebar,
            main,
            sizing_mode="stretch_both",
            styles={"background": BG},
        )

    # ── Stats table ───────────────────────────────────────────────────────────
    def _stats_text(self, means):
        def zrow(label, key, fmt):
            if not means:
                return f"| **{label}** | — | — | — |"
            v = [means.get(z, {}).get(key, 0.0) for z in (1, 2, 3)]
            return f"| **{label}** | {v[0]:{fmt}} | {v[1]:{fmt}} | {v[2]:{fmt}} |"

        remaining = self.dose - self.lobule.total_mass_exited
        lines = [
            f"**Initial Dose:** {self.dose:.1f} µM &nbsp; "
            f"**Remaining:** {remaining:.1f} µM",
            "",
            "| | Z1 | Z2 | Z3 |",
            "|---|---|---|---|",
            zrow("APAP hepa", "P", ".1f"),
            zrow("NAPQI", "NAPQI", ".3f"),
            zrow("Ci", "Ci", ".4f"),
            zrow("GSH", "GSH", ".1f"),
            zrow("SO₄", "Sulfate", ".1f"),
        ]
        return "\n".join(lines)

    # ── Zone means ────────────────────────────────────────────────────────────
    def _get_zone_means(self):
        out = {}
        for z in (1, 2, 3):
            mask = (self.metab.zone_map == z) & self.metab.hep_mask
            n_px = mask.sum()
            out[z] = {
                "P": self.metab.P[mask].mean() if n_px else 0.0,
                "NAPQI": self.metab.NAPQI[mask].mean() if n_px else 0.0,
                "GSH": self.metab.GSH[mask].mean() if n_px else 0.0,
                "Sulfate": self.metab.Sulfate[mask].mean() if n_px else 0.0,
                "Ci": self.metab.Ci[mask].mean() if n_px else 0.0,
            }
        return out

    # ── Simulation tick ───────────────────────────────────────────────────────
    def _tick(self):
        for _ in range(self.steps_frame):
            C_full = self.lobule.compute_flux(dt=self.dt)

            # C_hep_transport = C_full * self.lobule.hep_mask
            # delta = C_hep_transport - self.metab.P
            # self.metab.P = np.maximum(self.metab.P + delta, 0.0)
            # P_new = self.metab.step()
            # self.lobule.C = (C_full * self.lobule.sin_mask) + (
            #     P_new * self.lobule.hep_mask
            # )
            self._step += 1

        means = self._get_zone_means()
        C = self.lobule.C
        hm = self.metab.hep_mask
        n = self.lobule.physio_grid.shape[0]

        def robust_vmax(arr):
            """Mean + 2 std of non-zero interior pixels — robust to border spikes."""
            interior = arr[1:-1, 1:-1]
            vals = interior[interior > 0]
            if vals.size == 0:
                return 1e-9
            return max(float(vals.mean() + 2 * vals.std()), 1e-9)

        vmax_conc = robust_vmax(C)
        vmax_napqi = robust_vmax(self.metab.NAPQI * hm)
        vmax_ci = robust_vmax(self.metab.Ci * hm)

        self.pipe_conc.send({"data": C, "vmax": vmax_conc})

        hep_data = np.zeros((n, n, 2))
        hep_data[:, :, 0] = np.where(hm, C, np.nan)
        hep_data[:, :, 1] = np.where(
            (self.metab.Ci >= CI_DEATH_THRESHOLD) & hm, 1.0, np.nan
        )
        self.pipe_hep.send({"data": hep_data, "vmax": vmax_conc})

        self.pipe_napqi.send({"data": self.metab.NAPQI * hm, "vmax": vmax_napqi})
        self.pipe_ci.send({"data": self.metab.Ci * hm, "vmax": vmax_ci})

        self._stats_md.object = self._stats_text(
            means,
        )
        self._step_md.object = (
            f"**Step:** {self._step:,} &nbsp; **t:** {self._step * self.dt:.3f} s"
            f"\n\n**Dose:** {self.dose:.0f} µmol &nbsp;"
        )

    # ── Button handlers ───────────────────────────────────────────────────────
    def _on_play(self, _=None):
        if not self._running:
            self._running = True
            self._cb = pn.state.add_periodic_callback(self._tick, period=50)

    def _on_pause(self, _=None):
        if self._running and self._cb is not None:
            self._running = False
            self._cb.stop()
            self._cb = None

    def _on_reset(self, _=None):
        self._on_pause()
        self._step = 0
        self._build_model()
        self._build_pipes()
        n = self.lobule.physio_grid.shape[0]
        blank = np.zeros((n, n))
        blank3 = np.zeros((n, n, 2))
        for pipe, d in [
            (self.pipe_conc, {"data": blank, "vmax": 1.0}),
            (self.pipe_hep, {"data": blank3, "vmax": 1.0}),
            (self.pipe_napqi, {"data": blank, "vmax": 1.0}),
            (self.pipe_ci, {"data": blank, "vmax": 1.0}),
        ]:
            pipe.send(d)
        self._stats_md.object = self._stats_text({})
        self._step_md.object = "**Step:** 0 &nbsp; **t:** 0.000 s"


# ── Serve ─────────────────────────────────────────────────────────────────────
sim = SimController()
sim.layout.servable(title="Liver Lobule Simulation")
