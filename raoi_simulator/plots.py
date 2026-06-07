# -*- coding: utf-8 -*-
"""
Statistical analysis figures for RAOI swarm simulation runs.

Receives the dict returned by statistical_run() and generates a set of
publication-quality figures describing swarm behaviour across multiple
replicas. Each figure is saved as a high-resolution PNG.

Figures generated:
  1. convergence_distribution   — violin + box of convergence_time vs
                                  physical_convergence_time per stimulus.
  2. cohesion_evolution         — temporal cohesion curve (mean ± std)
                                  derived from full position reports.
  3. fragmentation_evolution    — temporal fragmentation index (mean ± std)
                                  with first-arrival event markers per stimulus.
  4. stimulus_load_balance      — occupancy fraction and mean robots per
                                  stimulus, with individual-replica scatter.
  5. convergence_ecdf           — empirical CDF of convergence_time and
                                  physical_convergence_time across replicas.
  6. fuzzy_wi_surface           — 2-D heatmap of the fuzzy inference surface
                                  w_I(density, distance) for one and several
                                  stimuli detected simultaneously.
  7. localization_timeline      — Gantt-style chart showing approach,
                                  group-buildup and dwelling phases per
                                  stimulus, based on median across replicas.
  8. raoi_state_evolution       — stacked area chart of the fraction of
                                  robots in each RAOI behavioural state
                                  (free, repulsion, orientation, attraction,
                                  influence) over simulation time.

Usage:
    from plots import generate_all

    results = statistical_run(replicas=30, ...)
    generate_all(results, all_reports, output_dir="figures")

    # Or from the command line using a saved .npy file:
    python plots.py results/stat_20260511_120000.npy

Authors: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — compatible with all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# ── Stimulus colour palette ───────────────────────────────────────────────────
STIM_COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261", "#6A4C93"]

# ── RAOI state labels and colours ─────────────────────────────────────────────
_STATE_LABELS = ["Free", "Repulsion", "Attraction", "Orientation", "Influence"]
_STATE_COLORS = ["#AAAAAA", "#E63946", "#457B9D", "#2A9D8F", "#F4A261"]

# ── Base style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stim_label(k: int, stimuli: list) -> str:
    """Short label for stimulus k."""
    s = stimuli[k]
    return f"S{k+1} ({s['x']},{s['y']}) r_s={s.get('r', 1.0)}"


def _save(fig: plt.Figure, path: str) -> str:
    """Save figure, close it and print confirmation."""
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Convergence time distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_distribution(
    results:    dict,
    output_dir: str = "figures",
) -> str:
    """
    Violin + box overlay para convergence_time vs physical_convergence_time.

    Cada par de violines muestra la distribución completa de iteraciones
    que tardan en completar la detección sensorial (convergence_time) y la
    llegada física (physical_convergence_time) para cada estímulo.  Superponer
    ambas métricas revela cuánto tiempo transcurre entre el primer contacto
    sensorial y la ocupación física real del estímulo.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    all_metrics = results["all"]
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)

    # Build per-stimulus data: two series per stimulus
    ct_per  = [[m["convergence_time_per_stimulus"][k]          for m in all_metrics]
               for k in range(n_stim)]
    pct_per = [[m["physical_convergence_time_per_stimulus"][k]  for m in all_metrics]
               for k in range(n_stim)]
    ct_global  = [m["convergence_time"]          for m in all_metrics]
    pct_global = [m["physical_convergence_time"]  for m in all_metrics]

    n_cols = n_stim + 1   # global + one per stimulus
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    def _violin_pair(ax, data_a, data_b, label_a, label_b, col_a, col_b, title):
        """Draws two violins side by side with box + median marker."""
        parts_a = ax.violinplot(data_a, positions=[1], showmedians=False,
                                showextrema=False)
        parts_b = ax.violinplot(data_b, positions=[2], showmedians=False,
                                showextrema=False)
        for parts, col in ((parts_a, col_a), (parts_b, col_b)):
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.55)
        # Box + whiskers
        for pos, data in ((1, data_a), (2, data_b)):
            q1, med, q3 = np.percentile(data, [25, 50, 75])
            iqr = q3 - q1
            lo  = max(min(data), q1 - 1.5 * iqr)
            hi  = min(max(data), q3 + 1.5 * iqr)
            ax.vlines(pos, lo, hi, color="#333", lw=1.5, zorder=3)
            ax.scatter([pos], [med], color="white", s=40, zorder=4,
                       edgecolors="#333", lw=1.5)
            ax.add_patch(plt.Rectangle((pos - 0.12, q1), 0.24, iqr,
                                        fc="#33333320", ec="#333", lw=1.2, zorder=3))
        ax.set_xticks([1, 2])
        ax.set_xticklabels([label_a, label_b], fontsize=9)
        ax.set_xlabel("Convergence type")
        ax.set_ylabel("Iterations")
        ax.set_title(title)
        ax.set_xlim(0.4, 2.6)
        ax.set_ylim(bottom=0)

    _violin_pair(axes[0],
                 ct_global, pct_global,
                 "Sensory", "Physical",
                 "#378ADD", "#E67B22",
                 "Global convergence")

    for k in range(n_stim):
        col = STIM_COLORS[k % len(STIM_COLORS)]
        _violin_pair(axes[k + 1],
                     ct_per[k], pct_per[k],
                     "Sensory", "Physical",
                     col, "#888888",
                     _stim_label(k, stimuli))

    # Legend
    legend_els = [
        mpatches.Patch(fc="#378ADD", alpha=0.6, label="Sensory (r_I + r_s)"),
        mpatches.Patch(fc="#E67B22", alpha=0.6, label="Physical (r_s only)"),
    ]
    fig.legend(handles=legend_els, loc="lower center",
               ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Convergence Time Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    return _save(fig, os.path.join(output_dir, "1_convergence_distribution.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Cohesion temporal evolution
# ══════════════════════════════════════════════════════════════════════════════

def plot_cohesion_evolution(
    results:    dict,
    reports:    list,
    output_dir: str = "figures",
) -> str:
    """
    Curva temporal de cohesión calculada desde los reportes completos.

    Calcula la distancia media de cada robot al centroide del enjambre en
    cada iteración, promedia sobre todas las réplicas y traza la banda ±1 std.
    Un boxplot de cohesion_final se superpone como inset para mostrar la
    distribución al término de la simulación.

    Args:
        results:    Dict devuelto por statistical_run().
        reports:    Lista de arrays report, shape (T, N, ≥2).
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    T = reports[0].shape[0]
    t = np.arange(T)

    # Cohesion per timestep per replica
    coh_matrix = np.zeros((len(reports), T))
    for r, rep in enumerate(reports):
        positions = rep[:, :, :2]                          # (T, N, 2)
        centroid  = positions.mean(axis=1, keepdims=True)  # (T, 1, 2)
        dists     = np.linalg.norm(positions - centroid, axis=2)  # (T, N)
        coh_matrix[r] = dists.mean(axis=1)                # (T,)

    mean_c = coh_matrix.mean(axis=0)
    std_c  = coh_matrix.std(axis=0)

    fig, (ax, ax_box) = plt.subplots(
        1, 2, figsize=(10, 4.5),
        gridspec_kw={"width_ratios": [4, 1]},
    )

    # Left panel — temporal curve
    ax.plot(t, mean_c, color="#2471A3", lw=2, label="Mean cohesion")
    ax.fill_between(t, mean_c - std_c, mean_c + std_c,
                    color="#2471A3", alpha=0.18, label="± 1 std")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean distance to centroid (m)")
    ax.set_title("Swarm Cohesion Over Time")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")

    # Right panel — boxplot of final cohesion distribution
    cf_vals = [m["cohesion_final"] for m in results["all"]]
    bp = ax_box.boxplot(cf_vals, patch_artist=True,
                        medianprops=dict(color="#222", lw=1.8),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
    bp["boxes"][0].set_facecolor("#AED6F1")
    bp["boxes"][0].set_alpha(0.8)
    ax_box.set_xticks([1])
    ax_box.set_xticklabels(["Final"], fontsize=9)
    ax_box.set_ylabel("Distance to centroid (m)", fontsize=9)
    ax_box.set_title("Final\ncohesion", fontsize=9)
    ax_box.set_ylim(bottom=0)
    ax_box.tick_params(axis="y", labelsize=8)

    fig.suptitle("Swarm Cohesion Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, os.path.join(output_dir, "2_cohesion_evolution.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Fragmentation temporal evolution
# ══════════════════════════════════════════════════════════════════════════════

def plot_fragmentation_evolution(
    results:    dict,
    reports:    list,
    output_dir: str = "figures",
) -> str:
    """
    Curva temporal del índice de fragmentación con marcadores de primera llegada.

    Calcula la fracción de pares de robots separados por más de r_a (radio de
    atracción) en cada iteración sobre todas las réplicas.  Las líneas verticales
    indican la mediana de first_arrival por estímulo, conectando visualmente la
    fragmentación emergente con la llegada sucesiva del enjambre a cada fuente.

    Args:
        results:    Dict devuelto por statistical_run().
        reports:    Lista de arrays report, shape (T, N, ≥2).
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    stimuli    = results["config"]["stimuli"]
    n_stim     = len(stimuli)
    r_a        = results["config"]["r_a"]
    all_metrics = results["all"]
    T = reports[0].shape[0]
    t = np.arange(T)

    # Fragmentation: fraction of pairs with dist > r_attraction
    frag_matrix = np.zeros((len(reports), T))
    for r, rep in enumerate(reports):
        positions = rep[:, :, :2]   # (T, N, 2)
        N = positions.shape[1]
        for ti in range(T):
            pos = positions[ti]     # (N, 2)
            diffs = pos[:, None, :] - pos[None, :, :]   # (N, N, 2)
            dists = np.linalg.norm(diffs, axis=2)        # (N, N)
            pairs = N * (N - 1) / 2
            frag_matrix[r, ti] = (dists[np.triu_indices(N, k=1)] > r_a).sum() / max(pairs, 1)

    mean_f = frag_matrix.mean(axis=0)
    std_f  = frag_matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, mean_f, color="#534AB7", lw=2, label="Mean fragmentation")
    ax.fill_between(t, mean_f - std_f, mean_f + std_f,
                    color="#534AB7", alpha=0.18, label="± 1 std")

    # First-arrival markers (median across replicas per stimulus)
    if n_stim > 1:
        for k in range(n_stim):
            fa_k = np.median([m["first_arrival"][k] for m in all_metrics])
            col  = STIM_COLORS[k % len(STIM_COLORS)]
            ax.axvline(fa_k, color=col, lw=1.5, ls="--", alpha=0.8,
                       label=f"S{k+1} first arrival (med={fa_k:.0f})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fragmentation index")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, T - 1)
    ax.set_title("Swarm Fragmentation Over Time")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, os.path.join(output_dir, "3_fragmentation_evolution.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Stimulus load balance
# ══════════════════════════════════════════════════════════════════════════════

def plot_stimulus_load_balance(
    results:    dict,
    output_dir: str = "figures",
) -> str:
    """
    Equilibrio de carga entre estímulos: fracción de tiempo ocupada y robots medios.

    Muestra dos paneles con barras de error (media ± std sobre réplicas) y
    puntos semitransparentes por réplica individual superpuestos.  La capa de
    puntos individuales permite detectar bimodalidad o réplicas atípicas que
    una barra de error promedio ocultaría.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)
    all_metrics = results["all"]

    occ_mat = np.array([m["stimulus_occupancy"]      for m in all_metrics])  # (R, K)
    mra_mat = np.array([m["mean_robots_at_stimulus"]  for m in all_metrics])  # (R, K)

    x      = np.arange(n_stim)
    # Short tick labels to avoid overlap; full coords as a figure note below
    labels = [f"S{k+1}" for k in range(n_stim)]
    w      = 0.45

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, n_stim * 2.8), 5))
    jitter = 0.08

    for ax, mat, ylabel, title, ylim in [
        (ax1, occ_mat, "Fraction of time", "Stimulus Occupancy Rate",  (0, 1.05)),
        (ax2, mra_mat, "Number of robots",  "Mean Simultaneous Robots", (0, None)),
    ]:
        means = mat.mean(axis=0)
        stds  = mat.std(axis=0)
        cols  = [STIM_COLORS[k % len(STIM_COLORS)] for k in range(n_stim)]

        ax.bar(x, means, width=w, yerr=stds, capsize=5, color=cols,
               alpha=0.7, error_kw=dict(ecolor="#333", lw=1.5, capthick=1.5),
               zorder=2)

        # Individual replica scatter
        for k in range(n_stim):
            jx = x[k] + np.random.uniform(-jitter, jitter, mat.shape[0])
            ax.scatter(jx, mat[:, k], color=cols[k], alpha=0.45, s=18,
                       edgecolors="none", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim[1]:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)

    # Stimulus coordinate reference below the figure
    coord_note = "   ".join(
        f"S{k+1}: ({stimuli[k]['x']}, {stimuli[k]['y']})  r_s={stimuli[k].get('r', 1.0)}"
        for k in range(n_stim)
    )
    fig.text(0.5, -0.04, coord_note, ha="center", fontsize=8, color="#555")
    fig.suptitle("Stimulus Load Balance", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return _save(fig, os.path.join(output_dir, "4_stimulus_load_balance.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Empirical CDF of convergence time
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_ecdf(
    results:    dict,
    output_dir: str = "figures",
) -> str:
    """
    Función de distribución acumulada empírica (ECDF) del tiempo de convergencia.

    La ECDF permite leer directamente la probabilidad de que el enjambre haya
    convergido antes de la iteración t, lo que es más informativo para un paper
    científico que un histograma binned.  Se trazan dos curvas: detección sensorial
    (convergence_time) y llegada física (physical_convergence_time), más las ECDFs
    individuales por estímulo si n_stim > 1.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    all_metrics = results["all"]
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)
    T           = results["config"]["iterations"]

    def _ecdf(data):
        xs = np.sort(data)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return np.concatenate([[0], xs]), np.concatenate([[0], ys])

    n_cols = 2 if n_stim > 1 else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    # Global ECDF
    ax = axes[0]
    ct_g   = [m["convergence_time"]          for m in all_metrics]
    pct_g  = [m["physical_convergence_time"]  for m in all_metrics]
    xs, ys = _ecdf(ct_g)
    ax.step(xs, ys, where="post", color="#378ADD", lw=2, label="Sensory (r_I + r_s)")
    xs, ys = _ecdf(pct_g)
    ax.step(xs, ys, where="post", color="#E67B22", lw=2, label="Physical (r_s only)")
    ax.axhline(0.5,  color="#333", lw=0.8, ls=":")
    ax.axhline(0.9,  color="#888", lw=0.8, ls=":")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative probability")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_title("Global Convergence ECDF")
    ax.legend(fontsize=9)
    ax.text(T * 0.98, 0.51, "50 %", fontsize=7, color="#333", ha="right", va="bottom")
    ax.text(T * 0.98, 0.91, "90 %", fontsize=7, color="#888", ha="right", va="bottom")

    # Per-stimulus ECDF
    if n_stim > 1:
        ax2 = axes[1]
        for k in range(n_stim):
            col    = STIM_COLORS[k % len(STIM_COLORS)]
            ct_k   = [m["convergence_time_per_stimulus"][k]          for m in all_metrics]
            pct_k  = [m["physical_convergence_time_per_stimulus"][k]  for m in all_metrics]
            xs, ys = _ecdf(ct_k)
            ax2.step(xs, ys, where="post", color=col, lw=2,
                     label=_stim_label(k, stimuli))
            xs, ys = _ecdf(pct_k)
            ax2.step(xs, ys, where="post", color=col, lw=1.2, ls="--", alpha=0.7)
        ax2.axhline(0.5, color="#333", lw=0.8, ls=":")
        ax2.axhline(0.9, color="#888", lw=0.8, ls=":")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cumulative probability")
        ax2.set_xlim(0, T)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Per-stimulus Convergence ECDF")
        ax2.legend(fontsize=8)
        # Solid = sensory, dashed = physical — add line-style legend entries
        style_handles = [
            Line2D([0], [0], color="#555", lw=1.5,        label="— Sensory"),
            Line2D([0], [0], color="#555", lw=1.5, ls="--", label="-- Physical"),
        ]
        stim_handles = ax2.get_legend_handles_labels()[0]
        stim_labels  = ax2.get_legend_handles_labels()[1]
        ax2.legend(
            handles=stim_handles + style_handles,
            labels=stim_labels + ["— Sensory", "-- Physical"],
            fontsize=8, ncol=1, framealpha=0.9,
        )

    fig.suptitle("Convergence Time — Empirical CDF", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, os.path.join(output_dir, "5_convergence_ecdf.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Fuzzy w_I inference surface
# ══════════════════════════════════════════════════════════════════════════════

def plot_fuzzy_wi_surface(
    output_dir: str = "figures",
) -> str:
    """
    Superficie de inferencia del sistema difuso w_I(density, distance).

    Evalúa compute_wi() en una malla de 50×50 puntos para n_stimuli = 1
    y n_stimuli = 2, generando dos heatmaps que visualizan el comportamiento
    del controlador difuso.  Esta figura es estándar en papers de lógica difusa
    para demostrar que las reglas producen una superficie suave y justificable.

    No requiere datos de simulación — calcula la superficie analíticamente.

    Args:
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)

    from .fuzzy_influence import compute_wi, W_I_MIN, W_I_MAX

    N  = 50
    ds = np.linspace(0.0, 1.0, N)   # density
    di = np.linspace(0.0, 1.0, N)   # normalised distance
    DD, DI = np.meshgrid(ds, di)

    wi1 = np.zeros((N, N))
    wi2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            wi1[i, j] = compute_wi(DD[i, j], DI[i, j], n_stimuli=1)
            wi2[i, j] = compute_wi(DD[i, j], DI[i, j], n_stimuli=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, surface, title in [
        (ax1, wi1, "Single stimulus detected (n=1)"),
        (ax2, wi2, "Multiple stimuli detected (n≥2)"),
    ]:
        im = ax.imshow(surface, origin="lower", aspect="auto",
                       extent=[0, 1, 0, 1], cmap="RdYlGn_r",
                       vmin=W_I_MIN, vmax=W_I_MAX)
        cs = ax.contour(ds, di, surface, levels=6, colors="white",
                        alpha=0.5, linewidths=0.8)
        ax.clabel(cs, fmt="%.2f", fontsize=7, colors="white")
        ax.set_xlabel("Neighbor density (E1)")
        ax.set_ylabel("Normalised distance to stimulus (E2)")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("w_I output", fontsize=9)

    fig.suptitle("Fuzzy Inference Surface — Adaptive w_I",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, os.path.join(output_dir, "6_fuzzy_wi_surface.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Collective localization timeline
# ══════════════════════════════════════════════════════════════════════════════

def plot_localization_timeline(
    results:    dict,
    output_dir: str = "figures",
) -> str:
    """
    Diagrama tipo Gantt de las fases de localización colectiva por estímulo.

    Divide el proceso de localización en tres fases para cada estímulo:
      Approach  : [0, first_arrival]           — ningún robot ha llegado aún.
      Build-up  : [first_arrival, conv_time]   — robots entrando en zona sensorial.
      Dwelling  : [conv_time, T]               — umbral superado, permanencia activa.

    Los intervalos se basan en la mediana a través de las réplicas para dar una
    representación robusta.  Las barras de error horizontales muestran el IQR
    de cada transición.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    all_metrics = results["all"]
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)
    T           = results["config"]["iterations"]

    fig, ax = plt.subplots(figsize=(10, max(3, n_stim * 1.4 + 1.5)))

    phase_colors = {
        "Approach": "#D5E8D4",
        "Build-up": "#FFE6CC",
        "Dwelling": "#DAE8FC",
    }
    phase_edge = {
        "Approach": "#82B366",
        "Build-up": "#D6B656",
        "Dwelling": "#6C8EBF",
    }

    yticks, ylabels = [], []

    for k in range(n_stim - 1, -1, -1):   # top-to-bottom display
        y_pos = n_stim - 1 - k
        col   = STIM_COLORS[k % len(STIM_COLORS)]

        fa_vals  = np.array([m["first_arrival"][k]                         for m in all_metrics])
        ct_vals  = np.array([m["convergence_time_per_stimulus"][k]         for m in all_metrics])
        pct_vals = np.array([m["physical_convergence_time_per_stimulus"][k] for m in all_metrics])

        fa_med  = np.median(fa_vals)
        ct_med  = np.median(ct_vals)
        pct_med = np.median(pct_vals)

        fa_iqr  = np.percentile(fa_vals,  [25, 75])
        ct_iqr  = np.percentile(ct_vals,  [25, 75])
        pct_iqr = np.percentile(pct_vals, [25, 75])

        bar_h = 0.55

        # Phase 1: Approach
        ax.barh(y_pos, fa_med, left=0, height=bar_h,
                color=phase_colors["Approach"], edgecolor=phase_edge["Approach"],
                lw=1.2, zorder=2)
        # Phase 2: Build-up
        ax.barh(y_pos, ct_med - fa_med, left=fa_med, height=bar_h,
                color=phase_colors["Build-up"], edgecolor=phase_edge["Build-up"],
                lw=1.2, zorder=2)
        # Phase 3: Dwelling
        ax.barh(y_pos, pct_med - ct_med, left=ct_med, height=bar_h,
                color=phase_colors["Dwelling"], edgecolor=phase_edge["Dwelling"],
                lw=1.2, zorder=2)
        # Remaining time as light grey
        ax.barh(y_pos, T - pct_med, left=pct_med, height=bar_h,
                color="#F5F5F5", edgecolor="#CCCCCC", lw=0.8, zorder=2)

        # IQR error bars on key transitions
        for x_med, x_iqr, marker_col in [
            (fa_med,  fa_iqr,  phase_edge["Approach"]),
            (ct_med,  ct_iqr,  phase_edge["Build-up"]),
            (pct_med, pct_iqr, phase_edge["Dwelling"]),
        ]:
            ax.errorbar(x_med, y_pos,
                        xerr=[[x_med - x_iqr[0]], [x_iqr[1] - x_med]],
                        fmt="none", ecolor=marker_col, elinewidth=2,
                        capsize=4, capthick=1.5, zorder=4)

        # Stimulus colour dot
        ax.scatter([0], [y_pos], color=col, s=60, zorder=5,
                   edgecolors="white", lw=1.2)

        yticks.append(y_pos)
        ylabels.append(_stim_label(k, stimuli))

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, T)
    ax.set_title("Collective Localization Timeline (median across replicas, IQR error bars)")

    legend_patches = [
        mpatches.Patch(fc=phase_colors[p], ec=phase_edge[p], lw=1.2, label=p)
        for p in ("Approach", "Build-up", "Dwelling")
    ] + [mpatches.Patch(fc="#F5F5F5", ec="#CCC", lw=0.8, label="Post-convergence")]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle("Collective Localization Timeline", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(left=0.22)
    return _save(fig, os.path.join(output_dir, "7_localization_timeline.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — RAOI behavioural state evolution
# ══════════════════════════════════════════════════════════════════════════════

def plot_raoi_state_evolution(
    results:    dict,
    reports:    list,
    output_dir: str = "figures",
) -> str:
    """
    Evolución temporal de los estados conductuales RAOI como área apilada.

    Para cada iteración calcula la fracción de robots en cada uno de los cinco
    estados (Libre, Repulsión, Atracción, Orientación, Influencia) y traza el
    promedio sobre todas las réplicas.  Este gráfico verifica directamente que
    el modelo funciona como se diseñó: la fracción de robots en estado Influencia
    debe crecer conforme el enjambre converge hacia los estímulos.

    Requiere reports[:,,:,7] — columna de estado RAOI del reporte de simulación.

    Args:
        results:    Dict devuelto por statistical_run().
        reports:    Lista de arrays report, shape (T, N, ≥8).
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo PNG guardado.
    """
    _ensure_dir(output_dir)
    T  = reports[0].shape[0]
    N  = reports[0].shape[1]
    t  = np.arange(T)

    # State index mapping: 0=free, 1=repulsion, 2=attraction, 3=orientation, 4=influence
    n_states   = 5
    state_frac = np.zeros((len(reports), T, n_states))

    for r, rep in enumerate(reports):
        states = rep[:, :, 7].astype(int)   # (T, N)
        for s in range(n_states):
            state_frac[r, :, s] = (states == s).sum(axis=1) / N

    mean_frac = state_frac.mean(axis=0)   # (T, n_states)

    # Smooth with a small rolling average to reduce noise
    window = max(1, T // 40)
    kernel = np.ones(window) / window
    smoothed = np.array([np.convolve(mean_frac[:, s], kernel, mode="same")
                         for s in range(n_states)]).T  # (T, n_states)
    # Re-normalise rows to 1 after smoothing
    row_sums = smoothed.sum(axis=1, keepdims=True)
    smoothed = smoothed / np.maximum(row_sums, 1e-9)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    baseline = np.zeros(T)
    for s in range(n_states):
        ax.fill_between(t, baseline, baseline + smoothed[:, s],
                        color=_STATE_COLORS[s], alpha=0.82,
                        label=_STATE_LABELS[s])
        baseline += smoothed[:, s]

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of robots")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.0)
    ax.set_title("RAOI Behavioural State Distribution Over Time")

    # Add std band for Influence state only (most relevant for the paper)
    inf_std = state_frac[:, :, 4].std(axis=0)
    inf_mean = mean_frac[:, 4]
    # Recompute cumulative baseline for influence
    cum_below = mean_frac[:, :4].sum(axis=1)
    ax.fill_between(t,
                    cum_below - inf_std,
                    cum_below + inf_std,
                    color=_STATE_COLORS[4], alpha=0.18)

    handles = [mpatches.Patch(fc=_STATE_COLORS[s], alpha=0.85,
                               label=_STATE_LABELS[s])
               for s in range(n_states)]
    handles.append(mpatches.Patch(fc=_STATE_COLORS[4], alpha=0.2,
                                   label="Influence ± 1 std"))
    fig.legend(handles=handles, loc="lower center", fontsize=9,
               framealpha=0.95, ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    return _save(fig, os.path.join(output_dir, "8_raoi_state_evolution.png"))


# ══════════════════════════════════════════════════════════════════════════════
# Main function — generate all figures
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(
    results:    dict,
    reports:    list,
    output_dir: str = "figures",
) -> list:
    """
    Genera las 8 figuras de análisis estadístico en una sola llamada.

    Las figuras que requieren los reportes completos (2, 3, 7, 8) se omiten
    automáticamente si reports está vacío, lo que permite usar esta función
    también desde el entry point de línea de comandos con un .npy que no
    incluye los arrays de posición completos.

    Args:
        results:    Dict devuelto por statistical_run().
        reports:    Lista de arrays report de cada réplica, shape (T, N, ≥8).
                    Puede ser [] si no están disponibles.
        output_dir: Carpeta donde se guardan los PNG.

    Returns:
        Lista de rutas de los archivos generados.
    """
    _ensure_dir(output_dir)
    print(f"\n  Generating figures in '{output_dir}/'...")

    paths = []
    paths.append(plot_convergence_distribution(results, output_dir))
    if reports:
        paths.append(plot_cohesion_evolution(results, reports, output_dir))
        paths.append(plot_fragmentation_evolution(results, reports, output_dir))
    paths.append(plot_stimulus_load_balance(results, output_dir))
    paths.append(plot_convergence_ecdf(results, output_dir))
    paths.append(plot_fuzzy_wi_surface(output_dir))
    paths.append(plot_localization_timeline(results, output_dir))
    if reports:
        paths.append(plot_raoi_state_evolution(results, reports, output_dir))

    print(f"  {len(paths)} figures saved in '{output_dir}/'")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — load a saved .npy and generate figures
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m raoi_simulator.plots <stat.npy> [output_dir]")
        sys.exit(1)

    npy_path   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"

    print(f"Loading results from: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()

    # Full position reports are not stored in the .npy to keep file size small.
    # Figures 2, 3, 7, 8 are skipped automatically when reports=[].
    generate_all(data, reports=[], output_dir=output_dir)