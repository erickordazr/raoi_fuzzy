# -*- coding: utf-8 -*-
"""
Generación de gráficas de análisis para corridas estadísticas.

Recibe el dict devuelto por statistical_run() y genera un conjunto de
figuras que describen el comportamiento del enjambre sobre múltiples
réplicas. Cada figura se guarda como PNG en el directorio indicado.

Gráficas generadas:
  1. convergence_time         — distribución del tiempo de convergencia
                                (histograma + línea de media).
  2. cohesion_evolution       — boxplot de cohesion_mean por réplica y
                                curva temporal de cohesión promedio.
  3. fragmentation_evolution  — curva temporal del índice de fragmentación
                                (media ± std sobre réplicas).
  4. stimulus_occupancy       — tiempo de permanencia promedio por estímulo
                                (barras con error).
  5. transit_fraction         — distribución de la fracción en tránsito
                                y proporción de presencia en estímulos.

Uso:
    from plots import generate_all

    results = statistical_run(replicas=30, ...)
    reports = [r["_report"] for r in results["all"]]  # si guardaste reports
    generate_all(results, output_dir="figures")

    # O desde línea de comandos cargando un .npy guardado:
    python plots.py results/stat_20260511_120000.npy

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin pantalla — compatible con todos los entornos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── Paleta de colores por estímulo ────────────────────────────────────────────
STIM_COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261", "#6A4C93"]

# ── Estilo base ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stim_label(k: int, stimuli: list) -> str:
    """Etiqueta compacta para el estímulo k."""
    s = stimuli[k]
    return f"Estímulo {k+1}  ({s['x']},{s['y']})  r_s={s.get('r',1.0)}"


# ══════════════════════════════════════════════════════════════════════════════
# Figura 1 — Distribución del tiempo de convergencia
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_time(results: dict, output_dir: str = "figures") -> str:
    """
    Histograma de convergence_time sobre todas las réplicas.

    Muestra la distribución del número de iteraciones que tardó el enjambre
    en localizar el estímulo (centroide de todos los estímulos como referencia).
    La línea vertical indica la media. Si hay múltiples estímulos, también se
    muestra el histograma por estímulo individual.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo guardado.
    """
    _ensure_dir(output_dir)
    all_metrics = results["all"]
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)

    # Datos globales
    ct_global = [m["convergence_time"] for m in all_metrics]

    # Datos por estímulo
    ct_per = []
    for k in range(n_stim):
        ct_per.append([m["convergence_time_per_stimulus"][k] for m in all_metrics])

    n_cols = 1 + n_stim if n_stim > 1 else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    # Histograma global
    ax = axes[0]
    ax.hist(ct_global, bins="auto", color="#378ADD", alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(ct_global), color="#E63946", lw=2, label=f"Media = {np.mean(ct_global):.1f}")
    ax.axvline(np.median(ct_global), color="#F4A261", lw=1.5, ls="--",
               label=f"Mediana = {np.median(ct_global):.1f}")
    ax.set_title("Tiempo de convergencia global")
    ax.set_xlabel("Iteraciones")
    ax.set_ylabel("Réplicas")
    ax.legend(fontsize=9)

    # Histograma por estímulo
    for k, ct_k in enumerate(ct_per):
        ax = axes[k + 1] if n_stim > 1 else None
        if ax is None:
            break
        col = STIM_COLORS[k % len(STIM_COLORS)]
        ax.hist(ct_k, bins="auto", color=col, alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(ct_k), color="#222222", lw=2,
                   label=f"Media = {np.mean(ct_k):.1f}")
        ax.set_title(_stim_label(k, stimuli))
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Réplicas")
        ax.legend(fontsize=9)

    fig.suptitle("Distribución del tiempo de convergencia", y=1.02)
    path = os.path.join(output_dir, "1_convergence_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figura 2 — Evolución de la cohesión
# ══════════════════════════════════════════════════════════════════════════════

def plot_cohesion(results: dict, output_dir: str = "figures") -> str:
    """
    Boxplot de cohesion_mean y cohesion_final por réplica.

    Muestra la dispersión de la cohesión del enjambre entre réplicas, separando
    el valor medio durante la simulación y el valor final. Un enjambre que
    converge bien tendrá cohesion_final notablemente menor que cohesion_mean.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo guardado.
    """
    _ensure_dir(output_dir)
    all_metrics = results["all"]

    cm = [m["cohesion_mean"]  for m in all_metrics]
    cf = [m["cohesion_final"] for m in all_metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        [cm, cf],
        labels=["cohesion_mean", "cohesion_final"],
        patch_artist=True,
        medianprops=dict(color="#222222", lw=2),
    )
    colors = ["#B5D4F4", "#9FE1CB"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)

    # Puntos individuales superpuestos
    for i, data in enumerate([cm, cf], 1):
        ax.scatter(
            np.random.normal(i, 0.06, len(data)),
            data, alpha=0.5, s=20, color="#444444", zorder=3,
        )

    ax.set_ylabel("Distancia al centroide (m)")
    ax.set_title("Cohesión del enjambre — media vs. final")
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "2_cohesion.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figura 3 — Evolución temporal de la fragmentación
# ══════════════════════════════════════════════════════════════════════════════

def plot_fragmentation(
    results: dict,
    reports: list,
    output_dir: str = "figures",
) -> str:
    """
    Curva temporal del índice de fragmentación, media ± std sobre réplicas.

    Requiere los reportes completos de cada réplica (shape T, N, ≥2).
    Muestra cómo evoluciona la fragmentación del enjambre iteración a iteración:
    si el enjambre se fragmenta al distribuirse entre estímulos o si mantiene
    cohesión a lo largo de la simulación.

    Args:
        results:    Dict devuelto por statistical_run().
        reports:    Lista de arrays report de cada réplica, cada uno (T, N, ≥2).
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo guardado.
    """
    _ensure_dir(output_dir)

    from raoi_simulator.metrics import fragmentation_index

    # Calcular índice de fragmentación por iteración para cada réplica
    frags = np.array([fragmentation_index(r) for r in reports])  # (replicas, T)
    T     = frags.shape[1]
    t     = np.arange(T)

    mean_f = frags.mean(axis=0)
    std_f  = frags.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, mean_f, color="#534AB7", lw=2, label="Media")
    ax.fill_between(t, mean_f - std_f, mean_f + std_f,
                    color="#534AB7", alpha=0.2, label="± 1 std")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Índice de fragmentación")
    ax.set_ylim(0, 1)
    ax.set_title("Evolución temporal de la fragmentación del enjambre")
    ax.legend(fontsize=9)

    path = os.path.join(output_dir, "3_fragmentation_evolution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figura 4 — Permanencia por estímulo
# ══════════════════════════════════════════════════════════════════════════════

def plot_stimulus_occupancy(results: dict, output_dir: str = "figures") -> str:
    """
    Barras comparativas de permanencia media por estímulo.

    Muestra para cada estímulo: fracción del tiempo ocupado (stimulus_occupancy)
    y robots medios presentes (mean_robots_at_stimulus), con barras de error.
    Permite identificar qué estímulos atraen y retienen más robots.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo guardado.
    """
    _ensure_dir(output_dir)
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)
    all_metrics = results["all"]

    # Matrices (replicas × n_stimuli)
    occ_mat = np.array([m["stimulus_occupancy"]     for m in all_metrics])
    mra_mat = np.array([m["mean_robots_at_stimulus"] for m in all_metrics])

    labels = [f"Estím. {k+1}\nr_s={stimuli[k].get('r',1.0)}" for k in range(n_stim)]
    x      = np.arange(n_stim)
    w      = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(7, n_stim * 2.5), 4))

    # Fracción de tiempo ocupado
    means1 = occ_mat.mean(axis=0)
    stds1  = occ_mat.std(axis=0)
    bars1  = ax1.bar(x, means1, width=w, yerr=stds1, capsize=5,
                     color=[STIM_COLORS[k % len(STIM_COLORS)] for k in range(n_stim)],
                     alpha=0.8, error_kw=dict(ecolor="#444", lw=1.5))
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Fracción del tiempo")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Fracción de tiempo con robots en el estímulo")

    # Robots medios simultáneos
    means2 = mra_mat.mean(axis=0)
    stds2  = mra_mat.std(axis=0)
    ax2.bar(x, means2, width=w, yerr=stds2, capsize=5,
            color=[STIM_COLORS[k % len(STIM_COLORS)] for k in range(n_stim)],
            alpha=0.8, error_kw=dict(ecolor="#444", lw=1.5))
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Número de robots")
    ax2.set_ylim(bottom=0)
    ax2.set_title("Robots medios simultáneos por estímulo")

    fig.suptitle("Permanencia del enjambre en los estímulos", y=1.02)
    path = os.path.join(output_dir, "4_stimulus_occupancy.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Figura 5 — Distribución del tiempo: estímulos vs. tránsito
# ══════════════════════════════════════════════════════════════════════════════

def plot_time_distribution(results: dict, output_dir: str = "figures") -> str:
    """
    Gráfica de tarta y barras apiladas mostrando cómo distribuye el enjambre
    su tiempo entre cada estímulo y el tránsito/exploración.

    Permite ver de un vistazo qué fracción del tiempo total pasó el enjambre
    en cada estímulo y cuánto tiempo estuvo en movimiento sin estar en ninguno.

    Args:
        results:    Dict devuelto por statistical_run().
        output_dir: Carpeta de salida.

    Returns:
        Ruta del archivo guardado.
    """
    _ensure_dir(output_dir)
    stimuli     = results["config"]["stimuli"]
    n_stim      = len(stimuli)
    all_metrics = results["all"]
    replicas    = len(all_metrics)

    # Fracción de tiempo en cada estímulo: dwell_count / (T * N)
    iters = results["config"]["iterations"]
    indiv = results["config"]["individuals"]
    total = iters * indiv

    dwell_mat = np.array([m["dwell_count"] for m in all_metrics], dtype=float)
    transit   = np.array([m["transit_fraction"] for m in all_metrics])

    # Fracciones medias
    dwell_fracs = dwell_mat.mean(axis=0) / max(total, 1)
    transit_mean = transit.mean()

    # Normalizar para que sumen 1 (puede haber pequeños solapamientos)
    total_frac = dwell_fracs.sum() + transit_mean
    dwell_fracs_n  = dwell_fracs  / max(total_frac, 1e-9)
    transit_mean_n = transit_mean / max(total_frac, 1e-9)

    fracs  = list(dwell_fracs_n) + [transit_mean_n]
    labels_pie = [f"Estím. {k+1}" for k in range(n_stim)] + ["Tránsito"]
    colors_pie = [STIM_COLORS[k % len(STIM_COLORS)] for k in range(n_stim)] + ["#CCCCCC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Tarta
    wedges, texts, autotexts = ax1.pie(
        fracs, labels=labels_pie, colors=colors_pie,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", lw=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax1.set_title("Distribución media del tiempo")

    # Barras apiladas por réplica
    dwell_norm = dwell_mat / max(total, 1)
    bottom     = np.zeros(replicas)
    for k in range(n_stim):
        ax2.bar(range(replicas), dwell_norm[:, k], bottom=bottom,
                color=STIM_COLORS[k % len(STIM_COLORS)], alpha=0.8,
                label=f"Estím. {k+1}")
        bottom += dwell_norm[:, k]
    ax2.bar(range(replicas), transit, bottom=bottom,
            color="#CCCCCC", alpha=0.8, label="Tránsito")

    ax2.set_xlabel("Réplica")
    ax2.set_ylabel("Fracción del tiempo")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Distribución del tiempo por réplica")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("Uso del tiempo del enjambre", y=1.02)
    path = os.path.join(output_dir, "5_time_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Guardada: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Función principal — genera todas las gráficas
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(
    results:    dict,
    reports:    list,
    output_dir: str = "figures",
) -> list:
    """
    Genera las 5 gráficas de análisis estadístico en una sola llamada.

    Args:
        results:    Dict devuelto por statistical_run().
        reports:    Lista de arrays report de cada réplica, shape (T, N, ≥2).
                    Requerido para la figura 3 (evolución de fragmentación).
        output_dir: Carpeta donde se guardan los PNG.

    Returns:
        Lista de rutas de los archivos generados.
    """
    _ensure_dir(output_dir)
    print(f"\n  Generando gráficas en '{output_dir}/'...")

    paths = []
    paths.append(plot_convergence_time(results, output_dir))
    paths.append(plot_cohesion(results, output_dir))
    if reports:
        paths.append(plot_fragmentation(results, reports, output_dir))
    paths.append(plot_stimulus_occupancy(results, output_dir))
    paths.append(plot_time_distribution(results, output_dir))

    print(f"  {len(paths)} gráficas guardadas en '{output_dir}/'")
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — carga un .npy guardado y genera las gráficas
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python plots.py <ruta_stat.npy> [output_dir]")
        sys.exit(1)

    npy_path   = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"

    print(f"Cargando resultados de: {npy_path}")
    results = np.load(npy_path, allow_pickle=True).item()

    # Los reportes completos no se guardan en el .npy estadístico por tamaño.
    # La figura 3 se omite si no están disponibles.
    generate_all(results, reports=[], output_dir=output_dir)
