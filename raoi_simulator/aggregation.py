# -*- coding: utf-8 -*-
"""
Tarea de agregación con múltiples estímulos, peso difuso y obstáculos.

Implementa la tarea de agregación del modelo RAOI con tres capacidades
configurables de forma independiente:

  Múltiples estímulos de influencia:
    El enjambre responde a N fuentes simultáneas. Cada robot sigue al
    estímulo más cercano dentro de su radio de detección y campo de visión.
    Cuando los estímulos están suficientemente separados, el enjambre puede
    dividirse espontáneamente en subgrupos, uno por estímulo.

  Peso de influencia adaptativo (w_I difuso):
    En lugar de usar un peso de influencia constante, cada robot calcula
    su propio w_I en cada iteración según su densidad local de vecinos
    y su distancia al estímulo más cercano, usando el sistema difuso
    implementado en fuzzy_influence.py.

  Obstáculos estáticos:
    Cilindros bloqueantes distribuidos en el área. Se integran en la cadena
    de repulsión RAOI como fuentes de fuerza repulsiva virtual, y se aplica
    corrección de posición post-integración para garantizar que ningún robot
    penetra físicamente un obstáculo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León

Referencia:
    Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
    Using Influence Factor. J. Intelligent & Robotic Systems.
"""

import csv
import datetime
import math
import os
import random
import time
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from raoi_simulator import config
from raoi_simulator import metrics as mtr
from raoi_simulator import visualization as viz
from raoi_simulator.behavior import (
    combined_direction,
    detect_neighbors,
    repulsion_vector,
    select_voltage,
    wrap_angle,
)
from raoi_simulator.dynamics import DynamicsConstants, integrate_robot
from raoi_simulator.environment import detect_walls
from raoi_simulator.fuzzy_influence import (
    compute_density,
    compute_distance_norm,
    compute_wi,
)


# ══════════════════════════════════════════════════════════════════════════════
# Funciones internas
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_obstacle_collision(
    c_i:        np.ndarray,
    c_prev:     np.ndarray,
    obstacles:  list,
    area_limits: float,
) -> np.ndarray:
    """
    Corrige la posición de un robot que penetró un obstáculo circular.

    Se llama después de cada paso de integración. Si la nueva posición
    queda dentro del cuerpo de algún obstáculo, empuja al robot hasta
    el borde del obstáculo en la dirección radial de escape y refleja
    su orientación para que no vuelva a avanzar hacia el interior.

    Args:
        c_i:         Estado propuesto por integrate_robot, shape (6,).
        c_prev:      Estado anterior al paso de integración, shape (6,).
        obstacles:   Lista de dicts {'x', 'y', 'r'} en metros.
        area_limits: Lado del área cuadrada (m).

    Returns:
        Estado corregido, shape (6,). Igual a c_i si no hubo penetración.
    """
    c_out  = c_i.copy()
    rx, ry = float(c_out[0]), float(c_out[1])

    for obs in obstacles:
        ox, oy, orad = float(obs["x"]), float(obs["y"]), float(obs["r"])
        body     = float(config.ROBOT_BODY_RADIUS)
        dx, dy   = rx - ox, ry - oy
        dist     = math.sqrt(dx**2 + dy**2)
        min_dist = orad + body

        if dist >= min_dist:
            continue

        # Dirección de escape: radial hacia fuera del obstáculo
        if dist < 1e-9:
            angle  = random.uniform(0.0, 2 * math.pi)
            nx, ny = math.cos(angle), math.sin(angle)
        else:
            nx, ny = dx / dist, dy / dist

        # Sacar el robot al borde del obstáculo
        c_out[0] = ox + (min_dist + 0.005) * nx
        c_out[1] = oy + (min_dist + 0.005) * ny

        # Reflejar orientación si apunta hacia el interior del obstáculo
        vx = math.cos(c_out[3])
        vy = math.sin(c_out[3])
        if vx * nx + vy * ny < 0:
            vx -= 2 * (vx * nx + vy * ny) * nx
            vy -= 2 * (vx * nx + vy * ny) * ny
            norm     = max(math.sqrt(vx**2 + vy**2), 1e-9)
            c_out[3] = wrap_angle(
                math.atan2(vy / norm, vx / norm)
                + np.random.normal(0.0, 0.15)
            )

        rx, ry = float(c_out[0]), float(c_out[1])

    return c_out


def _detect_obstacles(
    pos:              np.ndarray,
    obstacles:        list,
    repulsion_radius: float,
) -> list:
    """
    Devuelve los puntos de superficie de los obstáculos dentro del radio
    de repulsión del robot.

    El punto devuelto por cada obstáculo es el punto más cercano sobre
    su borde. El formato es idéntico al de detect_walls(), de modo que
    behavior.repulsion_vector() los trata sin distinción.

    Args:
        pos:              Posición [x, y] del robot (m).
        obstacles:        Lista de dicts {'x', 'y', 'r'} (m).
        repulsion_radius: Radio de repulsión del robot (m).

    Returns:
        Lista de posiciones virtuales [[x, y], ...].
    """
    points = []
    rx, ry = float(pos[0]), float(pos[1])

    for obs in obstacles:
        ox, oy, orad = float(obs["x"]), float(obs["y"]), float(obs["r"])
        dx, dy = rx - ox, ry - oy
        dist   = math.sqrt(dx**2 + dy**2)

        if dist >= repulsion_radius + orad:
            continue

        if dist < 1e-9:
            angle = random.uniform(0.0, 2 * math.pi)
            points.append([ox + orad * math.cos(angle), oy + orad * math.sin(angle)])
        else:
            points.append([ox + orad * dx / dist, oy + orad * dy / dist])

    return points


def _detect_nearest_stimulus(
    robot_pos:        np.ndarray,
    robot_theta:      float,
    stimuli:          list,
    influence_radius: float,
    fov:              dict,
    n_repulsion:      int,
    n_walls:          int,
) -> tuple[float, float, float, int, int, int]:
    """
    Detecta los estímulos dentro del rango sensorial del robot y selecciona
    el más cercano.

    La detección ocurre cuando la zona sensorial del robot (radio r_I) intersecta
    el radio del estímulo (r_s): dist(robot, estímulo) ≤ r_I + r_s. Esto modela
    la percepción gradual — la señal es débil en el límite del rango y se
    intensifica conforme el robot se aproxima al estímulo.

    Solo se activa cuando no hay vecinos ni paredes en zona de repulsión.
    Cada robot decide de forma local e independiente, sin comunicación con
    el resto del enjambre.

    Args:
        robot_pos:        Posición [x, y] del robot (m).
        robot_theta:      Orientación del robot (rad).
        stimuli:          Lista de dicts {'x', 'y', 'r'} con las fuentes.
                          'r' es el radio de intensidad del estímulo (m).
        influence_radius: Radio sensorial del robot r_I (m).
        fov:              Dict de campos de visión (config.RAOI_FOV).
        n_repulsion:      Número de vecinos en zona de repulsión.
        n_walls:          Número de paredes detectadas.

    Returns:
        distance      : Distancia al estímulo seleccionado (m). 0 si ninguno.
        angle         : Ángulo hacia el estímulo con ruido gaussiano (rad).
        dist_norm     : Distancia normalizada ∈ [0, 1] → dist / (r_I + r_s).
        detected      : 1 si se detectó al menos un estímulo, 0 si no.
        stim_idx      : Índice del estímulo seleccionado, -1 si ninguno.
        n_detected    : Número total de estímulos detectados en este paso.
        effective_range: Rango efectivo r_I + r_s del estímulo seleccionado.
    """
    best_dist    = math.inf
    best_angle   = 0.0
    best_idx     = -1
    best_r_s     = config.DEFAULT_STIMULUS_RADIUS
    n_detected   = 0

    for k, stim in enumerate(stimuli):
        dx       = stim["x"] - robot_pos[0]
        dy       = stim["y"] - robot_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        angle    = wrap_angle(math.atan2(dy, dx))
        r_s      = float(stim.get("r", config.DEFAULT_STIMULUS_RADIUS))

        beta   = wrap_angle(angle - robot_theta)
        gamma  = wrap_angle(robot_theta - angle)
        i_diff = min(beta, gamma)

        # Detección: zona sensorial del robot toca el radio del estímulo
        if (i_diff < fov["fov_influence"] / 2
                and distance <= influence_radius + r_s
                and n_repulsion == 0 and n_walls == 0):
            n_detected += 1
            if distance < best_dist:
                best_dist  = distance
                best_angle = angle
                best_idx   = k
                best_r_s   = r_s

    if best_idx == -1:
        return 0.0, 0.0, 1.0, 0, -1, 0, influence_radius + config.DEFAULT_STIMULUS_RADIUS

    effective_range = influence_radius + best_r_s
    noise           = np.random.normal(0.0, config.INFLUENCE_NOISE_AMP)
    dist_norm       = compute_distance_norm(best_dist, influence_radius, best_r_s)
    return (best_dist, wrap_angle(best_angle + noise),
            dist_norm, 1, best_idx, n_detected, effective_range)


# ══════════════════════════════════════════════════════════════════════════════
# Simulación
# ══════════════════════════════════════════════════════════════════════════════

def run(
    iterations:        int,
    individuals:       int,
    r_r:               float,
    o_r:               float,
    a_r:               float,
    i_r:               float,
    stimuli:           Optional[list]                    = None,
    obstacles:         Optional[list]                    = None,
    use_fuzzy:         bool                              = True,
    animation:         bool                              = False,
    seed:              Optional[int]                     = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Ejecuta la simulación de agregación.

    Args:
        iterations:        Número de pasos de tiempo.
        individuals:       Número de robots.
        r_r:               Radio de repulsión (m), sumado a ROBOT_BODY_RADIUS.
        o_r:               Radio de orientación (m), sumado a ROBOT_BODY_RADIUS.
        a_r:               Radio de atracción (m), sumado a ROBOT_BODY_RADIUS.
        i_r:               Radio de detección de estímulos (m).
        stimuli:           Lista de dicts {'x', 'y'} con las fuentes de influencia.
                           None → usa config.STIMULI.
        obstacles:         Lista de dicts {'x', 'y', 'r'} con obstáculos circulares.
                           None → sin obstáculos.
        use_fuzzy:         True: w_I calculado por robot con lógica difusa.
                           False: w_I constante según config.RAOI_WEIGHTS.
        animation:         Si True, reproduce animación Pygame al terminar.
        seed:              Semilla aleatoria. None → config.SEED.
        progress_callback: Función f(t, total) llamada en cada iteración.

    Returns:
        report  : Estado del enjambre, shape (T, N, 9).
                    Columnas 0–6: x, y, z, θ(rad), θ(grados), v, ω.
                    Columna 7:    estado RAOI activo
                                  (0=libre, 1=repulsión, 2=atracción,
                                   3=orientación, 4=influencia).
                    Columna 8:    índice del estímulo seguido (-1=libre).
        data    : Estadísticas en 3 snapshots, shape (3, 5).
        metrics : Dict con métricas por estímulo, globales y extendidas.
    """
    _seed = seed if seed is not None else config.SEED
    random.seed(_seed)
    np.random.seed(_seed)

    area_limits   = config.AREA_LIMITS
    weights_base  = config.RAOI_WEIGHTS.copy()
    fov           = config.RAOI_FOV
    voltages      = config.VOLTAGE
    r_repulsion   = config.ROBOT_BODY_RADIUS + r_r
    r_orientation = config.ROBOT_BODY_RADIUS + o_r
    r_attraction  = config.ROBOT_BODY_RADIUS + a_r
    r_influence   = i_r

    if stimuli is None:
        stimuli = config.STIMULI
    if obstacles is None:
        obstacles = []
    n_stimuli = len(stimuli)

    dyn        = DynamicsConstants()
    C          = np.zeros((individuals, 6))
    report     = np.zeros((iterations, individuals, 9))
    free_iters = np.zeros(individuals, dtype=int)
    dirExp     = np.zeros(individuals)

    # ── Posicionamiento inicial ───────────────────────────────────────────────
    # El lado de la zona de spawn se ajusta dinámicamente para garantizar
    # que N robots quepan con la separación mínima, independientemente de r_r.
    min_sep    = max(config.SPAWN_MIN_SEPARATION, 2.0 * r_repulsion)
    grid_n     = math.ceil(math.sqrt(individuals))
    min_side   = min_sep * grid_n * 1.15
    max_side   = area_limits * 0.55
    spawn_side = float(min(max(min_side, config.AGGREGATION_SPAWN_SIDE), max_side))
    spawn_side = min(spawn_side, area_limits - r_repulsion * 2)

    for i in range(individuals):
        if i == 0:
            C[i, 0] = random.uniform(0, spawn_side)
            C[i, 1] = random.uniform(0, spawn_side)
        else:
            placed = False
            for _ in range(config.SPAWN_MAX_ATTEMPTS):
                cx = random.uniform(0, spawn_side)
                cy = random.uniform(0, spawn_side)
                if all(math.sqrt((cx - C[j, 0])**2 + (cy - C[j, 1])**2) >= min_sep
                       for j in range(i)):
                    C[i, 0] = cx
                    C[i, 1] = cy
                    placed   = True
                    break
            if not placed:
                raise ValueError(
                    f"No se pudo ubicar al robot {i} con separación mínima "
                    f"{min_sep:.3f} m en zona {spawn_side:.2f}×{spawn_side:.2f} m. "
                    "Reduce individuals o reduce r_r."
                )
        C[i, 3]   = random.uniform(0, 2 * math.pi)
        dirExp[i] = C[i, 3]

    # ── Snapshots ─────────────────────────────────────────────────────────────
    snap_mid = min(round(iterations / 2), iterations - 2)
    data     = np.zeros((3, 5))

    # ── Loop de simulación ────────────────────────────────────────────────────
    for t in range(iterations):
        for i in range(individuals):
            wn      = random.random() * 0.01
            pos_i   = C[i, :2]
            theta_i = C[i, 3]

            # Fuentes de repulsión: paredes, obstáculos, vecinos
            wall_pts    = detect_walls(pos_i, r_repulsion, area_limits)
            obs_pts     = _detect_obstacles(pos_i, obstacles, r_repulsion)
            n_walls     = len(wall_pts)
            nbrs        = detect_neighbors(
                i, C, r_repulsion, r_orientation, r_attraction, fov, n_walls
            )
            all_rep_pts = nbrs["rep_neighbors"] + wall_pts + obs_pts
            rep_vx, rep_vy = repulsion_vector(pos_i, all_rep_pts)
            n_rep       = nbrs["n_rep"] + n_walls + len(obs_pts)

            # Estímulo más cercano detectable
            (inf_dist, inf_angle, dist_norm,
             inf_detected, stim_idx, n_stim_detected,
             effective_range) = _detect_nearest_stimulus(
                pos_i, theta_i, stimuli,
                r_influence, fov, n_rep, n_walls,
            )

            # w_I: difuso por robot o constante global
            if use_fuzzy and inf_detected:
                n_nbrs  = nbrs["n_rep"] + nbrs["n_ori"] + nbrs["n_att"]
                density = compute_density(n_nbrs, individuals)
                wi      = compute_wi(density, dist_norm, n_stim_detected)
            else:
                wi = weights_base["w_I"]

            # Composición de vectores RAOI con prioridad canónica
            active    = {}
            weights_i = {**weights_base, "w_I": wi}

            if n_rep > 0:
                norm_r = max(math.sqrt(rep_vx**2 + rep_vy**2), 1e-9)
                active["R"] = (rep_vx / norm_r, rep_vy / norm_r)
            else:
                if nbrs["n_ori"] > 0:
                    ox_m = sum(nbrs["ox"]) / nbrs["n_ori"]
                    oy_m = sum(nbrs["oy"]) / nbrs["n_ori"]
                    norm_o = max(math.sqrt(ox_m**2 + oy_m**2), 1e-9)
                    active["O"] = (ox_m / norm_o, oy_m / norm_o)
                if nbrs["n_att"] > 0:
                    ax_m = sum(nbrs["ax"]) / nbrs["n_att"]
                    ay_m = sum(nbrs["ay"]) / nbrs["n_att"]
                    norm_a = max(math.sqrt(ax_m**2 + ay_m**2), 1e-9)
                    active["A"] = (ax_m / norm_a, ay_m / norm_a)
                if inf_detected:
                    active["I"] = (math.cos(inf_angle), math.sin(inf_angle))

            # Exploración libre si no hay ningún vector activo
            if not active:
                free_iters[i] += 1
                if free_iters[i] > config.EXPLORE_FREE_ITERS:
                    dirExp[i] = wrap_angle(
                        dirExp[i] + np.random.normal(0.0, config.EXPLORE_TURN_NOISE)
                    )
                desired_theta = dirExp[i]
                robot_state   = 0
            else:
                free_iters[i] = 0
                dirExp[i]     = wrap_angle(
                    C[i, 3] + np.random.normal(0.0, config.DIREXP_RESET_NOISE)
                )
                desired_theta = combined_direction(theta_i, active, weights_i)
                robot_state   = (1 if "R" in active else
                                 3 if "O" in active else
                                 2 if "A" in active else 4)

            # Integración y corrección de colisiones
            # effective_range = r_I + r_s del estímulo seleccionado.
            # select_voltage escala el voltaje con este rango: señal débil
            # (robot en el límite) → voltaje bajo; señal intensa → voltaje alto.
            voltage  = select_voltage(
                active, desired_theta, theta_i,
                inf_dist, effective_range, wn, voltages,
            )
            c_prev   = C[i].copy()
            C[i], _  = integrate_robot(C[i], voltage, dyn, r_repulsion, area_limits)
            if obstacles:
                C[i] = _resolve_obstacle_collision(C[i], c_prev, obstacles, area_limits)

            # Registro
            report[t, i, 0] = C[i, 0]
            report[t, i, 1] = C[i, 1]
            report[t, i, 2] = C[i, 2]
            report[t, i, 3] = C[i, 3]
            report[t, i, 4] = math.degrees(C[i, 3])
            report[t, i, 5] = C[i, 4]
            report[t, i, 6] = C[i, 5]
            report[t, i, 7] = robot_state
            report[t, i, 8] = stim_idx

        # Snapshots estadísticos en t=0, t_medio y t_final
        if t in (0, snap_mid, iterations - 1):
            snap_idx = 0 if t == 0 else (1 if t == snap_mid else 2)
            s = mtr.snapshot_stats(report[t, :, :2])
            data[snap_idx] = [s["mean_x"], s["mean_y"], s["std_x"], s["std_y"], s["area"]]

        if progress_callback:
            progress_callback(t + 1, iterations)

    # ── Métricas ──────────────────────────────────────────────────────────────
    # compute_all calcula todas las métricas en una sola pasada.
    all_metrics = mtr.compute_all(
        report, stimuli, r_influence,
        obstacles=obstacles if obstacles else None,
        r_rep=r_repulsion,
    )

    if animation:
        viz.animate_report(
            report[:, :, :8],
            environment={
                "area_limits":      area_limits,
                "influence_radius": r_influence,
                "stimuli":          stimuli,
                "obstacles":        obstacles,
            },
            interval     = config.ANIMATION_INTERVAL,
            show_zones   = config.SHOW_ZONES,
            show_trail   = config.SHOW_TRAIL,
            trail_length = config.TRAIL_LENGTH,
            save_path    = config.VIDEO_SAVE_PATH,
            screen_size  = config.SCREEN_SIZE,
        )

    return report, data, all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Exportación de resultados
# ══════════════════════════════════════════════════════════════════════════════

def _timestamp() -> str:
    """Marca de tiempo en formato YYYYMMDD_HHMMSS para nombres de archivo."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_single_run(
    report:      np.ndarray,
    data:        np.ndarray,
    metrics:     dict,
    stimuli:     list,
    obstacles:   list,
    use_fuzzy:   bool,
    results_dir: str = "results",
) -> tuple:
    """
    Guarda los resultados de una corrida individual en disco.

    Genera tres archivos en results_dir/:
      run_<timestamp>_report.npy   — estado completo del enjambre, shape (T, N, 9).
      run_<timestamp>_config.npy   — configuración y métricas completas.
      run_<timestamp>_summary.csv  — fila única con las métricas globales.

    Args:
        report:      Estado del enjambre, shape (T, N, 9).
        data:        Snapshots estadísticos, shape (3, 5).
        metrics:     Dict devuelto por run().
        stimuli:     Lista de estímulos usados.
        obstacles:   Lista de obstáculos usados.
        use_fuzzy:   Si se activó w_I difuso.
        results_dir: Carpeta destino (se crea si no existe).

    Returns:
        Tupla (report_path, config_path, csv_path).
    """
    os.makedirs(results_dir, exist_ok=True)
    ts   = _timestamp()
    base = os.path.join(results_dir, f"run_{ts}")

    report_path = f"{base}_report.npy"
    config_path = f"{base}_config.npy"
    csv_path    = f"{base}_summary.csv"

    np.save(report_path, report)
    np.save(config_path, {
        "data": data, "metrics": metrics,
        "stimuli": stimuli, "obstacles": obstacles, "use_fuzzy": use_fuzzy,
    }, allow_pickle=True)

    g = metrics
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_stimuli", "n_obstacles", "use_fuzzy",
                    "convergence_time", "physical_convergence_time", 
                    "cohesion_mean", "cohesion_final",
                    "fragmentation_mean", "fragmentation_final",
                    "distribution_entropy",
                    "dwell_count", "stimulus_occupancy",
                    "first_arrival", "physical_convergence_time_per_stimulus", 
                    "mean_robots_at_stimulus",
                    "transit_fraction", "obstacle_interaction_rate"])
        w.writerow([
            len(stimuli), len(obstacles), int(use_fuzzy),
            g.get("convergence_time", ""),
            g.get("physical_convergence_time", ""),
            g.get("cohesion_mean", ""), g.get("cohesion_final", ""),
            g.get("fragmentation_mean", ""), g.get("fragmentation_final", ""),
            g.get("distribution_entropy", ""),
            str(g.get("dwell_count", "")),
            str(g.get("stimulus_occupancy", "")),
            str(g.get("first_arrival", "")),
            g.get("physical_convergence_time_per_stimulus", ""),
            str(g.get("mean_robots_at_stimulus", "")),
            g.get("transit_fraction", ""),
            g.get("obstacle_interaction_rate", ""),
        ])

    print(f"\n  Guardado:")
    print(f"    {report_path}")
    print(f"    {config_path}")
    print(f"    {csv_path}")
    return report_path, config_path, csv_path

def _next_experiment_dir(base_dir: str = "results") -> str:
    """
    Devuelve la ruta de la siguiente carpeta de experimento dentro de base_dir.

    Busca subcarpetas con nombre numérico (1, 2, 3, ...) y devuelve
    base_dir/<N+1>, donde N es el máximo encontrado. Si no hay ninguna,
    devuelve base_dir/1.

    Args:
        base_dir: Carpeta raíz de resultados (p. ej. "results").

    Returns:
        Ruta de la nueva carpeta de experimento (aún no creada).
    """
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        int(d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()
    ]
    next_n = max(existing, default=0) + 1
    return os.path.join(base_dir, str(next_n))

def _save_statistical_run(
    result_dict: dict,
    results_dir: str = "results",
) -> tuple:
    """
    Guarda los resultados de una corrida estadística en disco.

    Genera dos archivos en results_dir/:
      stat_<timestamp>.csv — una fila por réplica con todas las métricas.
      stat_<timestamp>.npy — dict completo con configuración y métricas brutas.

    Args:
        result_dict: Dict devuelto por statistical_run().
        results_dir: Carpeta destino (se crea si no existe).

    Returns:
        Tupla (csv_path, npy_path).
    """
    os.makedirs(results_dir, exist_ok=True)
    ts       = _timestamp()
    csv_path = os.path.join(results_dir, f"stat_{ts}.csv")
    npy_path = os.path.join(results_dir, f"stat_{ts}.npy")

    cfg_used = result_dict["config"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["replica", "n_stimuli", "n_obstacles", "use_fuzzy",
                    "convergence_time", "physical_convergence_time", 
                    "cohesion_mean", "cohesion_final",
                    "fragmentation_mean", "fragmentation_final",
                    "distribution_entropy",
                    "dwell_count", "stimulus_occupancy",
                    "first_arrival", "physical_convergence_time_per_stimulus", 
                    "mean_robots_at_stimulus",
                    "transit_fraction", "obstacle_interaction_rate"])
        for rep, m in enumerate(result_dict["all"]):
            g = m
            w.writerow([
                rep,
                len(cfg_used["stimuli"]), len(cfg_used["obstacles"]),
                int(cfg_used["use_fuzzy"]),
                g.get("convergence_time", ""),
                g.get("physical_convergence_time", ""),
                g.get("cohesion_mean", ""), g.get("cohesion_final", ""),
                g.get("fragmentation_mean", ""), g.get("fragmentation_final", ""),
                g.get("distribution_entropy", ""),
                str(g.get("dwell_count", "")),
                str(g.get("stimulus_occupancy", "")),
                str(g.get("first_arrival", "")),
                str(g.get("physical_convergence_time_per_stimulus", "")),
                str(g.get("mean_robots_at_stimulus", "")),
                g.get("transit_fraction", ""),
                g.get("obstacle_interaction_rate", ""),
            ])

    np.save(npy_path, result_dict, allow_pickle=True)
    print(f"\n  Guardado:")
    print(f"    {csv_path}")
    print(f"    {npy_path}")
    return csv_path, npy_path


# ══════════════════════════════════════════════════════════════════════════════
# Entrada interactiva
# ══════════════════════════════════════════════════════════════════════════════

def _ask_int(prompt: str, default: int) -> int:
    """Solicita un entero por consola con valor por defecto."""
    while True:
        try:
            raw = input(f"  {prompt} [default {default}]: ").strip()
            return int(raw) if raw else default
        except ValueError:
            print("    Ingresa un número entero válido.")


def _ask_float(prompt: str, default: float) -> float:
    """Solicita un número decimal por consola con valor por defecto."""
    while True:
        try:
            raw = input(f"  {prompt} [default {default}]: ").strip()
            return float(raw) if raw else default
        except ValueError:
            print("    Ingresa un número decimal válido.")


def _ask_yn(prompt: str, default: bool) -> bool:
    """Solicita confirmación S/N por consola con valor por defecto."""
    ds = "S" if default else "N"
    while True:
        raw = input(f"  {prompt} [default {ds}] (S/N): ").strip().upper()
        if not raw:
            return default
        if raw in ("S", "SI", "SÍ", "Y", "YES"):
            return True
        if raw in ("N", "NO"):
            return False
        print("    Responde S o N.")


def _ask_stimuli(area: float) -> list:
    """
    Configura interactivamente las fuentes de influencia.

    Ofrece tres modos:
      1. Escenario predefinido (1–4 fuentes con posiciones distribuidas).
      2. Posiciones aleatorias (el usuario indica cuántas).
      3. Posiciones manuales (el usuario introduce x, y por cada fuente).

    Args:
        area: Lado del área cuadrada (m).

    Returns:
        Lista de dicts {'x': float, 'y': float}.
    """
    print("\n  ── Estímulos de influencia ──────────────────────────────────")
    print("    1. Escenario predefinido")
    print("    2. Posiciones aleatorias")
    print("    3. Posiciones manuales")

    while True:
        try:
            modo = int(input("  Modo [default 1]: ").strip() or "1")
            if modo in (1, 2, 3):
                break
            print("    Elige 1, 2 o 3.")
        except ValueError:
            print("    Ingresa 1, 2 o 3.")

    if modo == 1:
        print(f"\n    Escenarios predefinidos (área {area}×{area} m):")
        for k, s in config.STIMULI_SCENARIOS.items():
            coords = "  |  ".join(f"({st['x']},{st['y']})" for st in s)
            print(f"      {k} estímulo(s): {coords}")
        n = max(1, min(4, _ask_int("Número de estímulos (1–4)", 1)))
        return config.STIMULI_SCENARIOS[n]

    if modo == 2:
        n      = max(1, _ask_int("Número de estímulos", 2))
        margin = area * 0.1
        r_s    = max(0.1, _ask_float("radio de intensidad r_s para todos (m)", config.DEFAULT_STIMULUS_RADIUS))
        result = []
        for k in range(n):
            x = round(random.uniform(margin, area - margin), 2)
            y = round(random.uniform(margin, area - margin), 2)
            result.append({"x": x, "y": y, "r": r_s})
            print(f"    Estímulo {k+1}: ({x}, {y})  r_s={r_s}")
        return result

    n      = max(1, _ask_int("Número de estímulos", 1))
    result = []
    print(f"    Área: x ∈ [0, {area}]  y ∈ [0, {area}]")
    for k in range(n):
        print(f"\n    Estímulo {k+1}:")
        x   = float(np.clip(_ask_float("x", round(area * 0.75, 1)), 0, area))
        y   = float(np.clip(_ask_float("y", round(area * 0.75, 1)), 0, area))
        r_s = max(0.1, _ask_float("radio de intensidad r_s (m)", config.DEFAULT_STIMULUS_RADIUS))
        result.append({"x": x, "y": y, "r": r_s})
    return result


def _ask_obstacles(area: float) -> list:
    """
    Configura interactivamente los obstáculos estáticos.

    Ofrece cuatro modos:
      1. Sin obstáculos.
      2. Escenario predefinido (low / medium / high).
      3. Posiciones y radio aleatorios (el usuario indica cuántos).
      4. Posiciones manuales (el usuario introduce x, y, r por cada obstáculo).

    Args:
        area: Lado del área cuadrada (m).

    Returns:
        Lista de dicts {'x': float, 'y': float, 'r': float}.
        Lista vacía si se elige el modo 1.
    """
    print("\n  ── Obstáculos estáticos ─────────────────────────────────────")
    print("    1. Sin obstáculos")
    print("    2. Escenario predefinido (low / medium / high)")
    print("    3. Posiciones aleatorias")
    print("    4. Posiciones manuales")

    while True:
        try:
            modo = int(input("  Modo [default 1]: ").strip() or "1")
            if modo in (1, 2, 3, 4):
                break
            print("    Elige 1, 2, 3 o 4.")
        except ValueError:
            print("    Ingresa 1, 2, 3 o 4.")

    if modo == 1:
        return []

    if modo == 2:
        print("\n    Escenarios disponibles:")
        for key, obs_list in config.OBSTACLES_SCENARIOS.items():
            if key == "none":
                continue
            desc = "  |  ".join(f"({o['x']},{o['y']}) r={o['r']}" for o in obs_list)
            print(f"      {key:6s}: {len(obs_list)} obstáculos — {desc}")
        while True:
            raw = (input("  Escenario [low/medium/high, default medium]: ").strip().lower()
                   or "medium")
            if raw in config.OBSTACLES_SCENARIOS and raw != "none":
                return config.OBSTACLES_SCENARIOS[raw]
            print("    Elige low, medium o high.")

    if modo == 3:
        n      = max(1, _ask_int("Número de obstáculos", 3))
        rad    = max(0.1, _ask_float("Radio de cada obstáculo (m)", 0.4))
        margin = area * 0.1 + rad
        result = []
        for k in range(n):
            x = round(random.uniform(margin, area - margin), 2)
            y = round(random.uniform(margin, area - margin), 2)
            result.append({"x": x, "y": y, "r": rad})
            print(f"    Obstáculo {k+1}: ({x}, {y})  r={rad}")
        return result

    n      = max(1, _ask_int("Número de obstáculos", 2))
    result = []
    print(f"    Área: x ∈ [0, {area}]  y ∈ [0, {area}]")
    for k in range(n):
        print(f"\n    Obstáculo {k+1}:")
        x   = float(np.clip(_ask_float("x", round(area * 0.5, 1)), 0, area))
        y   = float(np.clip(_ask_float("y", round(area * 0.5, 1)), 0, area))
        rad = max(0.05, _ask_float("radio (m)", 0.4))
        result.append({"x": x, "y": y, "r": rad})
    return result


# ══════════════════════════════════════════════════════════════════════════════
# API pública
# ══════════════════════════════════════════════════════════════════════════════

def single_run(
    iterations:  Optional[int]   = None,
    individuals: Optional[int]   = None,
    r_r:         Optional[float] = None,
    o_r:         Optional[float] = None,
    a_r:         Optional[float] = None,
    i_r:         Optional[float] = None,
    stimuli:     Optional[list]  = None,
    obstacles:   Optional[list]  = None,
    use_fuzzy:   Optional[bool]  = None,
    animation:   bool            = False,
    seed:        Optional[int]   = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Ejecuta una simulación individual con configuración interactiva.

    Los parámetros pasados explícitamente se usan directamente; los que
    valen None se solicitan por consola. Esto permite usar la función tanto
    como punto de entrada interactivo como llamada programática.

    El flujo interactivo sigue este orden:
      1. Parámetros de simulación (iteraciones, robots, radios RAOI).
      2. Configuración de estímulos.
      3. Activación o desactivación de w_I difuso.
      4. Configuración de obstáculos.
      5. Reproducción de animación al terminar.
      6. Guardado de resultados en disco.

    Args:
        iterations:  Número de pasos. None → se pregunta.
        individuals: Número de robots. None → se pregunta.
        r_r:         Radio de repulsión (m). None → se pregunta.
        o_r:         Radio de orientación (m). None → se pregunta.
        a_r:         Radio de atracción (m). None → se pregunta.
        i_r:         Radio de influencia (m). None → se pregunta.
        stimuli:     Lista {'x','y'} de fuentes. None → se pregunta.
        obstacles:   Lista {'x','y','r'} de obstáculos. None → se pregunta.
        use_fuzzy:   True/False. None → se pregunta.
        animation:   Si True, reproduce animación al terminar sin preguntar.
        seed:        Semilla aleatoria. None → config.SEED.

    Returns:
        report, data, metrics  (ver run() para detalles).
    """
    area = config.AREA_LIMITS

    print("\n╔══════════════════════════════════════════════════╗")
    print("║      RAOI Swarm Simulator — Configuración        ║")
    print("╚══════════════════════════════════════════════════╝")

    print("\n  ── Parámetros de simulación ─────────────────────────────────")
    iters  = iterations  if iterations  is not None else _ask_int("Iteraciones",  config.DEFAULT_ITERATIONS)
    indivs = individuals if individuals is not None else _ask_int("Individuos",   config.DEFAULT_INDIVIDUALS)
    rr = r_r if r_r is not None else _ask_float("r_r — radio de repulsión (m)",   config.RAOI_RADII["r_repulsion"])
    ro = o_r if o_r is not None else _ask_float("r_o — radio de orientación (m)", config.RAOI_RADII["r_orientation"])
    ra = a_r if a_r is not None else _ask_float("r_a — radio de atracción (m)",   config.RAOI_RADII["r_attraction"])
    ri = i_r if i_r is not None else _ask_float("i_r — radio de influencia (m)",  config.DEFAULT_INFLUENCE_RADIUS)

    stims = stimuli   if stimuli   is not None else _ask_stimuli(area)

    print("\n  ── w_I difuso ───────────────────────────────────────────────")
    print("    Activado : cada robot calcula su propio w_I por iteración.")
    print("    Desactivado: w_I constante (config.RAOI_WEIGHTS['w_I']).")
    fuz = use_fuzzy if use_fuzzy is not None else _ask_yn("¿Activar w_I difuso?", True)

    obs = obstacles if obstacles is not None else _ask_obstacles(area)

    print("\n  ── Visualización ────────────────────────────────────────────")
    anim = animation if animation else _ask_yn("¿Reproducir animación al terminar?", False)

    print("\n  ── Guardado ─────────────────────────────────────────────────")
    do_save = _ask_yn("¿Guardar resultados en disco?", True)

    print("\n  ── Resumen ──────────────────────────────────────────────────")
    print(f"    Iteraciones  : {iters}")
    print(f"    Robots       : {indivs}")
    print(f"    Radios RAOI  : r_r={rr}  r_o={ro}  r_a={ra}  i_r={ri}")
    print(f"    Estímulos    : {len(stims)}")
    for k, s in enumerate(stims):
        print(f"      [{k+1}] x={s['x']}  y={s['y']}")
    print(f"    w_I difuso   : {'sí' if fuz else 'no'}")
    print(f"    Obstáculos   : {len(obs)}")
    for k, o in enumerate(obs):
        print(f"      [{k+1}] x={o['x']}  y={o['y']}  r={o['r']}")

    input("\n  Presiona Enter para iniciar...")

    t0   = time.time()
    pbar = tqdm(total=iters, desc="Simulando", unit="iter", ncols=80)

    def _cb(t: int, total: int) -> None:
        pbar.update(1)

    report, data, metrics_out = run(
        iterations=iters, individuals=indivs,
        r_r=rr, o_r=ro, a_r=ra, i_r=ri,
        stimuli=stims, obstacles=obs,
        use_fuzzy=fuz, animation=anim,
        seed=seed, progress_callback=_cb,
    )
    pbar.close()

    elapsed = time.time() - t0
    print(f"\n  ── Resultados ───────────────────────────────────────────────")
    print(f"    Tiempo de ejecución : {elapsed:.2f} s")
    g = metrics_out
    print(f"\n    Convergencia:")
    print(f"      convergence_time         : {g.get('convergence_time','—')} iter")
    print(f"      physical_convergence_time: {g.get('physical_convergence_time','—')} iter")
    print(f"\n    Cohesión y fragmentación:")
    print(f"      cohesion_mean            : {g.get('cohesion_mean',0):.4f} m")
    print(f"      cohesion_final           : {g.get('cohesion_final',0):.4f} m")
    print(f"      fragmentation_mean       : {g.get('fragmentation_mean',0):.4f}")
    print(f"      fragmentation_final      : {g.get('fragmentation_final',0):.4f}")
    print(f"\n    Distribución:")
    print(f"      robots_per_stimulus      : {g.get('robots_per_stimulus','—')}")
    print(f"      distribution_entropy     : {g.get('distribution_entropy',0):.4f} bits")
    print(f"\n    Permanencia:")
    print(f"      dwell_count              : {g.get('dwell_count','—')}")
    print(f"      stimulus_occupancy       : {[round(v,3) for v in g.get('stimulus_occupancy',[])]}")
    print(f"      first_arrival            : {g.get('first_arrival','—')}")
    print(f"      mean_robots_at_stimulus  : {[round(v,2) for v in g.get('mean_robots_at_stimulus',[])]}")
    print(f"      transit_fraction         : {g.get('transit_fraction',0):.4f}")
    if "obstacle_interaction_rate" in g:
        print(f"\n    Obstáculos:")
        print(f"      obstacle_interaction_rate: {g['obstacle_interaction_rate']:.4f}")
    if len(stims) > 1:
        ct_per = g.get("convergence_time_per_stimulus", [])
        print(f"\n    Convergencia por estímulo:")
        for k in range(len(stims)):
            ct_k = ct_per[k] if k < len(ct_per) else "—"
            print(f"      [{k+1}] ({stims[k]['x']},{stims[k]['y']}): "
                  f"convergence_time={ct_k}")

        ct_per = g.get("physical_convergence_time_per_stimulus", [])
        print(f"\n    Convergencia por estímulo (física):")
        for k in range(len(stims)):
            ct_k = ct_per[k] if k < len(ct_per) else "—"
            print(f"      [{k+1}] ({stims[k]['x']},{stims[k]['y']}): "
                  f"physical_convergence_time={ct_k}")

    if do_save:
        _save_single_run(report, data, metrics_out, stims, obs, fuz)

    return report, data, metrics_out


def statistical_run(
    replicas:     int,
    iterations:   Optional[int]   = None,
    individuals:  Optional[int]   = None,
    r_r:          Optional[float] = None,
    o_r:          Optional[float] = None,
    a_r:          Optional[float] = None,
    i_r:          Optional[float] = None,
    stimuli:      Optional[list]  = None,
    obstacles:    Optional[list]  = None,
    use_fuzzy:    Optional[bool]  = None,
    save_results: bool            = True,
    results_dir:  str             = "results",
) -> dict:
    """
    Ejecuta múltiples réplicas de la simulación con la misma configuración.

    La configuración se solicita una sola vez (igual que single_run) y se
    reutiliza en todas las réplicas. Cada réplica usa semilla config.SEED + índice,
    garantizando independencia estadística y reproducibilidad exacta.

    Args:
        replicas:     Número de réplicas.
        save_results: Si True, guarda CSV y npy en results_dir al terminar.
        results_dir:  Carpeta de salida (se crea si no existe).
        El resto de parámetros son idénticos a single_run().

    Returns:
        Dict con:
          'mean'     : métricas globales promediadas sobre réplicas.
          'std'      : desviaciones estándar.
          'all'      : lista de dicts de métricas por réplica.
          'config'   : configuración usada.
          'runtime'  : tiempo total de ejecución (s).
          'csv_path' : ruta del CSV (si save_results=True).
          'npy_path' : ruta del npy (si save_results=True).
    """
    # Determinar carpeta de experimento numerada (results/1, results/2, ...)
    results_dir = _next_experiment_dir(results_dir)  # <-- NUEVO

    area = config.AREA_LIMITS

    print("\n╔══════════════════════════════════════════════════╗")
    print("║    RAOI Swarm Simulator — Corrida estadística    ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Réplicas: {replicas}")
    print(f"  Experimento: {results_dir}")  # <-- NUEVO

    print("\n  ── Parámetros de simulación ─────────────────────────────────")
    iters  = iterations  if iterations  is not None else _ask_int("Iteraciones",  config.DEFAULT_ITERATIONS)
    indivs = individuals if individuals is not None else _ask_int("Individuos",   config.DEFAULT_INDIVIDUALS)
    rr = r_r if r_r is not None else _ask_float("r_r — radio de repulsión (m)",   config.RAOI_RADII["r_repulsion"])
    ro = o_r if o_r is not None else _ask_float("r_o — radio de orientación (m)", config.RAOI_RADII["r_orientation"])
    ra = a_r if a_r is not None else _ask_float("r_a — radio de atracción (m)",   config.RAOI_RADII["r_attraction"])
    ri = i_r if i_r is not None else _ask_float("i_r — radio de influencia (m)",  config.DEFAULT_INFLUENCE_RADIUS)

    stims = stimuli   if stimuli   is not None else _ask_stimuli(area)
    fuz   = use_fuzzy if use_fuzzy is not None else _ask_yn("¿Activar w_I difuso?", True)
    obs   = obstacles if obstacles is not None else _ask_obstacles(area)

    print(f"\n  {len(stims)} estímulo(s) | w_I difuso={'sí' if fuz else 'no'} | {len(obs)} obstáculo(s)")
    input("  Presiona Enter para iniciar...")

    all_results = []
    all_reports = []      # reportes completos para la figura de fragmentación temporal
    t0   = time.time()
    pbar = tqdm(total=replicas, desc="Réplicas", unit="rep", ncols=80)

    for rep in range(replicas):
        report, _, m = run(
            iterations=iters, individuals=indivs,
            r_r=rr, o_r=ro, a_r=ra, i_r=ri,
            stimuli=stims, obstacles=obs,
            use_fuzzy=fuz, animation=False,
            seed=config.SEED + rep,
        )
        all_results.append(m)
        all_reports.append(report)
        pbar.set_postfix(rep=rep + 1)
        pbar.update(1)

    pbar.close()
    elapsed = time.time() - t0

    # ── Métricas escalares: promedio y desviación directos ──────────────────
    scalar_keys = [
        "convergence_time", "physical_convergence_time", 
        "cohesion_mean", "cohesion_final",
        "fragmentation_mean", "fragmentation_final",
        "distribution_entropy", "transit_fraction",
    ]
    for k in ("obstacle_interaction_rate", "detour_ratio"):
        if k in all_results[0]:
            scalar_keys.append(k)
    scalar_keys = [k for k in scalar_keys if k in all_results[0]]

    mean_g = {k: float(np.mean([r[k] for r in all_results])) for k in scalar_keys}
    std_g  = {k: float(np.std( [r[k] for r in all_results])) for k in scalar_keys}

    # ── Métricas de lista: promedio elemento a elemento por estímulo ──────────
    # Cada réplica devuelve una lista de longitud n_estimulos. Se promedian
    # por posición para obtener el comportamiento medio de cada estímulo.
    list_keys = [
        "convergence_time_per_stimulus", "robots_per_stimulus",
        "dwell_count", "stimulus_occupancy", "first_arrival",
        "physical_convergence_time_per_stimulus", "mean_robots_at_stimulus",
    ]
    mean_lists: dict = {}
    std_lists:  dict = {}
    for k in list_keys:
        if k not in all_results[0]:
            continue
        mat = np.array([r[k] for r in all_results], dtype=float)
        mean_lists[k] = mat.mean(axis=0).tolist()
        std_lists[k]  = mat.std(axis=0).tolist()

    print(f"\n  ── Estadísticas ({replicas} réplicas) ───────────────────────────")
    print(f"  Métricas globales:")
    for k in mean_g:
        print(f"    {k:40s}: {mean_g[k]:.4f} ± {std_g[k]:.4f}")
    if mean_lists:
        print(f"\n  Métricas por estímulo (media ± std):")
        for k, vals in mean_lists.items():
            stds = std_lists[k]
            pairs = "  |  ".join(
                f"[{i+1}] {vals[i]:.3f}±{stds[i]:.3f}" for i in range(len(vals))
            )
            print(f"    {k:40s}: {pairs}")
    print(f"\n  Tiempo total  : {elapsed:.2f} s  ({elapsed/replicas:.2f} s/réplica)")

    result_dict = {
        "mean":       mean_g,
        "std":        std_g,
        "mean_lists": mean_lists,
        "std_lists":  std_lists,
        "all":        all_results,
        "config": {
            "iterations": iters, "individuals": indivs,
            "r_r": rr, "r_o": ro, "r_a": ra, "i_r": ri,
            "stimuli": stims, "obstacles": obs,
            "use_fuzzy": fuz, "seed_base": config.SEED,
        },
        "runtime": elapsed,
    }

    if save_results:
        csv_path, npy_path = _save_statistical_run(result_dict, results_dir)
        result_dict["csv_path"] = csv_path
        result_dict["npy_path"] = npy_path

    # ── Gráficas estadísticas ──────────────────────────────────────────────────
    # Import local para evitar dependencia circular si plots.py importara
    # algún módulo del simulador en el futuro.
    from .plots import generate_all
    figures_dir = os.path.join(results_dir, "figures")
    result_dict["figures_dir"] = figures_dir
    generate_all(result_dict, all_reports, output_dir=figures_dir)

    return result_dict