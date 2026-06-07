# -*- coding: utf-8 -*-
"""
Métricas de desempeño del enjambre RAOI — tarea de agregación.

Centraliza todos los indicadores de evaluación en un único módulo.
Las funciones son stateless: reciben el reporte completo de la simulación
y devuelven escalares, arrays o diccionarios. No dependen del estado interno
del simulador y pueden usarse desde cualquier script de análisis externo.

Métricas de convergencia:
  convergence_time              — iteración en que el enjambre detecta el estímulo
                                  (rango r_I + r_s). Excluye t=0 (estado inicial).
  convergence_time_per_stimulus — convergence_time calculado por estímulo individual.
  physical_convergence_time     — iteración en que el porcentaje umbral del enjambre
                                  llega físicamente al cuerpo del estímulo (rango r_s).

Métricas de cohesión y fragmentación:
  cohesion_mean             — distancia media al centroide, promediada en el tiempo.
  cohesion_final            — distancia media al centroide en la última iteración.
  fragmentation_index       — fracción de pares de robots separados por encima del
                              umbral de fragmentación, calculada por iteración.
  fragmentation_mean        — promedio temporal del índice de fragmentación.
  fragmentation_final       — índice de fragmentación en la última iteración.

Métricas de distribución entre estímulos:
  distribution_entropy      — entropía de Shannon de la distribución final del
                              enjambre entre estímulos (bits).
  robots_per_stimulus       — conteo de robots en cada estímulo al final.

Métricas de permanencia:
  dwell_count               — iteraciones-robot dentro del radio de cada estímulo.
  stimulus_occupancy        — fracción del tiempo con al menos un robot en cada estímulo.
  first_arrival             — primera iteración en que un robot entró al radio del estímulo.
  mean_robots_at_stimulus   — número medio de robots simultáneamente dentro de cada estímulo.
  transit_fraction          — fracción del tiempo que los robots pasaron fuera de
                              cualquier estímulo (en tránsito o exploración).

Métricas de obstáculos:
  obstacle_interaction_rate — fracción de iteraciones-robot con repulsión activa
                              frente a algún obstáculo.
  detour_ratio              — alargamiento de trayectoria relativo a una línea base
                              sin obstáculos.

Función principal:
  compute_all               — calcula todas las métricas en una sola llamada.

Funciones auxiliares internas (prefijo _):
  _cohesion                 — cohesión de un instante.
  _swarm_area               — área de la elipse de desviaciones estándar.
  _path_length              — longitud de trayectoria por robot.
  snapshot_stats            — estadísticas de un instante para los snapshots canónicos.

Nota sobre la eliminación de f2, f3 y f4:
  La métrica f2 (área media) quedó reemplazada por la combinación de cohesion_mean
  y fragmentation_mean, que separan compacidad interna de separación entre subgrupos.
  La métrica f3 (máximo robots fuera de zona) quedó reemplazada por transit_fraction,
  que es un promedio temporal más estable que el peor caso puntual.
  La métrica f4 (número de robots) es un parámetro de configuración, no de desempeño.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León

Referencia:
  Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
  Using Influence Factor. J. Intelligent & Robotic Systems.
"""

import math
from typing import Optional

import numpy as np

from . import config


# ══════════════════════════════════════════════════════════════════════════════
# Funciones auxiliares internas
# ══════════════════════════════════════════════════════════════════════════════

def _cohesion(positions: np.ndarray) -> float:
    """
    Cohesión instantánea: distancia media de cada robot al centroide del enjambre.

    Args:
        positions: Posiciones de los robots en un instante, shape (N, 2).

    Returns:
        Distancia media al centroide (m). Valor bajo = enjambre compacto.
    """
    centroid  = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return float(np.mean(distances))


def _swarm_area(positions: np.ndarray) -> float:
    """
    Área estimada del enjambre como elipse de desviaciones estándar.

    Área = π · σ_x · σ_y. Usada internamente para el snapshot_stats.
    Acotada a config.AREA_COVERAGE_MAX_FRACTION del área total del escenario.

    Args:
        positions: Posiciones de los robots en un instante, shape (N, 2).

    Returns:
        Área estimada (m²).
    """
    centroid = np.mean(positions, axis=0)
    std_x    = math.sqrt(float(np.mean((positions[:, 0] - centroid[0])**2)))
    std_y    = math.sqrt(float(np.mean((positions[:, 1] - centroid[1])**2)))
    area     = math.pi * std_x * std_y
    max_area = (config.AREA_LIMITS**2) * config.AREA_COVERAGE_MAX_FRACTION
    return min(area, max_area)


def _path_length(report: np.ndarray) -> np.ndarray:
    """
    Longitud de trayectoria acumulada por cada robot a lo largo de la simulación.

    Args:
        report: Estado del enjambre, shape (T, N, ≥2).

    Returns:
        Array de shape (N,) con la distancia total recorrida por cada robot (m).
    """
    T, N, _ = report.shape
    lengths  = np.zeros(N)
    for t in range(1, T):
        dx = report[t, :, 0] - report[t-1, :, 0]
        dy = report[t, :, 1] - report[t-1, :, 1]
        lengths += np.sqrt(dx**2 + dy**2)
    return lengths


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de convergencia
# ══════════════════════════════════════════════════════════════════════════════

def convergence_time(
    report:             np.ndarray,
    influence_position: list,
    influence_radius:   float,
    threshold:          float = config.LOCALIZATION_THRESHOLD,
) -> int:
    """
    Iteración en que la fracción 'threshold' del enjambre alcanzó el estímulo.

    Un robot "alcanza" el estímulo cuando su distancia al centro es menor que
    influence_radius. Si el umbral nunca se alcanza, devuelve el número total
    de iteraciones como peor caso.

    Args:
        report:             Estado del enjambre, shape (T, N, ≥8).
        influence_position: Posición [x, y] del estímulo (m).
        influence_radius:   Radio efectivo de detección r_I + r_s (m).
        threshold:          Fracción mínima del enjambre requerida ∈ (0, 1].

    Returns:
        Iteración de convergencia ∈ [0, T]. T si nunca se alcanzó.
    """
    iterations, n_robots, _ = report.shape
    inf_pos = np.array(influence_position)

    for t in range(1, iterations):   # t=0 excluido: es el estado inicial, no convergencia
        distances = np.linalg.norm(report[t, :, :2] - inf_pos, axis=1)
        if np.sum(distances < influence_radius) / n_robots >= threshold:
            return t

    return iterations


def convergence_time_per_stimulus(
    report:   np.ndarray,
    stimuli:  list,
    i_r:      float,
    threshold: float = config.LOCALIZATION_THRESHOLD,
) -> list:
    """
    convergence_time calculado individualmente para cada estímulo.

    Útil cuando el enjambre se fragmenta: permite comparar la velocidad
    de convergencia a cada fuente por separado.

    Args:
        report:    Estado del enjambre, shape (T, N, ≥2).
        stimuli:   Lista de dicts {'x', 'y', 'r'}.
        i_r:       Radio sensorial del robot r_I (m).
        threshold: Fracción mínima del enjambre requerida.

    Returns:
        Lista de enteros, uno por estímulo. T si el umbral no se alcanzó.
    """
    return [
        convergence_time(
            report,
            [s["x"], s["y"]],
            i_r + float(s.get("r", 1.0)),
            threshold,
        )
        for s in stimuli
    ]



def physical_convergence_time_per_stimulus(
    report:    np.ndarray,
    stimuli:   list,
    threshold: float = config.LOCALIZATION_THRESHOLD,
) -> list:
    """
    Iteración en que una fracción del enjambre llega físicamente a cada estímulo.

    A diferencia de convergence_time (que usa el rango de detección r_I + r_s),
    esta métrica usa únicamente r_s como radio de referencia. Mide cuándo
    el enjambre no solo detecta el estímulo sino que llega a su cuerpo físico.

    Es la versión porcentual de first_arrival: mientras first_arrival registra
    cuándo llega el primer robot (1/N), physical_convergence_time registra
    cuándo llega la fracción 'threshold' del enjambre.

    Ejemplos de interpretación con threshold=0.5:
      - physical_convergence_time[k] = 80 → en la iteración 80 la mitad del
        enjambre estaba dentro del radio r_s del estímulo k.
      - physical_convergence_time[k] = T  → menos de la mitad del enjambre
        llegó físicamente al estímulo durante la simulación.

    Args:
        report:    Estado del enjambre, shape (T, N, ≥2).
        stimuli:   Lista de dicts {'x', 'y', 'r'}.
        threshold: Fracción mínima del enjambre requerida ∈ (0, 1].
                   Default: config.LOCALIZATION_THRESHOLD.

    Returns:
        Lista de enteros, uno por estímulo. T si el umbral no se alcanzó.
    """
    T, N, _ = report.shape
    results  = []

    for stim in stimuli:
        sx  = float(stim["x"])
        sy  = float(stim["y"])
        r_s = float(stim.get("r", 1.0))
        found = T

        for t in range(1, T):   # t=0 excluido: estado inicial, no convergencia
            distances = np.linalg.norm(report[t, :, :2] - np.array([sx, sy]), axis=1)
            if np.sum(distances < r_s) / N >= threshold:
                found = t
                break

        results.append(found)

    return results

def physical_convergence_time(
    report:    np.ndarray,
    stimuli:   list,
    threshold: float = config.LOCALIZATION_THRESHOLD,
) -> int:
    """
    Iteración en que la fracción 'threshold' del enjambre completo logró
    llegar físicamente a *cualquier* estímulo del escenario (unión de estados).

    Args:
        report:    Estado del enjambre, shape (T, N, ≥2).
        stimuli:   Lista de dicts {'x', 'y', 'r'}.
        threshold: Fracción mínima del enjambre requerida.

    Returns:
        Iteración de convergencia global ∈ [0, T]. T si nunca se alcanzó.
    """
    T, N, _ = report.shape

    for t in range(1, T):   # t=0 excluido
        robots_at_any_stim = 0
        for i in range(N):
            px = float(report[t, i, 0])
            py = float(report[t, i, 1])
            in_any = any(
                math.sqrt((px - float(s["x"]))**2 + (py - float(s["y"]))**2)
                < float(s.get("r", 1.0))
                for s in stimuli
            )
            if in_any:
                robots_at_any_stim += 1

        if robots_at_any_stim / N >= threshold:
            return t

    return T


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de cohesión y fragmentación
# ══════════════════════════════════════════════════════════════════════════════

def cohesion_mean(report: np.ndarray) -> float:
    """
    Distancia media de cada robot al centroide del enjambre, promediada en el tiempo.

    Mide qué tan unidos viajaron los robots durante toda la simulación.
    Un valor bajo indica enjambre compacto; alto indica dispersión crónica.

    Args:
        report: Estado del enjambre, shape (T, N, ≥2).

    Returns:
        Cohesión media (m).
    """
    cohs = np.array([_cohesion(report[t, :, :2]) for t in range(report.shape[0])])
    return float(np.mean(cohs))


def cohesion_final(report: np.ndarray) -> float:
    """
    Distancia media de cada robot al centroide en la última iteración.

    Complementa cohesion_mean: un enjambre puede estar disperso durante
    el viaje pero compacto al llegar. cohesion_final captura el estado final.

    Args:
        report: Estado del enjambre, shape (T, N, ≥2).

    Returns:
        Cohesión en t=T-1 (m).
    """
    return _cohesion(report[-1, :, :2])


def fragmentation_index(
    report: np.ndarray,
    d_min:  float = 3.0,
) -> np.ndarray:
    """
    Índice de fragmentación del enjambre en cada iteración.

    Definido como la fracción de pares de robots cuya distancia mutua supera
    d_min. Un enjambre cohesionado tiene índice ≈ 0; uno completamente
    fragmentado tiene índice ≈ 1.

    La distancia d_min se interpreta como la separación mínima a partir de
    la cual dos robots ya no pueden atraerse mutuamente con los radios RAOI
    configurados.

    Args:
        report: Estado del enjambre, shape (T, N, ≥2).
        d_min:  Umbral de separación (m). Default 3.0 m ≈ 1.5× r_attraction.

    Returns:
        Array de shape (T,) con el índice en cada iteración ∈ [0, 1].
    """
    T, N, _ = report.shape
    indices  = np.zeros(T)

    for t in range(T):
        pos   = report[t, :, :2]
        diff  = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        upper = dists[np.triu_indices(N, k=1)]
        if len(upper) > 0:
            indices[t] = float(np.mean(upper > d_min))

    return indices


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de distribución entre estímulos
# ══════════════════════════════════════════════════════════════════════════════

def distribution_entropy(
    report:    np.ndarray,
    stimuli:   list,
    threshold: float = 1.5,
) -> tuple:
    """
    Distribución final del enjambre entre estímulos y su entropía de Shannon.

    Evalúa la última iteración. Cada robot se asigna al estímulo más cercano
    si su distancia es menor que threshold; de lo contrario queda sin asignar.

    La entropía mide el equilibrio de la distribución:
      - Alta entropía: los robots se distribuyeron equitativamente entre fuentes.
      - Baja entropía: la mayoría convergió a un mismo estímulo.

    Args:
        report:    Estado del enjambre, shape (T, N, ≥2).
        stimuli:   Lista de dicts {'x', 'y', 'r'}.
        threshold: Radio de asignación (m).

    Returns:
        Tupla (robots_per_stimulus, entropy):
          robots_per_stimulus: lista de enteros, conteo por estímulo.
          entropy:             entropía de Shannon en bits.
    """
    final_pos   = report[-1, :, :2]
    n_robots    = final_pos.shape[0]
    counts      = np.zeros(len(stimuli), dtype=int)
    no_assigned = 0

    for i in range(n_robots):
        dists   = [math.sqrt((final_pos[i, 0] - s["x"])**2 + (final_pos[i, 1] - s["y"])**2)
                   for s in stimuli]
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= threshold:
            counts[min_idx] += 1
        else:
            no_assigned += 1

    fractions = counts / max(n_robots, 1)
    probs     = fractions[fractions > 0]
    entropy   = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    return counts.tolist(), entropy


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de permanencia en estímulos
# ══════════════════════════════════════════════════════════════════════════════

def dwell_count(
    report:  np.ndarray,
    stimuli: list,
) -> list:
    """
    Total de iteraciones-robot dentro del radio r_s de cada estímulo.

    Si 3 robots pasan 10 iteraciones dentro del estímulo 1, el valor es 30.
    Mide cuánto fue usado cada estímulo a lo largo de la simulación.

    Args:
        report:  Estado del enjambre, shape (T, N, ≥2).
        stimuli: Lista de dicts {'x', 'y', 'r'}.

    Returns:
        Lista de enteros, uno por estímulo.
    """
    T, N, _ = report.shape
    counts   = [0] * len(stimuli)

    for t in range(T):
        for k, stim in enumerate(stimuli):
            dx = report[t, :, 0] - float(stim["x"])
            dy = report[t, :, 1] - float(stim["y"])
            r_s = float(stim.get("r", 1.0))
            counts[k] += int(np.sum(np.sqrt(dx**2 + dy**2) < r_s))

    return counts


def stimulus_occupancy(
    report:  np.ndarray,
    stimuli: list,
) -> list:
    """
    Fracción del tiempo en que cada estímulo tuvo al menos un robot dentro de r_s.

    Un valor de 0.92 significa que el 92% de las iteraciones el estímulo estuvo
    "ocupado" por al menos un robot.

    Args:
        report:  Estado del enjambre, shape (T, N, ≥2).
        stimuli: Lista de dicts {'x', 'y', 'r'}.

    Returns:
        Lista de floats ∈ [0, 1], uno por estímulo.
    """
    T, N, _ = report.shape
    occupied = [0] * len(stimuli)

    for t in range(T):
        for k, stim in enumerate(stimuli):
            dx  = report[t, :, 0] - float(stim["x"])
            dy  = report[t, :, 1] - float(stim["y"])
            r_s = float(stim.get("r", 1.0))
            if np.any(np.sqrt(dx**2 + dy**2) < r_s):
                occupied[k] += 1

    return [o / max(T, 1) for o in occupied]


def first_arrival(
    report:  np.ndarray,
    stimuli: list,
) -> list:
    """
    Primera iteración en que un robot entró al radio r_s de cada estímulo.

    Diferente de convergence_time: convergence_time mide cuándo el porcentaje
    umbral del enjambre detecta el estímulo (entra a r_I + r_s). first_arrival
    mide cuándo el primer robot pisa físicamente el cuerpo del estímulo (entra a r_s).
    El estado inicial t=0 se excluye — se considera que el robot llegó al estímulo
    solo si se movió hacia él durante la simulación.

    Args:
        report:  Estado del enjambre, shape (T, N, ≥2).
        stimuli: Lista de dicts {'x', 'y', 'r'}.

    Returns:
        Lista de enteros, uno por estímulo. T si no se alcanzó ninguno.
    """
    T, N, _ = report.shape
    arrivals = [T] * len(stimuli)

    for t in range(1, T):   # t=0 excluido: estado inicial, no movimiento
        for k, stim in enumerate(stimuli):
            if arrivals[k] < T:
                continue
            dx  = report[t, :, 0] - float(stim["x"])
            dy  = report[t, :, 1] - float(stim["y"])
            r_s = float(stim.get("r", 1.0))
            if np.any(np.sqrt(dx**2 + dy**2) < r_s):
                arrivals[k] = t

    return arrivals


def mean_robots_at_stimulus(
    report:  np.ndarray,
    stimuli: list,
) -> list:
    """
    Número medio de robots simultáneamente dentro del radio r_s de cada estímulo.

    Promediado a lo largo de toda la simulación. Responde a: ¿cuántos robots
    "vivieron" en cada estímulo en promedio?

    Args:
        report:  Estado del enjambre, shape (T, N, ≥2).
        stimuli: Lista de dicts {'x', 'y', 'r'}.

    Returns:
        Lista de floats, uno por estímulo.
    """
    T, N, _ = report.shape
    means    = []

    for stim in stimuli:
        sx  = float(stim["x"])
        sy  = float(stim["y"])
        r_s = float(stim.get("r", 1.0))
        counts = np.zeros(T)
        for t in range(T):
            dx = report[t, :, 0] - sx
            dy = report[t, :, 1] - sy
            counts[t] = float(np.sum(np.sqrt(dx**2 + dy**2) < r_s))
        means.append(float(np.mean(counts)))

    return means


def transit_fraction(
    report:  np.ndarray,
    stimuli: list,
) -> float:
    """
    Fracción del tiempo en que los robots estuvieron fuera de todos los estímulos.

    Mide qué proporción de iteraciones-robot transcurrió sin que el robot estuviera
    dentro del radio r_s de ningún estímulo. Un valor alto indica que el enjambre
    pasó mucho tiempo en tránsito o exploración; uno bajo indica que los robots
    llegaron a los estímulos y permanecieron allí.

    Args:
        report:  Estado del enjambre, shape (T, N, ≥2).
        stimuli: Lista de dicts {'x', 'y', 'r'}.

    Returns:
        Fracción ∈ [0, 1].
    """
    T, N, _ = report.shape
    outside  = 0

    for t in range(T):
        for i in range(N):
            px = float(report[t, i, 0])
            py = float(report[t, i, 1])
            in_any = any(
                math.sqrt((px - float(s["x"]))**2 + (py - float(s["y"]))**2)
                < float(s.get("r", 1.0))
                for s in stimuli
            )
            if not in_any:
                outside += 1

    return outside / max(T * N, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Métricas de obstáculos
# ══════════════════════════════════════════════════════════════════════════════

def obstacle_interaction_rate(
    report:    np.ndarray,
    obstacles: list,
    r_rep:     float,
) -> float:
    """
    Fracción de iteraciones-robot con repulsión activa frente a algún obstáculo.

    Un robot "interactúa" con un obstáculo cuando entra en su zona de repulsión,
    es decir, cuando dist(robot, obstáculo) < r_rep + r_obstáculo. Mide qué tan
    perturbadores fueron los obstáculos para el comportamiento RAOI del enjambre.

    Args:
        report:    Estado del enjambre, shape (T, N, ≥2).
        obstacles: Lista de dicts {'x', 'y', 'r'} (m).
        r_rep:     Radio de repulsión efectivo del robot r_r + r_cuerpo (m).

    Returns:
        Fracción ∈ [0, 1]. 0 = ningún robot entró en zona de repulsión de obstáculos.
    """
    T, N, _ = report.shape
    count    = 0

    for t in range(T):
        for i in range(N):
            px, py = float(report[t, i, 0]), float(report[t, i, 1])
            for obs in obstacles:
                dist = math.sqrt((px - float(obs["x"]))**2 + (py - float(obs["y"]))**2)
                if dist < r_rep + float(obs["r"]):
                    count += 1
                    break

    return count / max(T * N, 1)


def detour_ratio(
    report_with_obstacles: np.ndarray,
    report_baseline:       np.ndarray,
) -> float:
    """
    Alargamiento de trayectoria causado por los obstáculos.

    Compara la longitud media de trayectoria entre una corrida con obstáculos
    y una de referencia sin obstáculos (misma semilla y configuración RAOI).
    Un ratio de 1.15 indica un 15% de recorrido adicional.

    Args:
        report_with_obstacles: Reporte con obstáculos, shape (T, N, ≥2).
        report_baseline:       Reporte sin obstáculos (misma semilla), shape (T, N, ≥2).

    Returns:
        Ratio ≥ 1.0. 1.0 = los obstáculos no alargaron el recorrido.
    """
    len_obs  = float(np.mean(_path_length(report_with_obstacles)))
    len_base = float(np.mean(_path_length(report_baseline)))
    if len_base < 1e-9:
        return 1.0
    return len_obs / len_base


# ══════════════════════════════════════════════════════════════════════════════
# Comparación entre configuraciones
# ══════════════════════════════════════════════════════════════════════════════

def adaptation_gain(ct_a: float, ct_b: float) -> float:
    """
    Ganancia relativa en convergence_time entre dos configuraciones.

    Positivo: la configuración A converge antes que B.
    Negativo: B converge antes que A.
    Útil para comparar w_I difuso vs. constante.

    Args:
        ct_a: convergence_time de la configuración A.
        ct_b: convergence_time de la configuración B (referencia).

    Returns:
        Ganancia relativa ∈ (-∞, 1].
    """
    if ct_b < 1e-9:
        return 0.0
    return float((ct_b - ct_a) / ct_b)


# ══════════════════════════════════════════════════════════════════════════════
# Función principal — calcula todo en una sola pasada
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(
    report:                np.ndarray,
    stimuli:               list,
    i_r:                   float,
    obstacles:             Optional[list]       = None,
    report_baseline:       Optional[np.ndarray] = None,
    r_rep:                 float                = config.ROBOT_BODY_RADIUS,
) -> dict:
    """
    Calcula todas las métricas disponibles en una sola llamada.

    Las métricas de obstáculos solo se calculan si obstacles no es None.
    detour_ratio solo se calcula si además report_baseline no es None.

    Args:
        report:          Estado del enjambre, shape (T, N, ≥2).
        stimuli:         Lista de dicts {'x', 'y', 'r'} con los estímulos.
        i_r:             Radio sensorial del robot r_I (m).
        obstacles:       Lista de dicts {'x', 'y', 'r'} con obstáculos. None = sin obstáculos.
        report_baseline: Reporte de referencia sin obstáculos para detour_ratio.
        r_rep:           Radio de repulsión efectivo r_r + r_cuerpo (m).

    Returns:
        Dict con las siguientes claves:

        Convergencia:
          'convergence_time'               : int — iteración de detección global (rango r_I+r_s).
          'convergence_time_per_stimulus'  : list[int] — detección por estímulo.
          "physical_convergence_time"                   : int — iteración de detección global (rango r_s), 
          'physical_convergence_time_per_stimulus'      : list[int] — llegada física al cuerpo del estímulo (rango r_s).

        Cohesión y fragmentación:
          'cohesion_mean'                  : float (m).
          'cohesion_final'                 : float (m).
          'fragmentation_mean'             : float ∈ [0, 1].
          'fragmentation_final'            : float ∈ [0, 1].

        Distribución:
          'robots_per_stimulus'            : list[int].
          'distribution_entropy'           : float (bits).

        Permanencia:
          'dwell_count'                    : list[int] (iter-robot).
          'stimulus_occupancy'             : list[float] ∈ [0, 1].
          'first_arrival'                  : list[int] (iteración).
          'mean_robots_at_stimulus'        : list[float].
          'transit_fraction'               : float ∈ [0, 1].

        Obstáculos (solo si obstacles no es None):
          'obstacle_interaction_rate'      : float ∈ [0, 1].
          'detour_ratio'                   : float ≥ 1.0 (solo si report_baseline no es None).
    """
    T, N, _ = report.shape

    # ── Convergencia ─────────────────────────────────────────────────────────
    # El radio efectivo de detección es r_I + r_s de cada estímulo.
    # Para la métrica global se usa el centroide de los estímulos con r_s promedio.
    r_s_mean = float(np.mean([float(s.get("r", 1.0)) for s in stimuli]))
    cx  = sum(s["x"] for s in stimuli) / len(stimuli)
    cy  = sum(s["y"] for s in stimuli) / len(stimuli)

    ct = convergence_time(report, [cx, cy], i_r + r_s_mean)

    # ── Cohesión y fragmentación ──────────────────────────────────────────────
    cohs = np.array([_cohesion(report[t, :, :2]) for t in range(T)])
    frag = fragmentation_index(report)

    # ── Distribución ─────────────────────────────────────────────────────────
    rps, entropy = distribution_entropy(report, stimuli)

    # ── Permanencia ──────────────────────────────────────────────────────────
    dc  = dwell_count(report, stimuli)
    occ = stimulus_occupancy(report, stimuli)
    fa  = first_arrival(report, stimuli)
    mra = mean_robots_at_stimulus(report, stimuli)
    tf  = transit_fraction(report, stimuli)

    out = {
        # Convergencia
        "convergence_time":              ct,
        "convergence_time_per_stimulus": convergence_time_per_stimulus(report, stimuli, i_r),
        "physical_convergence_time":     physical_convergence_time(report, stimuli),
        "physical_convergence_time_per_stimulus": physical_convergence_time_per_stimulus(report, stimuli),
        # Cohesión y fragmentación
        "cohesion_mean":                 float(np.mean(cohs)),
        "cohesion_final":                float(cohs[-1]),
        "fragmentation_mean":            float(np.mean(frag)),
        "fragmentation_final":           float(frag[-1]),
        # Distribución
        "robots_per_stimulus":           rps,
        "distribution_entropy":          entropy,
        # Permanencia
        "dwell_count":                   dc,
        "stimulus_occupancy":            occ,
        "first_arrival":                 fa,
        "mean_robots_at_stimulus":       mra,
        "transit_fraction":              tf,
    }

    # ── Obstáculos ────────────────────────────────────────────────────────────
    if obstacles:
        out["obstacle_interaction_rate"] = obstacle_interaction_rate(
            report, obstacles, r_rep
        )
        if report_baseline is not None:
            out["detour_ratio"] = detour_ratio(report, report_baseline)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades para la simulación
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_stats(positions: np.ndarray) -> dict:
    """
    Estadísticas de posición para un instante de tiempo único.

    Usada por el loop de simulación para los tres snapshots canónicos
    (t=0, t_medio, t_final).

    Args:
        positions: Posiciones de los robots, shape (N, 2).

    Returns:
        Dict con 'mean_x', 'mean_y', 'std_x', 'std_y', 'area'.
    """
    mean_x = float(np.mean(positions[:, 0]))
    mean_y = float(np.mean(positions[:, 1]))
    std_x  = math.sqrt(float(np.mean((positions[:, 0] - mean_x)**2)))
    std_y  = math.sqrt(float(np.mean((positions[:, 1] - mean_y)**2)))
    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x":  std_x,
        "std_y":  std_y,
        "area":   math.pi * std_x * std_y,
    }