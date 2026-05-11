# -*- coding: utf-8 -*-
"""
Sistema difuso para el peso de influencia w_I.

Calcula w_I por robot y por iteración a partir de tres variables locales:

  E1 — density   : fracción de vecinos detectados en las zonas R, O y A
                   respecto al total de robots del enjambre (N − 1).
  E2 — distance  : distancia al estímulo más cercano normalizada por el
                   rango efectivo de detección (r_I + r_s del estímulo).
  E3 — n_stimuli : número de estímulos detectados dentro del rango efectivo.
                   Cuando E3 = 0 el sistema difuso no se activa.

La salida w_I ∈ [W_I_MIN, W_I_MAX] pondera la componente de influencia I
en combined_direction(). W_I_MAX > w_o + w_a garantiza que un w_I alto
permita a la señal dominar sobre la cohesión grupal.

Lógica central:
  - Pocos vecinos + estímulo único cercano  → w_I alto  (señal clara, sin grupo).
  - Muchos vecinos + estímulo detectado     → w_I bajo  (el grupo orienta al robot).
  - Varios estímulos + pocos vecinos        → w_I alto  (decisión reactiva al más cercano).
  - Varios estímulos + muchos vecinos       → w_I bajo  (grupo absorbe la ambigüedad).

Implementa inferencia Mamdani con α-corte mínimo y defuzzificación por
centroide ponderado. Sin dependencias externas.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import numpy as np

# ── Rango de salida ───────────────────────────────────────────────────────────

W_I_MIN: float = 0.10
"""Peso mínimo de w_I. La señal de influencia nunca desaparece por completo."""

W_I_MAX: float = 1.20
"""
Peso máximo de w_I. Supera la suma w_o + w_a (típicamente 0.5 + 0.5 = 1.0),
garantizando que la señal de influencia domine la dirección de movimiento
cuando el sistema difuso asigna activación alta.
"""


# ── Funciones de membresía ────────────────────────────────────────────────────

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Función de membresía triangular.

    Vale 1.0 en el vértice b y 0.0 fuera del soporte [a, c].

    Args:
        x: Valor de entrada.
        a: Pie izquierdo.
        b: Vértice (membresía máxima).
        c: Pie derecho.

    Returns:
        Grado de membresía en [0, 1].
    """
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / max(b - a, 1e-9)
    return (c - x) / max(c - b, 1e-9)


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Función de membresía trapezoidal.

    Vale 1.0 en el núcleo [b, c] y 0.0 fuera del soporte [a, d].

    Args:
        x: Valor de entrada.
        a, b, c, d: Parámetros del trapecio (a ≤ b ≤ c ≤ d).

    Returns:
        Grado de membresía en [0, 1].
    """
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return (x - a) / max(b - a, 1e-9)
    if x <= c:
        return 1.0
    return (d - x) / max(d - c, 1e-9)


# ── Fuzzificación ─────────────────────────────────────────────────────────────

def _fuzzify_density(density: float) -> dict:
    """
    Fuzzifica la fracción de vecinos detectados en tres conjuntos lingüísticos.

    Los umbrales son fijos sobre [0, 1]. La normalización por (N − 1) en
    compute_density() hace que la escala sea independiente del tamaño del enjambre.

    Conjuntos:
      'pocos'  — robot con poca cohesión grupal, explorador o aislado.
      'normal' — densidad típica dentro del enjambre.
      'muchos' — robot completamente integrado en el grupo.

    Args:
        density: Densidad normalizada ∈ [0, 1].

    Returns:
        Dict con grados de membresía para 'pocos', 'normal' y 'muchos'.
    """
    return {
        "pocos":  _trapmf(density, 0.0, 0.0, 0.20, 0.40),
        "normal": _trimf(density,  0.25, 0.50, 0.75),
        "muchos": _trapmf(density, 0.60, 0.80, 1.0,  1.0),
    }


def _fuzzify_distance(distance: float) -> dict:
    """
    Fuzzifica la distancia normalizada al estímulo en tres conjuntos.

    La distancia se normaliza respecto al rango efectivo de detección
    r_I + r_s, donde r_s es el radio del estímulo más cercano.
    Así, distance = 0 significa que el robot está sobre el estímulo y
    distance = 1 significa que acaba de detectarlo en el límite del rango.

    Conjuntos:
      'cerca' — señal intensa, robot próximo al estímulo.
      'medio' — señal moderada, distancia intermedia.
      'lejos' — señal débil, robot en el límite de detección.

    Args:
        distance: Distancia normalizada ∈ [0, 1].

    Returns:
        Dict con grados de membresía para 'cerca', 'medio' y 'lejos'.
    """
    return {
        "cerca": _trapmf(distance, 0.0, 0.0, 0.25, 0.45),
        "medio": _trimf(distance,  0.30, 0.50, 0.70),
        "lejos": _trapmf(distance, 0.55, 0.80, 1.0,  1.0),
    }


# ── Bases de reglas ───────────────────────────────────────────────────────────
#
# E3 = uno (un solo estímulo detectado) — 9 reglas
#
#                density → pocos   normal   muchos
# distance ↓
#   cerca              alto     alto    medio
#   medio              alto    medio     bajo
#   lejos             medio     bajo     bajo
#
# Lectura: con un estímulo claro, la distancia domina sobre la densidad.
# Cerca siempre da w_I alto salvo enjambre muy denso (el grupo ya lo lleva).
# Lejos siempre da w_I bajo salvo robot aislado (necesita cualquier guía).

_RULES_ONE: list[tuple[str, str, str]] = [
    ("pocos",  "cerca", "alto"),
    ("normal", "cerca", "alto"),
    ("muchos", "cerca", "medio"),
    ("pocos",  "medio", "alto"),
    ("normal", "medio", "medio"),
    ("muchos", "medio", "bajo"),
    ("pocos",  "lejos", "medio"),
    ("normal", "lejos", "bajo"),
    ("muchos", "lejos", "bajo"),
]

#
# E3 = varios (≥2 estímulos detectados) — 9 reglas
#
#                density → pocos   normal   muchos
# distance ↓
#   cerca              alto    medio     bajo
#   medio             medio   medio      bajo
#   lejos             medio    bajo      bajo
#
# Lectura: la ambigüedad reduce la confianza en la señal.
# Robot aislado + varios estímulos → decisión reactiva al más cercano (w_I alto).
# Robot con vecinos + ambigüedad → el grupo absorbe la incertidumbre (w_I bajo).

_RULES_SEVERAL: list[tuple[str, str, str]] = [
    ("pocos",  "cerca", "alto"),
    ("normal", "cerca", "medio"),
    ("muchos", "cerca", "bajo"),
    ("pocos",  "medio", "medio"),
    ("normal", "medio", "medio"),
    ("muchos", "medio", "bajo"),
    ("pocos",  "lejos", "medio"),
    ("normal", "lejos", "bajo"),
    ("muchos", "lejos", "bajo"),
]

# Centroides de los conjuntos de salida para defuzzificación por centroide ponderado.
# Los valores están normalizados en [0, 1]; se desnormalizan al rango [W_I_MIN, W_I_MAX].
_OUTPUT_CENTROIDS: dict = {
    "bajo":  0.10,
    "medio": 0.50,
    "alto":  0.92,
}


# ── Inferencia y defuzzificación ──────────────────────────────────────────────

def _infer(
    mu_den:  dict,
    mu_dis:  dict,
    rules:   list,
) -> float:
    """
    Aplica inferencia Mamdani y defuzzifica por centroide ponderado.

    Args:
        mu_den: Grados de membresía de densidad.
        mu_dis: Grados de membresía de distancia.
        rules:  Lista de tuplas (density_set, distance_set, output_set).

    Returns:
        Valor desnormalizado de w_I en [W_I_MIN, W_I_MAX].
    """
    weights: dict = {"bajo": 0.0, "medio": 0.0, "alto": 0.0}

    for den_set, dis_set, out_set in rules:
        alpha = min(mu_den[den_set], mu_dis[dis_set])
        if alpha > 1e-9:
            weights[out_set] += alpha

    numerator   = sum(weights[s] * _OUTPUT_CENTROIDS[s] for s in weights)
    denominator = sum(weights[s] for s in weights)

    wi_norm = 0.0 if denominator < 1e-9 else numerator / denominator
    return float(W_I_MIN + wi_norm * (W_I_MAX - W_I_MIN))


def compute_wi(
    density:   float,
    distance:  float,
    n_stimuli: int,
) -> float:
    """
    Calcula el peso de influencia w_I dado el contexto local del robot.

    Selecciona la base de reglas según el número de estímulos detectados
    y aplica inferencia Mamdani con defuzzificación por centroide ponderado.
    Si n_stimuli == 0 el sistema difuso no debe activarse — el caller
    es responsable de no llamar esta función en ese caso.

    Args:
        density:   Fracción de vecinos detectados ∈ [0, 1].
                   Calculada como n_vecinos / (N − 1) — ver compute_density().
        distance:  Distancia normalizada al estímulo más cercano ∈ [0, 1].
                   Calculada como dist / (r_I + r_s) — ver compute_distance_norm().
        n_stimuli: Número de estímulos detectados dentro del rango efectivo.
                   1 → base de reglas para estímulo único.
                   ≥2 → base de reglas para ambigüedad múltiple.

    Returns:
        w_I en [W_I_MIN, W_I_MAX].
    """
    density  = float(np.clip(density,  0.0, 1.0))
    distance = float(np.clip(distance, 0.0, 1.0))

    mu_den = _fuzzify_density(density)
    mu_dis = _fuzzify_distance(distance)

    rules = _RULES_ONE if n_stimuli == 1 else _RULES_SEVERAL
    return _infer(mu_den, mu_dis, rules)


# ── Utilidades de normalización ───────────────────────────────────────────────

def compute_density(n_neighbors: int, n_robots: int) -> float:
    """
    Normaliza el número de vecinos detectados respecto al enjambre completo.

    Usa N − 1 como denominador (máximo teórico de vecinos que un robot puede
    detectar), de modo que la escala se ajusta automáticamente al tamaño del
    enjambre sin parámetros adicionales.

    Args:
        n_neighbors: Vecinos detectados en las zonas R, O y A sumadas.
        n_robots:    Número total de robots en el enjambre (N).

    Returns:
        Densidad normalizada en [0, 1].
    """
    return float(np.clip(n_neighbors / max(n_robots - 1, 1), 0.0, 1.0))


def compute_distance_norm(dist_real: float, r_I: float, r_s: float) -> float:
    """
    Normaliza la distancia al estímulo respecto al rango efectivo de detección.

    El rango efectivo es r_I + r_s: el robot detecta el estímulo cuando
    su zona sensorial r_I intersecta el radio del estímulo r_s. A esa
    distancia la señal es mínima (distance_norm = 1.0). Conforme el robot
    se acerca, la señal se intensifica (distance_norm → 0.0).

    Args:
        dist_real: Distancia euclidiana al centro del estímulo (m).
        r_I:       Radio de detección del robot (m).
        r_s:       Radio del estímulo — indica su intensidad (m).

    Returns:
        Distancia normalizada en [0, 1].
    """
    effective_range = max(r_I + r_s, 1e-9)
    return float(np.clip(dist_real / effective_range, 0.0, 1.0))
