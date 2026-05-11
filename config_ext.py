# -*- coding: utf-8 -*-
"""
Parámetros de configuración del simulador de enjambre RAOI.

Contiene los valores por defecto para la simulación de agregación:
radios de percepción, escenarios de estímulos y obstáculos, parámetros
del sistema difuso y umbrales de métricas.

Ningún valor literal debe aparecer en aggregation.py ni en metrics_ext.py;
todos los parámetros se leen desde este archivo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

# ── Parámetros de simulación ──────────────────────────────────────────────────

DEFAULT_ITERATIONS: int = 300
"""Número de pasos de tiempo por defecto."""

DEFAULT_INDIVIDUALS: int = 20
"""Número de robots por defecto."""

DEFAULT_INFLUENCE_RADIUS: float = 3.0
"""Radio sensorial del robot para detectar estímulos (r_I, en metros)."""

DEFAULT_STIMULUS_RADIUS: float = 1.0
"""
Radio por defecto de un estímulo (r_s, en metros).

Define la intensidad del estímulo: un estímulo con r_s grande puede ser
detectado desde mayor distancia. El robot detecta el estímulo cuando
dist(robot, estímulo) ≤ r_I + r_s.
"""


# ── Escenarios de estímulos ───────────────────────────────────────────────────

STIMULI: list = [
    {"x": 7.5, "y": 7.5, "r": 1.0},
]
"""Estímulo por defecto: una fuente en el cuadrante superior derecho del área."""

STIMULI_SCENARIOS: dict = {
    1: [
        {"x": 7.5, "y": 7.5, "r": 1.0},
    ],
    2: [
        {"x": 3.0, "y": 7.5, "r": 1.0},
        {"x": 7.5, "y": 3.0, "r": 1.0},
    ],
    3: [
        {"x": 2.5, "y": 7.5, "r": 1.0},
        {"x": 7.5, "y": 7.5, "r": 1.0},
        {"x": 5.0, "y": 2.5, "r": 1.0},
    ],
    4: [
        {"x": 2.5, "y": 7.5, "r": 1.0},
        {"x": 7.5, "y": 7.5, "r": 1.0},
        {"x": 2.5, "y": 2.5, "r": 1.0},
        {"x": 7.5, "y": 2.5, "r": 1.0},
    ],
}
"""
Escenarios predefinidos de estímulos, indexados por número de fuentes (1–4).

Cada estímulo se define por su posición (x, y) y su radio de intensidad (r).
Las posiciones están distribuidas para maximizar la separación entre fuentes
dentro del área de 10×10 m.
"""


# ── Escenarios de obstáculos ──────────────────────────────────────────────────

OBSTACLES_SCENARIOS: dict = {
    "none": [],
    "low": [
        {"x": 5.0, "y": 5.0, "r": 0.4},
        {"x": 3.0, "y": 7.0, "r": 0.3},
    ],
    "medium": [
        {"x": 5.0, "y": 5.0, "r": 0.4},
        {"x": 3.0, "y": 7.0, "r": 0.3},
        {"x": 7.0, "y": 3.0, "r": 0.4},
        {"x": 4.0, "y": 3.5, "r": 0.3},
    ],
    "high": [
        {"x": 5.0, "y": 5.0, "r": 0.4},
        {"x": 3.0, "y": 7.0, "r": 0.3},
        {"x": 7.0, "y": 3.0, "r": 0.4},
        {"x": 4.0, "y": 3.5, "r": 0.3},
        {"x": 6.5, "y": 6.5, "r": 0.35},
        {"x": 2.5, "y": 4.5, "r": 0.3},
    ],
}
"""
Escenarios predefinidos de obstáculos, indexados por densidad.

  'none'   — sin obstáculos.
  'low'    — 2 obstáculos.
  'medium' — 4 obstáculos.
  'high'   — 6 obstáculos.

Cada obstáculo se define por su centro (x, y) y radio físico (r) en metros.
"""


# ── Umbrales de métricas ──────────────────────────────────────────────────────

FRAGMENTATION_DISTANCE: float = 3.0
"""
Distancia mínima entre dos robots para considerarlos en subgrupos separados (m).

Equivale aproximadamente a 1.5 veces el radio de atracción por defecto,
garantizando que dos robots a esta distancia no pueden atraerse mutuamente.
"""

STIMULUS_ASSIGNMENT_THRESHOLD: float = 1.5
"""
Radio de asignación: un robot se considera convergido a un estímulo
si su distancia final al mismo es menor que este valor (m).
"""
