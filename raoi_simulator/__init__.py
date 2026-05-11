# -*- coding: utf-8 -*-
"""
raoi_simulator — Simulador de enjambre de robots con modelo RAOI.

Implementa la tarea de agregación con múltiples estímulos de influencia,
peso de influencia adaptativo mediante lógica difusa y obstáculos estáticos.

Uso principal:
    from raoi_simulator.aggregation import single_run, statistical_run

    # Modo interactivo: el usuario configura todo desde consola
    single_run()

    # Llamada directa con parámetros
    from raoi_simulator.aggregation import run
    report, metrics = run(
        iterations=300, individuals=20,
        r_r=0.3, o_r=1.0, a_r=2.0, i_r=3.0,
        stimuli=[{"x": 3.0, "y": 7.5, "r": 1.0},
                 {"x": 7.5, "y": 3.0, "r": 1.0}],
        obstacles=[{"x": 5.0, "y": 5.0, "r": 0.4}],
        use_fuzzy=True,
    )

Autores:
    Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
    FIME — Universidad Autónoma de Nuevo León

Referencia:
    Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
    Using Influence Factor. J. Intelligent & Robotic Systems.
"""

from .aggregation import run, single_run, statistical_run
from .fuzzy_influence import compute_wi
from .metrics import compute_all

from . import config, metrics, behavior, dynamics, environment, visualization

__all__ = [
    "run",
    "single_run",
    "statistical_run",
    "compute_wi",
    "compute_all",
    "config",
    "metrics",
    "behavior",
    "dynamics",
    "environment",
    "visualization",
]
