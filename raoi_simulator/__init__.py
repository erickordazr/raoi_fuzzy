# -*- coding: utf-8 -*-
"""
raoi_simulator — Simulador de enjambre de robots con modelo RAOI.

Implementa la tarea de agregación con tres capacidades configurables:
  - Múltiples estímulos de influencia simultáneos.
  - Peso de influencia adaptativo (w_I) mediante lógica difusa Mamdani.
  - Obstáculos estáticos circulares integrados en la cadena de repulsión RAOI.

Uso principal:
    from raoi_simulator.aggregation import single_run, statistical_run

    # Modo interactivo
    single_run()

    # Llamada directa con parámetros
    from raoi_simulator.aggregation import run
    report, data, metrics = run(
        iterations=300, individuals=20,
        r_r=0.3, o_r=1.0, a_r=2.0, i_r=3.0,
        stimuli=[{"x": 3.0, "y": 7.5, "r": 1.0},
                 {"x": 7.5, "y": 3.0, "r": 1.0}],
        obstacles=[{"x": 5.0, "y": 5.0, "r": 0.4}],
        use_fuzzy=True,
    )

    # Corrida estadística con gráficas automáticas
    from raoi_simulator.aggregation import statistical_run
    results = statistical_run(replicas=30)
    # Las gráficas se guardan en results/figures/ automáticamente.

    # Generar gráficas desde resultados existentes
    from raoi_simulator.plots import generate_all
    import numpy as np
    results = np.load("results/stat_YYYYMMDD_HHMMSS.npy", allow_pickle=True).item()
    generate_all(results, reports=[], output_dir="figures")

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
from .plots import generate_all

from . import config, metrics, behavior, dynamics, environment, visualization, plots

__all__ = [
    "run",
    "single_run",
    "statistical_run",
    "compute_wi",
    "compute_all",
    "generate_all",
    "config",
    "metrics",
    "behavior",
    "dynamics",
    "environment",
    "visualization",
    "plots",
]