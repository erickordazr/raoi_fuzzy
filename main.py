# -*- coding: utf-8 -*-
"""
Simulador de enjambre de robots con modelo RAOI.

Ejecutar con:
    python main.py

O desde código:
    from raoi_simulator.aggregation import run
    report, data, metrics = run(
        iterations=300, individuals=20,
        r_r=0.3, o_r=1.0, a_r=2.0, i_r=3.0,
        stimuli=[{"x": 3.0, "y": 7.5}, {"x": 7.5, "y": 3.0}],
        obstacles=[{"x": 5.0, "y": 5.0, "r": 0.4}],
        use_fuzzy=True,
    )

Autores:
    Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
    FIME — Universidad Autónoma de Nuevo León
"""

from raoi_simulator.aggregation import single_run, statistical_run


def main() -> None:
    """Menú interactivo del simulador RAOI."""
    menu = """
    ╔══════════════════════════════════════════════════╗
    ║         RAOI Swarm Simulator                     ║
    ╠══════════════════════════════════════════════════╣
    ║   1. Simulación individual                       ║
    ║   2. Corrida estadística (múltiples réplicas)    ║
    ║   3. Salir                                       ║
    ╚══════════════════════════════════════════════════╝
    """
    print(menu)

    while True:
        try:
            choice = int(input("Opción: "))
        except ValueError:
            print("Ingresa un número.")
            continue

        if choice == 1:
            single_run()
            break
        elif choice == 2:
            while True:
                try:
                    replicas = int(input("Número de réplicas: "))
                    break
                except ValueError:
                    print("Ingresa un número entero.")
            statistical_run(replicas)
            break
        elif choice == 3:
            break
        else:
            print("Opción inválida. Elige 1–3.")


if __name__ == "__main__":
    main()
