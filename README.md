# RAOI Swarm Simulator

Simulador de enjambre de robots basado en el modelo **RAOI** (Repulsión, Atracción, Orientación, Influencia). Implementa la tarea de agregación con tres capacidades configurables de forma independiente:

**Múltiples estímulos de influencia.** El enjambre puede responder a N fuentes simultáneas. Cada robot detecta un estímulo cuando su zona sensorial `r_I` intersecta el radio de intensidad `r_s` del estímulo — `dist(robot, estímulo) ≤ r_I + r_s` — y selecciona localmente la fuente más cercana sin comunicación con el resto del grupo. Cuando los estímulos están suficientemente separados, el enjambre puede fragmentarse espontáneamente en subgrupos.

**Peso de influencia adaptativo (w_I difuso).** En lugar de un peso fijo, cada robot calcula su propio `w_I` en cada iteración usando un sistema difuso Mamdani con tres entradas locales: densidad de vecinos, distancia al estímulo más cercano normalizada por el rango de detección, y número de estímulos detectados simultáneamente. Sin dependencias externas.

**Obstáculos estáticos.** Cilindros bloqueantes en el área que actúan como fuentes de repulsión virtual dentro de la cadena RAOI, con corrección física de posición para garantizar que ningún robot los atraviesa.

---

## Estructura del proyecto

```
raoi_simulator/
├── config.py           Parámetros base del modelo RAOI
├── behavior.py         Reglas RAOI y composición vectorial
├── dynamics.py         Modelo cinemático diferencial + integrador RK4
├── environment.py      Detección de paredes
├── fuzzy_influence.py  Sistema difuso para w_I
├── metrics.py          Todas las métricas de desempeño
├── aggregation.py      Simulación principal y API pública
├── visualization.py    Renderizado Pygame y figuras estáticas
└── __init__.py

config_ext.py           Escenarios predefinidos de estímulos y obstáculos
plots.py                Generación de gráficas para corridas estadísticas
main.py                 Punto de entrada interactivo
requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso rápido

### Modo interactivo

```bash
python main.py
```

Guía al usuario paso a paso: parámetros RAOI → estímulos → fuzzy → obstáculos → animación → guardado.

### Desde código

```python
from raoi_simulator.aggregation import run

_, _, metrics = run(
    iterations=300, individuals=20,
    r_r=0.3, o_r=1.0, a_r=2.0, i_r=3.0,
    stimuli=[{"x": 3.0, "y": 7.5, "r": 1.0},
             {"x": 7.5, "y": 3.0, "r": 1.5}],
    obstacles=[{"x": 5.0, "y": 5.0, "r": 0.4}],
    use_fuzzy=True,
    seed=42,
)
```

### Corrida estadística + gráficas

```python
from raoi_simulator.aggregation import statistical_run, run
from plots import generate_all

results = statistical_run(replicas=30)

# Guardar reportes para la figura de fragmentación temporal
all_reports = []
for rep in range(30):
    report, _, _ = run(..., seed=42 + rep)
    all_reports.append(report)

generate_all(results, all_reports, output_dir="figures")
```

---

## Parámetros de simulación

| Parámetro | Tipo | Descripción |
|---|---|---|
| `iterations` | int | Número de pasos de tiempo |
| `individuals` | int | Número de robots |
| `r_r` | float (m) | Radio de repulsión (se suma al radio del cuerpo del robot) |
| `o_r` | float (m) | Radio de orientación |
| `a_r` | float (m) | Radio de atracción |
| `i_r` | float (m) | Radio sensorial del robot para detectar estímulos (r_I) |
| `stimuli` | list | Lista de `{"x", "y", "r"}` — posición y radio de intensidad |
| `obstacles` | list | Lista de `{"x", "y", "r"}` — posición y radio físico |
| `use_fuzzy` | bool | Activar w_I adaptativo (True) o usar w_I constante (False) |
| `seed` | int | Semilla aleatoria para reproducibilidad |

---

## Detección de estímulos

Un robot detecta un estímulo cuando su zona sensorial intersecta el radio del estímulo:

```
dist(robot, estímulo) ≤ r_I + r_s
```

La señal percibida es proporcional a la proximidad: mínima en el límite del rango (`dist = r_I + r_s`), máxima cuando el robot está sobre el centro del estímulo (`dist = 0`). Esta distancia normalizada `dist / (r_I + r_s)` es la entrada E2 del sistema difuso y la referencia para escalar la velocidad del robot.

---

## Sistema difuso

`fuzzy_influence.compute_wi(density, distance, n_stimuli)` → `w_I ∈ [0.10, 1.20]`

### Entradas

**E1 — density** `∈ [0, 1]`
Fracción de vecinos detectados en las zonas R+O+A respecto a N−1. 0 = robot completamente aislado. 1 = todos los demás robots son vecinos.

Conjuntos lingüísticos:
- `pocos` : trapecio [0.00, 0.00, 0.20, 0.40]
- `normal`: triángulo [0.25, 0.50, 0.75]
- `muchos`: trapecio [0.60, 0.80, 1.00, 1.00]

**E2 — distance** `∈ [0, 1]`
Distancia al estímulo más cercano detectado, normalizada por `r_I + r_s`. 0 = robot sobre el estímulo. 1 = robot en el límite exacto de detección.

Conjuntos lingüísticos:
- `cerca`: trapecio [0.00, 0.00, 0.25, 0.45]
- `medio`: triángulo [0.30, 0.50, 0.70]
- `lejos`: trapecio [0.55, 0.80, 1.00, 1.00]

**E3 — n_stimuli** (entero)
Número de estímulos detectados en este paso. Selecciona la base de reglas activa. Si E3 = 0 el sistema difuso no se activa y se usa el w_I constante de configuración.

### Salida

**w_I** `∈ [0.10, 1.20]`
Peso de la componente de influencia en la composición vectorial RAOI. W_MAX = 1.20 supera la suma `w_o + w_a` (típicamente 1.0), garantizando que un w_I alto permita a la señal dominar sobre la cohesión grupal.

Conjuntos de salida (defuzzificación por centroide ponderado):
- `bajo` : centroide = 0.10 → w_I ≈ 0.22 m
- `medio`: centroide = 0.50 → w_I ≈ 0.65 m
- `alto` : centroide = 0.92 → w_I ≈ 1.10 m

### Base de reglas — E3 = 1 (un solo estímulo)

| density \ distance | cerca | medio | lejos |
|---|---|---|---|
| pocos | **alto** | **alto** | medio |
| normal | **alto** | medio | bajo |
| muchos | medio | bajo | bajo |

Con señal unívoca, la distancia domina: cerca siempre es alto salvo enjambre muy denso.

### Base de reglas — E3 ≥ 2 (varios estímulos)

| density \ distance | cerca | medio | lejos |
|---|---|---|---|
| pocos | **alto** | medio | medio |
| normal | medio | medio | bajo |
| muchos | bajo | bajo | bajo |

Con señal ambigua, la densidad domina: muchos vecinos siempre dan bajo porque el grupo absorbe la incertidumbre.

---

## Métricas

`metrics.compute_all(report, stimuli, i_r, obstacles, report_baseline)` devuelve:

### Convergencia
| Métrica | Descripción | Unidad | Ideal |
|---|---|---|---|
| `convergence_time` | Iteración en que el umbral del enjambre entra al rango del estímulo | iter | bajo |
| `convergence_time_per_stimulus` | `convergence_time` individual por cada estímulo | list[iter] | bajo |

### Cohesión y fragmentación
| Métrica | Descripción | Unidad | Ideal |
|---|---|---|---|
| `cohesion_mean` | Distancia media de cada robot al centroide, promediada en el tiempo | m | bajo |
| `cohesion_final` | Ídem en la última iteración | m | bajo |
| `fragmentation_mean` | Fracción de pares separados > umbral, promedio temporal | [0,1] | depende |
| `fragmentation_final` | Ídem en la última iteración | [0,1] | depende |

### Distribución entre estímulos
| Métrica | Descripción | Unidad | Ideal |
|---|---|---|---|
| `robots_per_stimulus` | Robots dentro del radio de asignación por estímulo al final | list[int] | — |
| `distribution_entropy` | Entropía de Shannon de la distribución final | bits | depende |

### Permanencia
| Métrica | Descripción | Unidad | Ideal |
|---|---|---|---|
| `dwell_count` | Iteraciones-robot dentro de r_s por estímulo | list[int] | alto |
| `stimulus_occupancy` | Fracción del tiempo con al menos un robot en el estímulo | list[0,1] | alto |
| `first_arrival` | Primera iteración en que un robot entró al cuerpo del estímulo | list[iter] | bajo |
| `mean_robots_at_stimulus` | Robots medios simultáneamente dentro de r_s | list[float] | alto |
| `transit_fraction` | Fracción del tiempo que los robots pasaron fuera de todos los estímulos | [0,1] | bajo |

### Obstáculos
| Métrica | Descripción | Unidad | Ideal |
|---|---|---|---|
| `obstacle_interaction_rate` | Fracción de iteraciones-robot con repulsión activa por obstáculo | [0,1] | bajo |
| `detour_ratio` | Alargamiento de trayectoria vs. corrida sin obstáculos | ≥1.0 | cerca de 1 |

---

## Escenarios predefinidos

```python
import config_ext as cfg

cfg.STIMULI_SCENARIOS[1]            # 1 fuente  en (7.5, 7.5)
cfg.STIMULI_SCENARIOS[2]            # 2 fuentes en esquinas opuestas
cfg.STIMULI_SCENARIOS[3]            # 3 fuentes en triángulo
cfg.STIMULI_SCENARIOS[4]            # 4 fuentes en cuadrícula

cfg.OBSTACLES_SCENARIOS["none"]     # sin obstáculos
cfg.OBSTACLES_SCENARIOS["low"]      # 2 obstáculos
cfg.OBSTACLES_SCENARIOS["medium"]   # 4 obstáculos
cfg.OBSTACLES_SCENARIOS["high"]     # 6 obstáculos
```

---

## Gráficas (plots.py)

| # | Archivo | Contenido |
|---|---|---|
| 1 | `1_convergence_time.png` | Histograma de convergence_time global y por estímulo |
| 2 | `2_cohesion.png` | Boxplot cohesion_mean vs. cohesion_final |
| 3 | `3_fragmentation_evolution.png` | Curva temporal de fragmentación media ± std |
| 4 | `4_stimulus_occupancy.png` | Barras de fracción de tiempo ocupado y robots medios por estímulo |
| 5 | `5_time_distribution.png` | Tarta y barras apiladas de distribución del tiempo |

---

## Autor

Erick Ordaz-Rivas — erick.ordazrv@uanl.edu.mx  
FIME — Universidad Autónoma de Nuevo León

**Referencia:**  
Ordaz-Rivas et al. (2018). *Collective Tasks for a Flock of Robots Using Influence Factor.* Journal of Intelligent & Robotic Systems.
