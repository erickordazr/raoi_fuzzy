# RAOI Swarm Simulator

Simulador de enjambre de robots basado en el modelo **RAOI** (Repulsión, Atracción, Orientación, Influencia). Implementa la tarea de agregación con tres capacidades configurables de forma independiente:

- **Múltiples estímulos de influencia.** El enjambre puede responder a N fuentes simultáneas. Cada robot detecta un estímulo cuando su zona sensorial `r_I` intersecta el radio de intensidad `r_s` del estímulo — `dist(robot, estímulo) ≤ r_I + r_s` — y selecciona localmente la fuente más cercana sin comunicación con el resto del grupo.
- **Peso de influencia adaptativo (w_I difuso).** Cada robot calcula su propio `w_I` en cada iteración usando un sistema difuso Mamdani con tres entradas locales: densidad de vecinos, distancia normalizada al estímulo más cercano, y número de estímulos detectados.
- **Obstáculos estáticos.** Cilindros bloqueantes integrados en la cadena de repulsión RAOI como fuentes de fuerza virtual, con corrección de posición post-integración.

---

## Estructura del proyecto

```
raoi_simulator/
├── config.py           Parámetros base del modelo RAOI y escenarios de agregación
├── behavior.py         Reglas RAOI y composición vectorial
├── dynamics.py         Modelo cinemático diferencial + integrador RK4 vectorizado
├── environment.py      Detección de paredes y parcelas
├── fuzzy_influence.py  Sistema difuso Mamdani para w_I adaptativo
├── metrics.py          Todas las métricas de desempeño (17 claves)
├── plots.py            Figuras de análisis estadístico (8 figuras)
├── aggregation.py      Simulación principal y API pública
├── visualization.py    Renderizado Pygame y video OpenCV
└── __init__.py

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

### Desde código — corrida individual

```python
from raoi_simulator.aggregation import run

report, data, metrics = run(
    iterations=300, individuals=20,
    r_r=0.3, o_r=1.0, a_r=2.0, i_r=3.0,
    stimuli=[{"x": 3.0, "y": 7.5, "r": 1.0},
             {"x": 7.5, "y": 3.0, "r": 1.5}],
    obstacles=[{"x": 5.0, "y": 5.0, "r": 0.4}],
    use_fuzzy=True,
    seed=42,
)
```

### Desde código — corrida estadística

```python
from raoi_simulator.aggregation import statistical_run

results = statistical_run(replicas=30)
# Las 8 gráficas se guardan automáticamente en results/figures/
```

### Regenerar gráficas desde un .npy guardado

```python
# Con reportes completos (figuras 2, 3, 8 incluidas)
from raoi_simulator.plots import generate_all
generate_all(results, all_reports, output_dir="figures")

# Sin reportes (solo métricas escalares — figuras 1, 4, 5, 6, 7)
import numpy as np
results = np.load("results/stat_YYYYMMDD_HHMMSS.npy", allow_pickle=True).item()
generate_all(results, reports=[], output_dir="figures")

# O desde línea de comandos
python -m raoi_simulator.plots results/stat_YYYYMMDD_HHMMSS.npy figures/
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

La señal percibida es proporcional a la proximidad: mínima en el límite del rango (`dist = r_I + r_s`), máxima cuando el robot está sobre el centro del estímulo (`dist = 0`). Esta distancia normalizada `dist / (r_I + r_s)` es la entrada E2 del sistema difuso.

---

## Sistema difuso

`fuzzy_influence.compute_wi(density, distance, n_stimuli)` → `w_I ∈ [0.10, 1.20]`

### Entradas

| Variable | Símbolo | Rango | Conjuntos lingüísticos |
|---|---|---|---|
| Densidad de vecinos | E1 | [0, 1] | pocos · normal · muchos |
| Distancia al estímulo (norm.) | E2 | [0, 1] | cerca · medio · lejos |
| Número de estímulos detectados | E3 | entero | selecciona base de reglas |

### Salida

`w_I ∈ [0.10, 1.20]` — pesa la componente de influencia en la composición vectorial RAOI. W_MAX = 1.20 supera la suma `w_o + w_a`, garantizando que la señal de influencia pueda dominar sobre la cohesión grupal cuando el contexto lo justifica.

### Base de reglas — E3 = 1 (un solo estímulo)

| density \ distance | cerca | medio | lejos |
|---|---|---|---|
| pocos | **alto** | **alto** | medio |
| normal | **alto** | medio | bajo |
| muchos | medio | bajo | bajo |

### Base de reglas — E3 ≥ 2 (varios estímulos)

| density \ distance | cerca | medio | lejos |
|---|---|---|---|
| pocos | **alto** | medio | medio |
| normal | medio | medio | bajo |
| muchos | bajo | bajo | bajo |

Con señal ambigua (varios estímulos), la densidad domina: muchos vecinos siempre producen `w_I` bajo porque el grupo absorbe la incertidumbre.

---

## Métricas

`metrics.compute_all(report, stimuli, i_r, obstacles, r_rep)` devuelve un dict con las siguientes claves.

### Convergencia

| Métrica | Descripción | Unidad | Valor ideal | Interpretación |
|---|---|---|---|---|
| `convergence_time` | Primera iteración en que ≥ `LOCALIZATION_THRESHOLD` del enjambre entra en el rango sensorial (`r_I + r_s`) de cualquier estímulo. | iter | bajo | Si devuelve `T`, el enjambre nunca convergió. Compara con `physical_convergence_time` para cuantificar cuánto tiempo tardan los robots en cruzar de la zona de detección al cuerpo del estímulo. |
| `physical_convergence_time` | Igual que `convergence_time` pero usando solo `r_s` como radio de llegada. Mide ocupación física real, no sensorial. | iter | bajo | Siempre ≥ `convergence_time`. Una brecha grande entre ambos indica que los robots detectan el estímulo desde lejos pero convergen lentamente al centro. |
| `convergence_time_per_stimulus` | Lista con `convergence_time` individual para cada estímulo. | list[iter] | bajos y similares | Valores muy desiguales revelan asimetría: algún estímulo atrae robots mucho antes. Útil para detectar sesgos posicionales o de `r_s`. |
| `physical_convergence_time_per_stimulus` | Ídem para llegada física por estímulo. | list[iter] | bajos y similares | Relación garantizada: `first_arrival[k] ≤ physical_conv[k] ≤ conv_time[k]` NO — en realidad `conv_time[k] ≤ physical_conv[k]`. Úsalo junto con `first_arrival` para ver la secuencia completa. |

### Cohesión y fragmentación

| Métrica | Descripción | Unidad | Valor ideal | Interpretación |
|---|---|---|---|---|
| `cohesion_mean` | Distancia media de cada robot al centroide del enjambre, promediada en el tiempo. | m | bajo | En escenarios multi-estímulo, un valor alto no es necesariamente malo: puede indicar fragmentación emergente correcta donde subgrupos se distribuyen entre fuentes. Comparar con `fragmentation_mean` para diferenciar ambos casos. |
| `cohesion_final` | Ídem en la última iteración. | m | bajo | Si el enjambre convergió exitosamente, `cohesion_final << cohesion_mean`. Si son similares, el enjambre no se compactó al final. |
| `fragmentation_mean` | Fracción de pares de robots separados por más de `r_a`, promedio temporal. | [0, 1] | depende de n_stimuli | Para 1 estímulo: idealmente decrece a 0. Para k estímulos con distribución uniforme de N/k robots cada uno: valor esperado ≈ (k−1)/k. Si es sistemáticamente más alto, hay subgrupos que no convergieron. |
| `fragmentation_final` | Ídem en la última iteración. | [0, 1] | depende de n_stimuli | Indicador del estado final de distribución. Compara con `fragmentation_mean` para ver si la fragmentación es estable o transitoria. |

### Distribución entre estímulos

| Métrica | Descripción | Unidad | Valor ideal | Interpretación |
|---|---|---|---|---|
| `robots_per_stimulus` | Número de robots dentro del radio de asignación (`STIMULUS_ASSIGNMENT_THRESHOLD`) por estímulo al finalizar. | list[int] | N/k por estímulo | Distribución desigual puede deberse a diferencias de `r_s`, posición o dinámica emergente. No es necesariamente un fallo: es el comportamiento descentralizado. |
| `distribution_entropy` | Entropía de Shannon de la distribución final de robots entre estímulos. | bits | alto (≤ log₂(k)) | Máxima cuando robots se distribuyen uniformemente (log₂(k) bits para k estímulos). Un valor bajo indica que el enjambre se concentró en un subconjunto de estímulos. Con un solo estímulo, siempre es 0. |

### Permanencia en estímulos

| Métrica | Descripción | Unidad | Valor ideal | Interpretación |
|---|---|---|---|---|
| `dwell_count` | Total de robot-iteraciones dentro de `r_s` por estímulo a lo largo de toda la simulación. | list[int] | alto | Mide la inversión total de tiempo del enjambre en cada fuente. Proporcional a la "calidad percibida" del estímulo. Compara entre estímulos para detectar asimetrías de atracción. |
| `stimulus_occupancy` | Fracción del tiempo (post primera llegada) con al menos un robot dentro de `r_s`. | list[float ∈ [0,1]] | 1.0 | Un valor < 1 indica que el estímulo fue abandonado en algún momento. Valor cercano a 0 = los robots pasan pero no se quedan. |
| `first_arrival` | Primera iteración en que cualquier robot entró físicamente en el cuerpo del estímulo (`dist ≤ r_s`). | list[iter] | bajo | Mide la eficiencia de exploración pura. Su diferencia con `convergence_time_per_stimulus` muestra cuánto tarda el primer robot versus que el grupo alcance el umbral. |
| `mean_robots_at_stimulus` | Media temporal de robots simultáneamente dentro de `r_s` por estímulo. | list[float] | alto | Distingue ocupación sostenida (valor alto) de visitas fugaces (valor bajo aunque `stimulus_occupancy` sea alto). |
| `transit_fraction` | Fracción del tiempo total que los robots pasaron fuera de TODOS los estímulos. | float ∈ [0,1] | bajo | Decrece conforme la tarea avanza. Un valor final alto indica que el enjambre pasa más tiempo navegando que en los destinos. Con T suficientemente grande debe converger a 0. |

### Obstáculos (solo cuando `obstacles` no es vacío)

| Métrica | Descripción | Unidad | Valor ideal | Interpretación |
|---|---|---|---|---|
| `obstacle_interaction_rate` | Fracción de robot-iteraciones donde se activó repulsión por obstáculo. | float ∈ [0,1] | bajo | Mide el impacto de los obstáculos en la dinámica. Un valor alto indica que los robots pasan mucho tiempo maniobrado alrededor de obstáculos. |
| `detour_ratio` | Longitud de trayectoria media con obstáculos dividida por la misma corrida sin obstáculos. | ≥ 1.0 | cercano a 1.0 | 1.0 = obstáculos sin efecto. >1.5 = rodeos significativos. Compara directamente el costo de navegación que imponen los obstáculos. |

---

## Gráficas estadísticas

Las 8 figuras se generan automáticamente en `results/figures/` al ejecutar `statistical_run()`. Las figuras 2, 3 y 8 requieren los reportes completos de posición y estado; las demás solo necesitan las métricas escalares.

### Figura 1 — `1_convergence_distribution.png`

**Qué muestra:** Distribución completa del tiempo de convergencia mediante violines con caja superpuesta. Cada estímulo tiene su propio panel. Se muestran dos violines por panel: detección sensorial (`convergence_time`) y llegada física (`physical_convergence_time`).

**Cómo leer:** La anchura del violín en cada altura indica la densidad de réplicas con ese valor. La caja interior muestra Q1–Q3 y la mediana como punto blanco. Un violin estrecho y alto indica consistencia; uno ancho y bajo indica alta variabilidad entre réplicas. La brecha horizontal entre el violín sensorial y el físico cuantifica cuánto tiempo transcurre entre que los robots *detectan* un estímulo y cuando *llegan físicamente*.

### Figura 2 — `2_cohesion_evolution.png`

**Qué muestra:** Panel izquierdo: curva temporal de cohesión media ± std calculada directamente desde las posiciones de todos los reportes. Panel derecho: boxplot de `cohesion_final` sobre todas las réplicas.

**Cómo leer:** El patrón típico es: cohesión baja inicial (robots juntos en el spawn) → sube durante la navegación y fragmentación → baja cuando el enjambre converge a los estímulos. Si la curva nunca baja, el enjambre no convergió. Si la curva sube y se mantiene alta, hay fragmentación permanente (esperado con múltiples estímulos separados). El panel derecho muestra si la cohesión final es consistente entre réplicas o si hay runs atípicos.

### Figura 3 — `3_fragmentation_evolution.png`

**Qué muestra:** Fracción de pares de robots separados por más de `r_a` en cada iteración, media ± std sobre todas las réplicas. Las líneas verticales punteadas indican la mediana de `first_arrival` por estímulo.

**Cómo leer:** Para un estímulo único: la fragmentación debe decrecer monotónicamente hacia 0 conforme el enjambre converge. Para múltiples estímulos: se espera que *suba* conforme los robots se dividen entre fuentes — esto es el comportamiento emergente deseado de la Contribución 1. Los marcadores de primera llegada muestran qué estímulo fue el detonante de la fragmentación: si la curva sube justo después de un marcador, ese estímulo atrajo un subgrupo antes de que el resto llegara.

### Figura 4 — `4_stimulus_load_balance.png`

**Qué muestra:** Dos paneles de barras con error (media ± std) para `stimulus_occupancy` y `mean_robots_at_stimulus` por estímulo. Los puntos semitransparentes muestran cada réplica individual.

**Cómo leer:** Barras de igual altura indican distribución equilibrada. Barras muy desiguales revelan asimetría emergente: algún estímulo atrae sistemáticamente más robots. Los puntos individuales son clave: si están dispersos alrededor de la barra, la distribución varía por réplica (comportamiento estocástico). Si están agrupados, la asimetría es estructural (posición o `r_s` diferente).

### Figura 5 — `5_convergence_ecdf.png`

**Qué muestra:** Función de distribución acumulada empírica (ECDF) de `convergence_time` y `physical_convergence_time`. Si hay múltiples estímulos, panel derecho con ECDF por estímulo (sólido = sensorial, punteado = físico).

**Cómo leer:** El valor en el eje X donde la curva cruza el 50% es la mediana; donde cruza el 90% es el percentil 90. Una curva con forma de S suave indica distribución concentrada (alta consistencia). Una curva casi plana con escalones grandes indica alta variabilidad entre réplicas. La diferencia horizontal entre la curva sensorial y la física es el "gap de llegada" promedio. Para comparar escenarios, una curva desplazada a la izquierda es estrictamente mejor.

### Figura 6 — `6_fuzzy_wi_surface.png`

**Qué muestra:** Dos mapas de calor 2D de la superficie de inferencia `w_I(density, distance)` evaluada analíticamente: panel izquierdo para 1 estímulo detectado, panel derecho para ≥2 estímulos.

**Cómo leer:** El eje X es la densidad de vecinos (E1): izquierda = robot aislado, derecha = robot rodeado. El eje Y es la distancia normalizada al estímulo (E2): abajo = robot encima del estímulo, arriba = robot en el límite de detección. Verde = `w_I` bajo (cohesión grupal domina), rojo = `w_I` alto (influencia domina). La esquina inferior-izquierda (aislado, cerca del estímulo) siempre es roja: un robot solo y próximo a la fuente debe seguirla fuertemente. La comparación entre paneles muestra cómo la ambigüedad de múltiples estímulos hace al sistema más conservador (más verde en general).

### Figura 7 — `7_localization_timeline.png`

**Qué muestra:** Diagrama de Gantt con tres fases por estímulo basado en medianas sobre réplicas. Las barras de error horizontales muestran el IQR de cada transición.

**Fases:**
- **Approach** (verde): desde t=0 hasta `first_arrival`. Ningún robot ha llegado al estímulo aún.
- **Build-up** (naranja): desde `first_arrival` hasta `convergence_time_per_stimulus`. Robots entrando en la zona de detección.
- **Dwelling** (azul): desde `convergence_time_per_stimulus` hasta `physical_convergence_time_per_stimulus`. El umbral sensorial se superó; robots llegando físicamente.
- **Post-convergence** (gris claro): tiempo restante hasta T.

**Cómo leer:** Una barra de Approach larga indica exploración lenta (ajustar `r_I` o `r_a`). Una barra de Build-up larga indica que la señal de influencia es débil o que los robots llegan de uno en uno. Si la barra de Dwelling o Post-convergence domina, el enjambre tiene tiempo más que suficiente — T puede reducirse. Si no aparece la fase Dwelling, el estímulo nunca alcanzó el umbral de convergencia.

### Figura 8 — `8_raoi_state_evolution.png`

**Qué muestra:** Distribución temporal de los estados conductuales RAOI como área apilada, promediada sobre todas las réplicas. Estados: Libre, Repulsión, Atracción, Orientación, Influencia. La banda semitransparente sobre Influencia muestra ± 1 std.

**Cómo leer:** Esta figura verifica directamente que el modelo funciona como se diseñó. El patrón esperado es: Libre domina al inicio (robots buscando), Atracción aparece conforme el enjambre se forma, Influencia crece conforme los robots detectan estímulos y debe dominar en el estado final. Si Influencia nunca sube, revisar `r_I` (demasiado pequeño) o posición de estímulos. Si Repulsión es persistentemente alta, el área está sobrepoblada para los radios configurados. Si Libre domina hasta el final, los estímulos no están siendo detectados.

---

## Relaciones entre métricas

```
first_arrival[k]  ≤  convergence_time_per_stimulus[k]  ≤  physical_convergence_time_per_stimulus[k]  ≤  T
cohesion_final  ≤  cohesion_mean  (por construcción)
distribution_entropy  ≤  log₂(n_stimuli)
```

---

## Escenarios predefinidos

```python
import raoi_simulator.config as cfg

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

## Autor

Erick Ordaz-Rivas — erick.ordazrv@uanl.edu.mx  
FIME — Universidad Autónoma de Nuevo León

**Referencia:**  
Ordaz-Rivas et al. (2018). *Collective Tasks for a Flock of Robots Using Influence Factor.* Journal of Intelligent & Robotic Systems.