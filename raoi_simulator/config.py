# -*- coding: utf-8 -*-
"""
Configuración global del simulador RAOI.

Todos los parámetros del modelo, del escenario y de la visualización
residen aquí. Ningún módulo debe contener valores literales —
siempre importar desde este archivo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

# ── Reproducibilidad ──────────────────────────────────────────────────────────

SEED: int = 42
"""Semilla global. Cada réplica usa SEED + replica_index para independencia estadística."""

# ── Simulación ────────────────────────────────────────────────────────────────

DT: float = 1.0
"""
Paso de tiempo en segundos.

Los voltajes del modelo dinámico están calibrados para DT=1.0 s.
Reducir DT sin recalibrar los voltajes produce desplazamientos
imperceptibles. Si se requiere un paso más fino, escalar los voltajes
proporcionalmente o recalibrar el modelo.
"""

RK4_SUBSTEPS: int = 10
"""
Subdivisiones internas del integrador RK4.

10 subdivisiones dan precisión equivalente al integrador odeint original.
Reducir a 4–5 para mayor velocidad con menor precisión numérica.
"""

# ── Parámetros RAOI ───────────────────────────────────────────────────────────

RAOI_WEIGHTS: dict = {
    "w_r": 0.8,   # Peso de repulsión  (dominante)
    "w_o": 0.5,   # Peso de orientación
    "w_a": 0.3,   # Peso de atracción
    "w_I": 0.2,   # Peso de influencia
}
"""
Pesos de cada zona de percepción.

Con w_r=1.0 y el resto en 0 se reproduce la prioridad absoluta del paper 2018.
Con pesos intermedios se obtiene comportamiento combinado con inercia adaptativa:
la inercia del robot escala como (1 - max_weight), de modo que el robot con
w_r=0.8 gira más suavemente que con w_r=1.0.
"""

ROBOT_BODY_RADIUS: float = 0.075
"""Radio físico del robot (m). Corresponde al Khepera III / e-puck."""

RAOI_RADII: dict = {
    "r_repulsion":   0.075,
    "r_orientation": 1.0,
    "r_attraction":  2.0,
}
"""
Radios de zona RAOI en metros.

Estos valores se suman a ROBOT_BODY_RADIUS en tiempo de ejecución.
Rangos típicos según la literatura: r_r ∈ [0, 0.2], r_o ∈ [0.4, 0.6], r_a ∈ [1, 1.2].
"""

RAOI_FOV: dict = {
    "fov_repulsion":   math.pi,        # ±90°  — frontal
    "fov_orientation": 2 * math.pi,    # 360°  — omnidireccional
    "fov_attraction":  math.pi,        # ±90°  — frontal
    "fov_influence":   math.pi,        # ±90°  — frontal
}
"""
Campos de visión por zona (radianes).

math.pi  → 180° (semicírculo frontal).
2*math.pi → 360° (omnidireccional).
"""

# ── Modelo dinámico — robot diferencial ──────────────────────────────────────

ROBOT_MASS: float       = 0.38      # Masa total (kg)
ROBOT_INERTIA: float    = 0.005     # Momento de inercia (kg·m²)
ROBOT_D: float          = 0.02      # Distancia centroide → eje de ruedas (m)
ROBOT_WHEEL_R: float    = 0.03      # Radio de rueda (m)
ROBOT_WHEEL_SEP: float  = 0.05      # Semiseparación entre ruedas (m)

MOTOR_Ts: float = 0.434             # Constante de tiempo del motor (s)
MOTOR_Ks: float = 2.745             # Ganancia de velocidad (rad / s·N·m)
MOTOR_Kl: float = 1460.2705         # Ganancia de corriente (rad / s·V)

VOLTAGE: dict = {
    "repulsion":   2.0,   # ~15 cm/s
    "orientation": 2.7,   # ~20 cm/s
    "attraction":  3.7,   # ~30 cm/s
}
"""
Voltajes de referencia por estado RAOI (V).

La influencia usa interpolación lineal entre V_repulsion y V_attraction
en función de la distancia normalizada a la fuente.
"""

# ── Límites físicos del Khepera III ───────────────────────────────────────────

OMEGA_MAX: float    = 10.0   # Velocidad angular máxima (rad/s)
V_MAX_LINEAR: float = 0.5    # Velocidad lineal máxima (m/s)

# ── Controlador de giro proporcional ─────────────────────────────────────────

KP_TURN: float = 0.8
"""
Ganancia del controlador de voltaje diferencial.

v_diff = KP_TURN * theta_error  →  voltajes = [v_base + v_diff, v_base - v_diff]

KP_TURN = 0.0 → voltajes simétricos (sin giro activo)
KP_TURN = 0.5 → giro suave
KP_TURN = 2.0 → giro agresivo
"""

# ── Escenario ─────────────────────────────────────────────────────────────────

AREA_LIMITS: float = 10.0
"""
Lado del area cuadrada de simulacion para aggregation y foraging (m).
Farming usa FARMING_AREA_LIMITS definido en su propia seccion.
"""

FARMING_AREA_LIMITS: float = 100.0
"""Lado del área cuadrada de simulación (m)."""

INFLUENCE_POSITION: list = [AREA_LIMITS * 0.75, AREA_LIMITS * 0.75]
"""Posición [x, y] de la fuente de influencia (punto de agregación)."""

# ── Zona de spawn ─────────────────────────────────────────────────────────────

SPAWN_FRACTION: float     = 0.22
"""Fraccion del area usada como zona de spawn para aggregation (lado del cuadrado SW)."""

AGGREGATION_SPAWN_SIDE: float = AREA_LIMITS * SPAWN_FRACTION
"""
Lado del cuadrado de spawn para aggregation (m).
Coincide exactamente con el cuadrado gris que se dibuja en la animacion.
Con AREA_LIMITS=10 y SPAWN_FRACTION=0.22: 2.2 m de lado.
"""
"""Fracción del lado del área usada como zona de spawn inicial."""

SPAWN_MIN_SEPARATION: float = 0.3
"""
Separación mínima entre robots en el spawn (m).

En tiempo de ejecución se eleva automáticamente a 2*r_repulsion si
este valor resulta menor, para evitar que los robots arranquen en
zona de repulsión mutua.
"""

SPAWN_MAX_ATTEMPTS: int = 10_000
"""Intentos máximos para ubicar un robot con la separación requerida."""

# ── Ruido del sensor de influencia ───────────────────────────────────────────

INFLUENCE_NOISE_AMP: float = 0.05
"""
Amplitud del ruido gaussiano añadido al ángulo percibido de la fuente (rad).

El ruido modela incertidumbre en la dirección del estímulo, no en la
detección (radio de detección permanece determinístico).
"""

# ── Exploración libre ─────────────────────────────────────────────────────────

EXPLORE_FREE_ITERS: int   = 10
"""
Iteraciones consecutivas en estado libre antes de activar el random walk.

Durante las primeras EXPLORE_FREE_ITERS iteraciones sin vecinos el robot
mantiene su última dirección activa. Después de ese umbral, la dirección
acumula un giro gaussiano por iteración (EXPLORE_TURN_NOISE).
"""

EXPLORE_TURN_NOISE: float = 0.15
"""
Amplitud del giro gaussiano por iteración en estado de exploración activa (rad).

0.15 rad ≈ ±8.6° — exploración en arco suave.
Aumentar para cobertura más agresiva del área.
"""

DIREXP_RESET_NOISE: float = 0.1
"""
Perturbación gaussiana aplicada a dirExp al entrar en estado libre (rad).

Rompe la inercia de la última dirección activa sin redirigir bruscamente.
"""

# ── Métricas ──────────────────────────────────────────────────────────────────

LOCALIZATION_THRESHOLD: float = 0.6
"""
Fracción mínima del enjambre que debe estar dentro del radio de influencia
para considerar la tarea de localización completada (f1).
"""

AREA_COVERAGE_MAX_FRACTION: float = 0.8
"""Cota superior del área ocupada como fracción del área total."""

# ── Visualización ─────────────────────────────────────────────────────────────

ROBOT_VISUAL_SCALE: float = 1.5
"""
Multiplicador del radio visual del robot en la animación Pygame.

1.5 es apropiado para 20 robots en un área de 10×10 m con pantalla de 800 px.
Aumentar para pocos robots, reducir para enjambres grandes.
"""

SHOW_ROBOT_IDS: bool  = True
"""Mostrar número de ID sobre cada robot en la animación."""

SHOW_ZONES: bool      = False
"""Mostrar radios de percepción RAOI alrededor de cada robot."""

SHOW_TRAIL: bool      = False
"""Mostrar rastro de trayectoria de los últimos TRAIL_LENGTH pasos."""

TRAIL_LENGTH: int     = 15
"""Número de pasos mostrados en el rastro de trayectoria."""

ANIMATION_INTERVAL: int = 100
"""
Milisegundos entre frames en la animación (FuncAnimation / Pygame).

100 ms = 10 fps (reproducción acelerada).
1000 ms = 1 fps = tiempo real (1 iteración ≡ 1 segundo físico simulado).
"""

FARMING_ANIMATION_INTERVAL: int = 20
"""
Milisegundos entre frames en la animación de farming (Pygame).

Farming usa un área 100x100 m con miles de iteraciones, por lo que se
reproduce a mayor velocidad (20 ms = 50 fps). La simulación física no
cambia — solo la velocidad de reproducción de la animación.
"""

SCREEN_SIZE: int       = 800
"""Tamaño de la ventana Pygame en píxeles (cuadrada)."""

VIDEO_SAVE_PATH: str   = "simulation.mp4"
"""
Ruta del archivo de video grabado con OpenCV.

Cambiar a None para desactivar la grabación.
Requiere ffmpeg instalado para el codec mp4v.
"""

FIGURE_SIZE: tuple     = (10, 10)
"""Tamaño de la figura Matplotlib para snapshots estáticos (pulgadas)."""

STATE_COLORS: dict = {
    0: "gray",    # Sin vecinos — exploración libre
    1: "red",     # Repulsión activa
    2: "blue",    # Atracción activa
    3: "green",   # Orientación activa
    4: "orange",  # Influencia activa
}
"""Colores Matplotlib por estado RAOI, usados en snapshots estáticos."""


# ══════════════════════════════════════════════════════════════════════════════
# Parámetros exclusivos de la tarea de FORAGING
# ══════════════════════════════════════════════════════════════════════════════
# Referencia:
#   Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
#   based on repulsion, attraction and influence. Autonomous Robots.

FORAGING_ITERS_PER_TRIP: int = 400
"""
Iteraciones estimadas por viaje completo de un robot (búsqueda + traslado + retorno).

Con área 10×10 m, velocidad ~0.3 m/s y DT=1.0 s, un viaje en condiciones
ideales toma ~150 iters. El valor 400 incluye un factor de seguridad ×2.5
para absorber tiempo de búsqueda y congestión entre robots.

El límite real se calcula en run() como:
  max_iter = max(FORAGING_MIN_ITER, ⌈n_objects / individuals⌉ × ITERS_PER_TRIP)
"""

FORAGING_MIN_ITER: int = 500
"""
Piso mínimo de iteraciones para foraging, independientemente del cálculo dinámico.

Evita límites demasiado bajos con muchos robots y pocos objetos.
"""

# ── Nest ──────────────────────────────────────────────────────────────────────

FORAGING_NEST_AREA_SIDE: float = 0.2 * AREA_LIMITS
"""
Lado del área cuadrada del nest (m).

Los robots depositan objetos en posiciones aleatorias dentro de este cuadrado,
evitando superposición en el punto exacto del nest.
"""

FORAGING_NEST_POSITION: list = [
    FORAGING_NEST_AREA_SIDE / 2,
    FORAGING_NEST_AREA_SIDE / 2,
]
"""Posición [x, y] del centro del nest (m). Esquina suroeste del área."""

FORAGING_NEST_RADIUS: float = 4.0
"""
Radio de detección del nest (m).

Un robot con objeto detecta el nest cuando está dentro de este radio
y en su campo de visión de influencia.
"""

FORAGING_DEPOSIT_RADIUS: float = 0.4
"""
Radio de depósito efectivo (m).

Un robot deposita el objeto cuando su distancia al nest es ≤ este valor.
Debe ser mayor que el desplazamiento por iteración (~0.3 m/iter con DT=1.0).
"""

# ── Objectbox ─────────────────────────────────────────────────────────────────

FORAGING_BOX_CENTER: float = 0.75
"""
Centro normalizado del objectbox (fracción de AREA_LIMITS).

Con AREA_LIMITS=10: centro en (7.5, 7.5).
"""

FORAGING_BOX_LIMITS: float = 0.2
"""
Mitad del lado normalizado del objectbox.

Con BOX_CENTER=0.75 y BOX_LIMITS=0.2:
  objetos distribuidos en x ∈ [5.5, 9.5], y ∈ [5.5, 9.5].
"""

FORAGING_BOX_RADIUS: float = 2.5
"""
Radio de detección del objectbox (m).

Un robot sin objeto detecta la zona de caja cuando está dentro de este radio
y en su campo de visión de influencia.
"""

FORAGING_PICK_RADIUS: float = 0.4
"""
Radio de recolección efectivo (m).

Un robot recoge un objeto disponible cuando su distancia al objeto es ≤ este valor.
Debe ser mayor que el desplazamiento por iteración (~0.3 m/iter con DT=1.0)
para garantizar que el robot no pase por encima del objeto sin detectarlo.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros exclusivos de la tarea de FARMING
# ══════════════════════════════════════════════════════════════════════════════
# Referencia:
#   Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
#   based on repulsion, attraction and influence. Autonomous Robots.

# ── Parcelas ──────────────────────────────────────────────────────────────────

FARMING_PLOT_LENGTH: float = 75.0
"""
Longitud de cada segmento de parcela (m).

El segmento se centra en X dentro del área. Con FARMING_AREA_LIMITS=100:
el segmento va de x=(10-6)/2=2.0 a x=8.0.
Debe ser < AREA_LIMITS para dejar pasillos laterales navegables.
"""

FARMING_PLOT_REPULSION: float = 0.3
"""
Radio de repulsión virtual de las parcelas (m).

Los robots detectan la parcela como fuente repulsiva cuando su distancia
al punto más cercano del segmento es ≤ este valor. Debe ser ≥ pick_radius
para evitar que un robot intente recoger un objeto y colisione con la parcela.
"""

FARMING_STRIP_WIDTH: float = 1.0

FARMING_PLOT_SEPARATION: float = 4
"""
Distancia centro a centro entre parcelas adyacentes (m).

Con STRIP_WIDTH=4.0 y PLOT_REPULSION=1.0:
  corredor libre entre franjas = 12 - 2*4 - 2*1 = 2 m — suficiente para robots Khepera III.
"""

FARMING_MAX_PLOTS: int = 20
"""
Número máximo de parcelas que caben en el área con los parámetros actuales.

Calculado como: floor((FARMING_AREA_LIMITS - 2*FARMING_PLOT_MARGIN_Y - 2*(STRIP_WIDTH+REPULSION))
                      / SEPARATION) + 1
Se recalcula automáticamente en run() y se usa para validar n_plots del usuario.
"""

FARMING_PLOT_MARGIN_Y: float = 15.0
"""
Margen en Y en los extremos del área (m).

Reserva espacio para el nest (esquina SW) y para maniobra libre de robots
en los bordes superior e inferior sin interferir con las parcelas.
"""

"""
Ancho de la franja de objetos a cada lado de la parcela (m).

Los objetos se distribuyen aleatoriamente en y ∈ [y_plot + REPULSION, y_plot + REPULSION + STRIP_WIDTH]
(franja superior) y simétricamente en la franja inferior.
"""

# ── Nest ──────────────────────────────────────────────────────────────────────

FARMING_NEST_AREA_SIDE: float = 0.2 * FARMING_AREA_LIMITS
"""Lado del área cuadrada del nest (m). Idéntico a foraging."""

FARMING_NEST_POSITION: list = [
    FARMING_NEST_AREA_SIDE / 2,
    FARMING_NEST_AREA_SIDE / 2,
]
"""Posición [x, y] del centro del nest (m). Esquina suroeste del área."""

FARMING_NEST_RADIUS: float = 40.0
"""
Radio de detección del nest (m).

En farming los objetos pueden estar en cualquier punto del área, por lo que
el radio del nest debe cubrir la diagonal completa (~14.1 m para área 10×10).
Un valor de 15.0 garantiza que cualquier robot con objeto siempre detecte el nest.
"""

FARMING_DEPOSIT_RADIUS: float = 0.5
"""
Radio de depósito efectivo (m).
Debe ser > desplazamiento por iteración (~0.3 m/iter con DT=1.0).
"""

# ── Recolección ───────────────────────────────────────────────────────────────

FARMING_PICK_RADIUS: float = 0.4
"""
Radio de recolección efectivo (m).

Debe ser > desplazamiento por iteración (~0.3 m/iter con DT=1.0) para
garantizar que el robot no pase por encima del objeto sin detectarlo.
Calibrado igual que FORAGING_PICK_RADIUS para consistencia entre variantes.
"""

FARMING_INFLUENCE_RADIUS: float = 8.0
"""
Radio de deteccion de la senal de influencia hacia el objeto objetivo (m).

Cubre aproximadamente una franja mas el corredor lateral
(FARMING_PLOT_SEPARATION + FARMING_STRIP_WIDTH + 3.0 m de holgura).
Un radio local fuerza exploracion real: cada robot solo ve objetos cercanos
y el enjambre se distribuye naturalmente por el area sin coordinacion
explicita. Un radio de 20 m producia convergencia masiva al mismo objeto.
"""

FARMING_CORRIDOR_RADIUS: float = 12.0
"""
Radio de deteccion de la senal de influencia hacia el corredor lateral (m).

Cuando el robot no tiene objetos visibles en su franja actual, recibe una
senal de influencia hacia el pasillo izquierdo (x < plot.x0) de la franja
mas cercana con objetos disponibles. Este radio debe ser mayor que
FARMING_INFLUENCE_RADIUS para que el robot siempre tenga una senal de
navegacion de area aunque no vea objetos individuales.
Analogo al box_radius de foraging: senala una zona, no un objeto puntual.
"""

# ── Limite de iteraciones ──────────────────────────────────────────────────────

FARMING_ITERS_PER_TRIP: int = 8000
"""
Iteraciones estimadas por viaje completo (busqueda + traslado + retorno).

Con area 100x100 m, diagonal ~141 m y velocidad ~0.3 m/s, un viaje completo
en peor caso ronda 940 iters solo de movimiento. Con rodeos obligatorios por
parcelas y congestion entre robots el overhead real puede duplicar ese valor.
El factor 8000 da margen suficiente para escenarios densos y bajo ratio
robots/objetos. Se escala adicionalmente por n_plots con PLOTS_TIME_FACTOR.
"""

FARMING_MIN_ITER: int = 2000
"""Piso minimo de iteraciones para farming."""

FARMING_PLOTS_TIME_FACTOR: float = 2.5
"""
Factor multiplicador de max_iter por cada parcela presente en el escenario.

Con muchas parcelas los robots deben rodear mas barreras, lo que incrementa
el tiempo efectivo por viaje. El max_iter se escala como:
  max_iter_final = max_iter_base * (1 + (n_plots - 1) * PLOTS_TIME_FACTOR / 10)
Con 1 parcela el factor es 1.0 (sin cambio). Con 5 parcelas y factor=2.5:
  escalado = 1 + 4 * 2.5 / 10 = 2.0 --> doble de tiempo base.
"""

FARMING_CHASE_TIMEOUT: int = 500
"""
Iteraciones maximas que un robot persigue el mismo objeto sin recogerlo.

Si un robot mantiene target_object[i] >= 0 durante este numero de iteraciones
sin lograr recoger el objeto, libera el target y reasigna al siguiente objeto
disponible mas cercano en una franja diferente. Evita que objetos de franjas
internas queden inalcanzables de forma indefinida.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros de la tarea de AGREGACIÓN (extensión MICAI 2026)
# ══════════════════════════════════════════════════════════════════════════════
# Anteriormente en config_ext.py. Consolidados aquí para mantener una sola
# fuente de verdad en todo el proyecto.

# ── Valores por defecto de simulación ────────────────────────────────────────

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