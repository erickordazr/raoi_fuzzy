# -*- coding: utf-8 -*-
"""
**************************************************************************
 *   Copyright (C) 2023 Erick Ordaz                                      *
 *   erick.ordazrv@uanl.edu.mx                                           *
 *                                                                       *
 *   Módulo de visualización — Aggregation task                          *
 *                                                                       *
 *   Language: Python                                                    *
 *   Rev: 5.0  (Pygame + OpenCV)                                         *
 **************************************************************************

Estrategia v5.0:
  - Pygame para renderizado: robots como polígonos rotados correctamente
  - OpenCV para grabación de video (.mp4)
  - Matplotlib solo para snapshots estáticos (publicación)
  - Robot diferencial: cuerpo + 2 ruedas + flecha de orientación
    Todo construido como vértices rotados → sin artefactos de transformación
  - Capas opcionales desde config.py:
      SHOW_ZONES  → sectores de percepción RAOI
      SHOW_TRAIL  → rastro de los últimos TRAIL_LENGTH pasos
"""

import math
import os
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from collections import deque
from typing import Optional

from . import config


# ── Paleta ────────────────────────────────────────────────────────────────────

def _hex(h: str) -> tuple:
    """Convierte color hex a tupla RGB para pygame."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# Colores RGB
BG_COLOR       = (255, 255, 255)
BORDER_COLOR   = ( 40,  40,  40)
GRID_COLOR     = (220, 220, 220)
SPAWN_COLOR    = (230, 230, 230)
SPAWN_BORDER   = (160, 160, 160)
INFLUENCE_COL  = (200,  40,  50)
TEXT_COLOR     = ( 30,  30,  30)
TRAIL_ALPHA    = 120   # 0-255

STATE_RGB = {
    0: (160, 160, 160),   # Sin vecinos   — gris
    1: (220,  50,  60),   # Repulsión     — rojo
    2: ( 60, 120, 170),   # Atracción     — azul
    3: ( 38, 160, 140),   # Orientación   — verde
    4: (220, 185,  80),   # Influencia    — dorado
}
STATE_LABELS = {
    0: "Free exploration",
    1: "Repulsion",
    2: "Attraction",
    3: "Orientation",
    4: "Influence",
}
ZONE_RGBA = {
    "repulsion":   (220,  50,  60, 35),
    "orientation": ( 38, 160, 140, 20),
    "attraction":  ( 60, 120, 170, 15),
}

# Colores para múltiples estímulos de influencia.
# Cada estímulo se identifica con un color distinto del ramp para que el
# usuario pueda seguir visualmente qué subgrupo del enjambre converge a cuál.
STIMULUS_COLORS = [
    (200,  40,  50),   # rojo (estímulo 1 — equivalente al INFLUENCE_COL legacy)
    ( 70, 130, 200),   # azul
    ( 90, 180, 100),   # verde
    (230, 160,  40),   # naranja
    (160,  80, 200),   # violeta
    ( 40, 180, 180),   # cian
    (210, 110, 160),   # rosa
    (130, 130,  60),   # oliva
]

# Color para obstáculos estáticos.
# Gris oscuro para distinguirlos claramente del fondo, los robots y los estímulos.
OBSTACLE_COLOR = (105, 105, 115)


# ── Geometría del robot ───────────────────────────────────────────────────────

def _rotate_points(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rota un array de puntos (N,2) alrededor del origen."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _robot_polygons(cx: float, cy: float, theta: float, r: float) -> dict:
    """
    Calcula los vértices del icono de robot diferencial en píxeles.

    v6.1 — Cuerpo circular con ruedas gruesas y nariz frontal triangular.
    La nariz actúa como indicador de dirección integrado al cuerpo.
    La flecha sale desde la punta de la nariz hacia adelante.

    Args:
        cx, cy: Centro en píxeles.
        theta:  Orientación en radianes (coordenadas de simulación).
        r:      Radio visual del cuerpo en píxeles.

    Returns:
        Dict con:
          'body_center'  — (cx, cy, r) para pygame.draw.circle
          'nose'         — triángulo de nariz frontal (polígono)
          'wheel_l'      — rectángulo rueda izquierda
          'wheel_r'      — rectángulo rueda derecha
          'arrow'        — ((x0,y0), (x1,y1)) flecha desde nariz
    """
    # ── Nariz frontal: triángulo que sale del círculo hacia adelante ──────────
    nose_pts = np.array([
        [ r * 1.45,  0.0      ],   # punta
        [ r * 0.80,  r * 0.38],   # base derecha  (tangente al círculo)
        [ r * 0.80, -r * 0.38],   # base izquierda
    ])

    # ── Ruedas: gruesas y prominentes ─────────────────────────────────────────
    wl  = r * 1.10   # largo (cubre longitud del cuerpo)
    wh  = r * 0.45   # ancho — prominente y visible
    wy  = r * 1.00   # distancia lateral al borde interno

    wl_pts = np.array([          # rueda izquierda (+y local)
        [-wl/2,  wy      ],
        [ wl/2,  wy      ],
        [ wl/2,  wy + wh ],
        [-wl/2,  wy + wh ],
    ])
    wr_pts = np.array([          # rueda derecha (-y local)
        [-wl/2, -wy - wh],
        [ wl/2, -wy - wh],
        [ wl/2, -wy     ],
        [-wl/2, -wy     ],
    ])

    # ── Flecha: desde punta de nariz hacia adelante ───────────────────────────
    arrow_start = np.array([[r * 1.45, 0.0]])
    arrow_end   = np.array([[r * 2.10, 0.0]])

    # Rotar todo con -theta (pygame Y invertido)
    angle  = -theta
    center = np.array([cx, cy])

    nose_r  = _rotate_points(nose_pts,    angle) + center
    wl_r    = _rotate_points(wl_pts,      angle) + center
    wr_r    = _rotate_points(wr_pts,      angle) + center
    arr_s   = (_rotate_points(arrow_start, angle) + center)[0]
    arr_e   = (_rotate_points(arrow_end,   angle) + center)[0]

    return {
        "body_center": (cx, cy, r),
        "nose":        nose_r.tolist(),
        "wheel_l":     wl_r.tolist(),
        "wheel_r":     wr_r.tolist(),
        "arrow":       (arr_s, arr_e),
    }


# ── Conversión mundo → pantalla ───────────────────────────────────────────────

class WorldToScreen:
    """
    Convierte coordenadas de simulación (metros) a píxeles de pantalla.
    Pygame tiene Y=0 arriba, la simulación tiene Y=0 abajo.
    """
    def __init__(self, area_m: float, screen_px: int, margin_px: int = 40):
        self.area_m    = area_m
        self.screen_px = screen_px
        self.margin    = margin_px
        self.drawable  = screen_px - 2 * margin_px
        self.scale     = self.drawable / area_m   # px/m

    def xy(self, xm: float, ym: float) -> tuple[int, int]:
        """Metro → píxel (con Y invertido)."""
        px = int(self.margin + xm * self.scale)
        py = int(self.margin + (self.area_m - ym) * self.scale)
        return px, py

    def r(self, rm: float) -> int:
        """Radio en metros → radio en píxeles."""
        return max(1, int(rm * self.scale))


# ── Dibujo de un frame en pygame ──────────────────────────────────────────────

def _draw_frame(
    surf: pygame.Surface,
    w2s: WorldToScreen,
    frame_data: dict,
    trails: list,
    font_sm,
    font_lg,
    font_title,
    environment: dict,
    show_zones: bool,
    show_trail: bool,
    frame_idx: int,
    total_frames: int,
    n_robots: int,
) -> None:
    """
    Renderiza un frame completo sobre 'surf'.

    Args:
        surf:         Superficie pygame.
        w2s:          Conversor mundo→pantalla.
        frame_data:   Dict con 'positions', 'orientations', 'states'.
        trails:       Lista de deques con historial de posiciones por robot.
        font_*:       Fuentes pygame cargadas.
        environment:  Dict del escenario.
        show_zones:   Mostrar radios RAOI.
        show_trail:   Mostrar rastro.
        frame_idx:    Índice del frame actual.
        total_frames: Total de frames de la simulación.
        n_robots:     Número de robots.
    """
    surf.fill(BG_COLOR)

    area_m  = environment["area_limits"]
    inf_r   = environment["influence_radius"]
    spawn_m = area_m * config.SPAWN_FRACTION

    # Compatibilidad: aceptar 'stimuli' (lista) o 'influence_position' (legacy)
    stimuli = environment.get("stimuli")
    if stimuli is None:
        legacy_pos = environment.get("influence_position")
        stimuli = [{"x": legacy_pos[0], "y": legacy_pos[1]}] if legacy_pos else []

    obstacles = environment.get("obstacles", [])

    # ── Grid sutil ────────────────────────────────────────────────────────────
    step = 1.0   # cada metro
    v = 0.0
    while v <= area_m:
        x0, y0 = w2s.xy(v, 0)
        x1, y1 = w2s.xy(v, area_m)
        pygame.draw.line(surf, GRID_COLOR, (x0, y0), (x1, y1), 1)
        x0, y0 = w2s.xy(0, v)
        x1, y1 = w2s.xy(area_m, v)
        pygame.draw.line(surf, GRID_COLOR, (x0, y0), (x1, y1), 1)
        v += step

    # ── Zona de spawn ─────────────────────────────────────────────────────────
    sx0, sy0 = w2s.xy(0, spawn_m)
    sx1, sy1 = w2s.xy(spawn_m, 0)
    pygame.draw.rect(surf, SPAWN_COLOR,
                     (sx0, sy0, sx1 - sx0, sy1 - sy0))
    pygame.draw.rect(surf, SPAWN_BORDER,
                     (sx0, sy0, sx1 - sx0, sy1 - sy0), 1)

    # ── Fuentes de influencia ────────────────────────────────────────────────
    # El círculo alrededor del estímulo representa r_s (intensidad del estímulo).
    # La zona sensorial del robot r_I se dibuja alrededor de cada robot (show_zones).
    # Un robot detecta el estímulo cuando dist(robot, estímulo) ≤ r_I + r_s.
    if len(stimuli) > 0:
        for k, stim in enumerate(stimuli):
            stim_col = STIMULUS_COLORS[k % len(STIMULUS_COLORS)]
            sx, sy = float(stim["x"]), float(stim["y"])
            icx, icy = w2s.xy(sx, sy)
            r_s    = float(stim.get("r", 1.0))
            rs_px  = max(4, w2s.r(r_s))
            # Relleno semitransparente del radio del estímulo (r_s)
            rs_surf = pygame.Surface((rs_px*2, rs_px*2), pygame.SRCALPHA)
            pygame.draw.circle(rs_surf, (*stim_col, 30), (rs_px, rs_px), rs_px)
            surf.blit(rs_surf, (icx - rs_px, icy - rs_px))
            # Borde del radio del estímulo
            pygame.draw.circle(surf, stim_col, (icx, icy), rs_px, 2)
            # Punto central con halo blanco
            pygame.draw.circle(surf, (255, 255, 255), (icx, icy), 8)
            pygame.draw.circle(surf, stim_col,        (icx, icy), 6)
            # Etiqueta numérica si hay más de un estímulo
            if len(stimuli) > 1:
                label = font_sm.render(str(k + 1), True, stim_col)
                surf.blit(label, (icx + rs_px + 4, icy - 14))

    # ── Obstáculos estáticos ──────────────────────────────────────────────────
    # Círculos sólidos en gris oscuro, distinguibles del enjambre y del fondo.
    for obs in obstacles:
        ox, oy, orad = float(obs["x"]), float(obs["y"]), float(obs["r"])
        ocx, ocy = w2s.xy(ox, oy)
        or_px    = max(2, w2s.r(orad))
        # Sombra
        pygame.draw.circle(surf, (185, 185, 185), (ocx + 2, ocy + 2), or_px)
        # Cuerpo
        pygame.draw.circle(surf, OBSTACLE_COLOR, (ocx, ocy), or_px)
        # Borde oscuro
        pygame.draw.circle(surf, (25, 25, 25), (ocx, ocy), or_px, 1)
        # Patrón cruzado interno para identificar como obstáculo, no robot
        cross_r = max(2, or_px // 2)
        pygame.draw.line(surf, (60, 60, 60),
                         (ocx - cross_r, ocy), (ocx + cross_r, ocy), 2)
        pygame.draw.line(surf, (60, 60, 60),
                         (ocx, ocy - cross_r), (ocx, ocy + cross_r), 2)

    # ── Borde del área ────────────────────────────────────────────────────────
    bx0, by0 = w2s.xy(0, area_m)
    bx1, by1 = w2s.xy(area_m, 0)
    pygame.draw.rect(surf, BORDER_COLOR,
                     (bx0, by0, bx1 - bx0, by1 - by0), 2)

    # ── Robots ────────────────────────────────────────────────────────────────
    positions    = frame_data["positions"]
    orientations = frame_data["orientations"]
    states       = frame_data["states"]

    r_rep_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_repulsion"])
    r_ori_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_orientation"])
    r_att_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_attraction"])
    body_px  = w2s.r(config.ROBOT_BODY_RADIUS * config.ROBOT_VISUAL_SCALE)

    for i in range(n_robots):
        xm, ym = positions[i]
        theta  = orientations[i]
        state  = int(states[i])
        color  = STATE_RGB.get(state, (128, 128, 128))
        cx, cy = w2s.xy(xm, ym)

        # Actualizar rastro
        trails[i].append((cx, cy))

        # ── Zonas de percepción (bajo el robot) ───────────────────────────────
        # r_I se dibuja aquí, alrededor del robot, igual que R/O/A.
        # Es la zona sensorial con la que el robot detecta estímulos.
        if show_zones:
            r_inf_px = w2s.r(inf_r)
            inf_zone_surf = pygame.Surface((r_inf_px*2+2, r_inf_px*2+2), pygame.SRCALPHA)
            pygame.draw.circle(inf_zone_surf, (255, 215, 0, 18),
                               (r_inf_px+1, r_inf_px+1), r_inf_px)
            pygame.draw.circle(inf_zone_surf, (255, 215, 0, 60),
                               (r_inf_px+1, r_inf_px+1), r_inf_px, 1)
            surf.blit(inf_zone_surf, (cx - r_inf_px - 1, cy - r_inf_px - 1))

            for (r_px, rgba, fov_key) in [
                (r_rep_px, ZONE_RGBA["repulsion"],   "fov_repulsion"),
                (r_ori_px, ZONE_RGBA["orientation"], "fov_orientation"),
                (r_att_px, ZONE_RGBA["attraction"],  "fov_attraction"),
            ]:
                fov = config.RAOI_FOV[fov_key]
                zone_surf = pygame.Surface((r_px*2+2, r_px*2+2), pygame.SRCALPHA)
                if fov >= 2*math.pi - 0.01:
                    pygame.draw.circle(zone_surf, rgba,
                                       (r_px+1, r_px+1), r_px)
                else:
                    # Sector: dibujar como polígono de arco
                    start_angle = -theta - fov/2
                    pts = [(r_px+1, r_px+1)]
                    steps = max(20, int(math.degrees(fov)))
                    for k in range(steps + 1):
                        a = start_angle + fov * k / steps
                        pts.append((
                            r_px+1 + r_px * math.cos(a),
                            r_px+1 + r_px * math.sin(a),
                        ))
                    if len(pts) > 2:
                        pygame.draw.polygon(zone_surf, rgba, pts)
                surf.blit(zone_surf, (cx - r_px - 1, cy - r_px - 1))

        # ── Rastro — gris neutro con degradado (v6.0) ───────────────────────
        # Color independiente del estado: siempre gris para no confundir
        # cambios de estado con la trayectoria histórica del robot.
        if show_trail and len(trails[i]) >= 2:
            pts = list(trails[i])
            n_pts = len(pts)
            for k in range(n_pts - 1):
                alpha = int((k + 1) / n_pts * TRAIL_ALPHA)
                tr_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                pygame.draw.line(
                    tr_surf,
                    (130, 130, 130, alpha),   # gris neutro fijo
                    pts[k], pts[k+1], 2,
                )
                surf.blit(tr_surf, (0, 0))

        # ── Icono del robot v6.1 — cuerpo circular ───────────────────────────
        polys  = _robot_polygons(cx, cy, theta, body_px)
        to_int = lambda pts: [(int(x), int(y)) for x, y in pts]
        bx, by, br = polys["body_center"]

        # Sombra del cuerpo circular (desplazada 2px)
        pygame.draw.circle(surf, (185, 185, 185), (bx + 2, by + 2), br)

        # Ruedas gruesas (wh=0.45r) — debajo del cuerpo
        wheel_color = (35,  35,  35)
        wheel_band  = (100, 100, 100)
        for wkey in ("wheel_l", "wheel_r"):
            wpts = polys[wkey]
            pygame.draw.polygon(surf, wheel_color, to_int(wpts))
            # Banda de rodadura central
            mid_l = ((wpts[0][0]+wpts[3][0])/2, (wpts[0][1]+wpts[3][1])/2)
            mid_r = ((wpts[1][0]+wpts[2][0])/2, (wpts[1][1]+wpts[2][1])/2)
            pygame.draw.line(surf, wheel_band,
                             (int(mid_l[0]), int(mid_l[1])),
                             (int(mid_r[0]), int(mid_r[1])), 1)
            pygame.draw.polygon(surf, (60, 60, 60), to_int(wpts), 1)

        # Cuerpo circular coloreado por estado
        pygame.draw.circle(surf, color,  (bx, by), br)
        pygame.draw.circle(surf, (25, 25, 25), (bx, by), br, 1)

        # Nariz frontal — triángulo más claro que el cuerpo
        r_c, g_c, b_c = color
        nose_color = (min(255, r_c+45), min(255, g_c+45), min(255, b_c+45))
        pygame.draw.polygon(surf, nose_color, to_int(polys["nose"]))
        pygame.draw.polygon(surf, (25, 25, 25), to_int(polys["nose"]), 1)

        # Eje central (hub)
        hub_r = max(2, br // 5)
        pygame.draw.circle(surf, (245, 245, 245), (bx, by), hub_r)
        pygame.draw.circle(surf, (40,  40,  40),  (bx, by), hub_r, 1)

        # Flecha integrada desde punta de nariz
        arr_s, arr_e = polys["arrow"]
        pygame.draw.line(surf, (15, 15, 15),
                         (int(arr_s[0]), int(arr_s[1])),
                         (int(arr_e[0]), int(arr_e[1])), 2)
        tip_x, tip_y = int(arr_e[0]), int(arr_e[1])
        tip_ang      = -theta
        tip_size     = max(5, br // 2)
        tip_pts = [
            (tip_x, tip_y),
            (int(tip_x - tip_size * math.cos(tip_ang - 0.42)),
             int(tip_y - tip_size * math.sin(tip_ang - 0.42))),
            (int(tip_x - tip_size * math.cos(tip_ang + 0.42)),
             int(tip_y - tip_size * math.sin(tip_ang + 0.42))),
        ]
        pygame.draw.polygon(surf, (15, 15, 15), tip_pts)

        # ID numérico sobre el robot
        if config.SHOW_ROBOT_IDS:
            id_surf = font_sm.render(str(i), True, (20, 20, 20))
            id_x    = cx - id_surf.get_width() // 2
            id_y    = cy - br - id_surf.get_height() - 1
            bg_w    = id_surf.get_width()  + 4
            bg_h    = id_surf.get_height() + 2
            bg      = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 160))
            surf.blit(bg,      (id_x - 2, id_y - 1))
            surf.blit(id_surf, (id_x,     id_y))

    # ── HUD: título, iteración, leyenda ───────────────────────────────────────
    W = surf.get_width()

    # Barra superior semitransparente
    hud_surf = pygame.Surface((W, 36), pygame.SRCALPHA)
    hud_surf.fill((240, 240, 240, 210))
    surf.blit(hud_surf, (0, 0))

    title_txt = font_title.render(
        "RAOI Swarm Simulator — Aggregation Task", True, TEXT_COLOR)
    surf.blit(title_txt, (10, 6))

    iter_txt = font_lg.render(
        f"Iteration {frame_idx+1:>4} / {total_frames}   |   "
        f"N = {n_robots} robots   |   "
        f"Stimuli = {len(stimuli)}   |   "
        f"Obstacles = {len(obstacles)}",
        True, (80, 80, 80))
    surf.blit(iter_txt, (W - iter_txt.get_width() - 12, 10))

    # ── Leyenda de estados ────────────────────────────────────────────────────
    leg_x, leg_y = 10, 44
    leg_surf = pygame.Surface((200, len(STATE_RGB)*22 + 10), pygame.SRCALPHA)
    leg_surf.fill((255, 255, 255, 200))
    surf.blit(leg_surf, (leg_x - 4, leg_y - 4))

    for state_id, label in STATE_LABELS.items():
        rgb = STATE_RGB[state_id]
        pygame.draw.circle(surf, rgb, (leg_x + 8, leg_y + 8), 7)
        pygame.draw.circle(surf, (30, 30, 30), (leg_x + 8, leg_y + 8), 7, 1)
        txt = font_sm.render(label, True, TEXT_COLOR)
        surf.blit(txt, (leg_x + 20, leg_y + 1))
        leg_y += 22

    # ── Barra de progreso ─────────────────────────────────────────────────────
    H       = surf.get_height()
    bar_h   = 6
    bar_y   = H - bar_h - 2
    bar_w   = int(W * (frame_idx + 1) / total_frames)
    pygame.draw.rect(surf, (210, 210, 210), (0, bar_y, W, bar_h))
    pygame.draw.rect(surf, STATE_RGB[3],    (0, bar_y, bar_w, bar_h))


# ── Función principal de animación ────────────────────────────────────────────

def animate_report(
    report: np.ndarray,
    environment: dict,
    interval: int = 100,
    show_zones: bool = False,
    show_trail: bool = False,
    trail_length: int = 15,
    save_path: str = "simulation.mp4",
    screen_size: int = 800,
    dpi: int = 150,
) -> None:
    """
    Anima el enjambre con Pygame y graba el video con OpenCV.

    La simulación debe haber corrido completa antes de llamar esta función.
    Abre una ventana Pygame interactiva y simultáneamente graba el video.

    Args:
        report:       Estado completo, shape (iterations, N, ≥8).
        environment:  Dict con claves:
                        'area_limits'       — lado del área cuadrada (m).
                        'influence_radius'  — radio de detección de estímulos (m).
                        'stimuli'           — lista [{'x','y'}, ...] (opcional).
                        'obstacles'         — lista [{'x','y','r'}, ...] (opcional).
                        'influence_position' — [x,y] (legacy, sólo si no hay 'stimuli').
        interval:     ms entre frames (default 100 ms ≈ 10 fps).
        show_zones:   Mostrar radios de percepción RAOI.
        show_trail:   Mostrar rastro de trayectoria.
        trail_length: Pasos del rastro.
        save_path:    Ruta del video de salida (.mp4). None para no guardar.
        screen_size:  Tamaño de la ventana en píxeles (cuadrada).
        dpi:          No usado en pygame (compatibilidad con llamadas antiguas).
    """
    iterations = report.shape[0]
    n_robots   = report.shape[1]

    # ── Inicializar Pygame ────────────────────────────────────────────────────
    os.environ.setdefault("SDL_VIDEODRIVER", "")   # pantalla real si disponible
    pygame.init()
    pygame.display.set_caption("RAOI Swarm Simulator")

    try:
        screen = pygame.display.set_mode((screen_size, screen_size))
        headless = False
    except Exception:
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        pygame.init()
        screen = pygame.Surface((screen_size, screen_size))
        headless = True

    w2s   = WorldToScreen(environment["area_limits"], screen_size, margin_px=50)
    clock = pygame.time.Clock()

    # Fuentes
    pygame.font.init()
    try:
        font_sm    = pygame.font.SysFont("DejaVu Sans", 13)
        font_lg    = pygame.font.SysFont("DejaVu Sans", 14)
        font_title = pygame.font.SysFont("DejaVu Sans Bold", 15)
    except Exception:
        font_sm = font_lg = font_title = pygame.font.Font(None, 16)

    # Historial para rastros
    trails = [deque(maxlen=trail_length) for _ in range(n_robots)]

    # ── Video writer (OpenCV) ─────────────────────────────────────────────────
    writer = None
    if save_path:
        fps    = max(1, int(1000 / interval))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps,
                                 (screen_size, screen_size))
        print(f"Recording video → '{save_path}'  ({fps} fps)")

    # ── Loop de animación ─────────────────────────────────────────────────────
    running = True
    frame   = 0

    while running and frame < iterations:
        # Eventos pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Pausa
                    paused = True
                    while paused:
                        for e2 in pygame.event.get():
                            if e2.type == pygame.KEYDOWN and e2.key == pygame.K_SPACE:
                                paused = False
                            if e2.type == pygame.QUIT:
                                paused = False
                                running = False
                        clock.tick(10)

        frame_data = {
            "positions":    report[frame, :, :2],
            "orientations": report[frame, :, 3],
            "states":       report[frame, :, 7],
        }

        _draw_frame(
            screen, w2s, frame_data, trails,
            font_sm, font_lg, font_title,
            environment, show_zones, show_trail,
            frame, iterations, n_robots,
        )

        if not headless:
            pygame.display.flip()

        # Capturar frame para video
        if writer is not None:
            px_array = pygame.surfarray.array3d(screen)
            # pygame → (W,H,3) con RGB, opencv quiere (H,W,3) BGR
            frame_bgr = cv2.cvtColor(
                np.transpose(px_array, (1, 0, 2)), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr)

        clock.tick(1000 // max(1, interval))
        frame += 1

        if frame % max(1, iterations // 10) == 0:
            pct = int(frame / iterations * 100)
            print(f"  Animating... {pct}%", end="\r")

    print(f"\nAnimation complete ({frame} frames rendered).")

    if writer is not None:
        writer.release()
        print(f"Video saved: {save_path}")

    pygame.quit()


# ── Snapshots estáticos (matplotlib — para publicación) ───────────────────────

def plot_snapshots(
    report: np.ndarray,
    environment: dict,
    show_zones: bool = False,
    show_trail: bool = False,
    trail_length: int = 15,
    filename: str = "snapshots.png",
    dpi: int = 300,
) -> None:
    """
    Genera figura estática publicable con 3 snapshots (matplotlib).

    Args:
        report:       Estado completo, shape (iterations, N, 8).
        environment:  Dict del escenario.
        show_zones:   Mostrar radios de percepción.
        show_trail:   Mostrar rastro.
        trail_length: Pasos del rastro.
        filename:     Archivo PNG de salida.
        dpi:          Resolución.
    """
    iterations  = report.shape[0]
    n_robots    = report.shape[1]
    indices     = [0, iterations // 2, iterations - 1]
    labels      = ["t = 0", f"t = {iterations // 2}", f"t = {iterations - 1}"]

    area_limits = environment["area_limits"]
    inf_r       = environment.get("influence_radius", 0.0)
    stimuli     = environment.get("stimuli") or []
    if not stimuli:
        legacy = environment.get("influence_position")
        if legacy:
            stimuli = [{"x": legacy[0], "y": legacy[1], "r": 1.0}]
    obstacles   = environment.get("obstacles", [])
    spawn_side  = area_limits * config.SPAWN_FRACTION
    robot_scale = config.ROBOT_BODY_RADIUS * 2.5

    r_rep = config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_repulsion"]
    r_ori = config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_orientation"]
    r_att = config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_attraction"]

    STIM_COLORS_MPL = [
        "#E63946", "#2A9D8F", "#E9C46A", "#457B9D",
        "#F4A261", "#6A4C93", "#2DC653",
    ]

    HEX_PALETTE = {
        0: "#AAAAAA", 1: "#DC3232",
        2: "#3C78AA", 3: "#26A08C", 4: "#DCC050",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "RAOI Swarm Simulator — Aggregation Task | Snapshots",
        fontsize=12, fontweight="bold", color="#222222",
    )

    for ax, idx, label in zip(axes, indices, labels):
        ax.set_facecolor("white")

        # Spawn
        ax.add_patch(patches.Rectangle(
            (0, 0), spawn_side, spawn_side,
            edgecolor="#AAAAAA", facecolor="#F5F5F5",
            linewidth=1.0, linestyle=":", zorder=0,
        ))

        # Obstáculos
        for obs in obstacles:
            ax.add_patch(plt.Circle(
                (float(obs["x"]), float(obs["y"])), float(obs["r"]),
                facecolor="#888888", alpha=0.85, zorder=2,
                edgecolor="#333333", linewidth=0.8,
            ))

        # Estímulos: r_s alrededor del estímulo (no r_I del robot)
        for k, stim in enumerate(stimuli):
            scol = STIM_COLORS_MPL[k % len(STIM_COLORS_MPL)]
            sx, sy = float(stim["x"]), float(stim["y"])
            r_s   = float(stim.get("r", 1.0))
            ax.add_patch(plt.Circle(
                (sx, sy), r_s,
                facecolor=scol, alpha=0.10, zorder=0,
            ))
            ax.add_patch(plt.Circle(
                (sx, sy), r_s,
                fill=False, edgecolor=scol,
                linewidth=1.5, linestyle="--", zorder=1,
            ))
            ax.plot(sx, sy, "*", color=scol, markersize=12, zorder=4)
            if len(stimuli) > 1:
                ax.text(sx + r_s + 0.1, sy, str(k + 1),
                        color=scol, fontsize=8, va="center", zorder=5)

        positions    = report[idx, :, :2]
        orientations = report[idx, :, 3]
        states       = report[idx, :, 7].astype(int)

        for i in range(n_robots):
            x, y  = positions[i]
            theta = orientations[i]
            color = HEX_PALETTE.get(states[i], "#AAAAAA")

            if show_zones:
                # r_I — zona sensorial del robot para detectar estímulos
                ax.add_patch(plt.Circle(
                    (x, y), inf_r,
                    facecolor="#FFD700", alpha=0.05, zorder=1,
                    edgecolor="#FFD700", linewidth=0.5, linestyle=":"
                ))
                for (r_z, zcol, fov_key, alpha) in [
                    (r_rep, "#DC3232", "fov_repulsion",   0.12),
                    (r_ori, "#26A08C", "fov_orientation", 0.07),
                    (r_att, "#3C78AA", "fov_attraction",  0.05),
                ]:
                    fov = config.RAOI_FOV[fov_key]
                    if fov >= 2*math.pi - 0.01:
                        ax.add_patch(plt.Circle(
                            (x, y), r_z, facecolor=zcol,
                            alpha=alpha, zorder=1,
                            edgecolor=zcol, linewidth=0.5, linestyle="--",
                        ))
                    else:
                        ax.add_patch(patches.Wedge(
                            (x, y), r_z,
                            math.degrees(theta - fov/2),
                            math.degrees(theta + fov/2),
                            facecolor=zcol, alpha=alpha,
                            edgecolor=zcol, linewidth=0.5,
                            linestyle="--", zorder=1,
                        ))

            if show_trail and idx > 0:
                trail = [(report[k, i, 0], report[k, i, 1])
                         for k in range(max(0, idx - trail_length), idx + 1)]
                if len(trail) >= 2:
                    n_tr = len(trail)
                    for k in range(n_tr - 1):
                        alpha = (k + 1) / n_tr * 0.5
                        ax.plot([trail[k][0], trail[k+1][0]],
                                [trail[k][1], trail[k+1][1]],
                                color=color, alpha=alpha, linewidth=0.8, zorder=2)

            # Robot: círculo + flecha
            ax.add_patch(plt.Circle(
                (x, y), robot_scale,
                facecolor=color, edgecolor="#222222",
                linewidth=0.8, zorder=3, alpha=0.92,
            ))
            ax.annotate(
                "", xy=(x + math.cos(theta)*robot_scale*1.7,
                        y + math.sin(theta)*robot_scale*1.7),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#111111",
                                lw=0.8, mutation_scale=10),
                zorder=5,
            )

        ax.set(xlim=(-0.3, area_limits+0.3), ylim=(-0.3, area_limits+0.3))
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10, color="#444444")
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.tick_params(labelsize=8)

        # Leyenda
        handles = [
            mlines.Line2D([], [], marker="o", linestyle="None",
                          markersize=7, markerfacecolor=HEX_PALETTE[s],
                          markeredgecolor="#333", markeredgewidth=0.5,
                          label=STATE_LABELS[s])
            for s in STATE_LABELS
        ]
        ax.legend(handles=handles, loc="upper left",
                  fontsize=7, framealpha=0.9, title="State", title_fontsize=7.5)

    plt.tight_layout()
    fig.savefig(filename, dpi=dpi, facecolor="white")
    print(f"Snapshots saved: {filename}")
    plt.show()



"""
EXTENSIÓN DE visualization.py — Animación de la tarea de foraging.

INSTRUCCIONES DE INTEGRACIÓN:
  Pegar este bloque al final de raoi_simulator/visualization.py, antes de
  la función save_figure(). No modificar las funciones existentes.

Diferencias respecto a animate_report():
  - Recibe objects_report (shape T, O, 2) para dibujar objetos en movimiento.
  - Dibuja nest (rojo) y objectbox (azul) como zonas separadas.
  - HUD muestra contador de objetos entregados en tiempo real.
  - Objetos cargados se resaltan con un aro blanco sobre el robot.
"""

# ── Colores adicionales para foraging ─────────────────────────────────────────
NEST_COL      = ( 30, 140,  80)   # verde oscuro — nest
OBJECTBOX_COL = ( 60, 120, 200)   # azul          — objectbox
OBJECT_COL    = (255, 140,   0)   # naranja brillante — objeto disponible
OBJECT_CARRIED= (255, 220,  40)   # amarillo dorado   — objeto en tránsito
OBJECT_DELIV  = (200, 200, 200)   # gris claro        — entregado (en nest)


def _draw_foraging_environment(
    surf: pygame.Surface,
    w2s: "WorldToScreen",
    env: dict,
) -> None:
    """
    Dibuja las zonas estáticas del escenario de foraging: nest y objectbox.

    Args:
        surf: Superficie pygame.
        w2s:  Conversor mundo→pantalla.
        env:  Dict del escenario (claves: nest_position, nest_radius,
              nest_area_side, objectbox_center, box_radius,
              box_center, box_limits, area_limits).
    """
    area_m = env["area_limits"]

    # ── Objectbox: círculo de influencia + rectángulo de spawn de objetos ──────
    box_cx, box_cy = env["objectbox_center"]
    box_r          = env["box_radius"]
    bcx, bcy       = w2s.xy(box_cx, box_cy)
    br_px          = w2s.r(box_r)

    box_surf = pygame.Surface((br_px * 2, br_px * 2), pygame.SRCALPHA)
    pygame.draw.circle(box_surf, (*OBJECTBOX_COL, 18), (br_px, br_px), br_px)
    surf.blit(box_surf, (bcx - br_px, bcy - br_px))
    pygame.draw.circle(surf, OBJECTBOX_COL, (bcx, bcy), br_px, 2)

    # Rectángulo de distribución de objetos
    box_c = env["box_center"]
    box_l = env["box_limits"]
    bx0m  = (box_c - box_l / 2) * area_m
    by0m  = (box_c - box_l / 2) * area_m
    bx1m  = (box_c + box_l / 2) * area_m
    by1m  = (box_c + box_l / 2) * area_m
    px0, py0 = w2s.xy(bx0m, by1m)
    px1, py1 = w2s.xy(bx1m, by0m)
    pygame.draw.rect(surf, OBJECTBOX_COL,
                     (px0, py0, px1 - px0, py1 - py0), 1)

    # ── Nest: círculo de influencia + rectángulo de zona ──────────────────────
    nx, ny = env["nest_position"]
    n_r    = env["nest_radius"]
    n_side = env["nest_area_side"]
    ncx, ncy = w2s.xy(nx, ny)
    nr_px    = w2s.r(n_r)

    nest_surf = pygame.Surface((nr_px * 2, nr_px * 2), pygame.SRCALPHA)
    pygame.draw.circle(nest_surf, (*NEST_COL, 18), (nr_px, nr_px), nr_px)
    surf.blit(nest_surf, (ncx - nr_px, ncy - nr_px))
    pygame.draw.circle(surf, NEST_COL, (ncx, ncy), nr_px, 2)

    # Rectángulo del nest
    npx0, npy0 = w2s.xy(0,      n_side)
    npx1, npy1 = w2s.xy(n_side, 0)
    pygame.draw.rect(surf, NEST_COL,
                     (npx0, npy0, npx1 - npx0, npy1 - npy0), 2)

    # Icono del nest (estrella)
    pygame.draw.circle(surf, NEST_COL, (ncx, ncy), 6)


def _draw_objects(
    surf: pygame.Surface,
    w2s: "WorldToScreen",
    objects_pos: np.ndarray,
    carried_set: set,
    nest_pos: np.ndarray,
    nest_area_side: float,
) -> None:
    """
    Dibuja los objetos del entorno distinguiendo tres estados visuales.

    Estados:
      - Naranja brillante : disponible en el objectbox, esperando ser recogido.
      - Omitido           : cargado por un robot — se dibuja en el loop de robots.
      - Gris claro        : depositado en el nest (posición fija en cuadrícula).

    Los objetos cargados se omiten aquí para evitar duplicación visual;
    se renderizan directamente sobre el cuerpo del robot en el loop principal.

    Args:
        surf:           Superficie pygame.
        w2s:            Conversor mundo→pantalla.
        objects_pos:    Posiciones de objetos en el frame, shape (O, 2).
        carried_set:    Conjunto de índices de objetos actualmente cargados.
        nest_pos:       Posición [x, y] del centro del nest (m).
        nest_area_side: Lado del área del nest (m), para detectar entregados.
    """
    obj_r_px = max(5, w2s.r(0.14))

    for o in range(objects_pos.shape[0]):
        # Objeto cargado: se dibuja encima del robot, no aquí
        if o in carried_set:
            continue

        ox, oy   = objects_pos[o]
        opx, opy = w2s.xy(ox, oy)

        # Determinar si está entregado (dentro del área del nest)
        in_nest = (
            abs(ox - nest_pos[0]) <= nest_area_side / 2 + 0.1
            and abs(oy - nest_pos[1]) <= nest_area_side / 2 + 0.1
        )

        if in_nest:
            col      = OBJECT_DELIV
            border   = (120, 120, 120)
        else:
            col      = OBJECT_COL
            border   = (180, 80, 0)

        pygame.draw.circle(surf, col,    (opx, opy), obj_r_px)
        pygame.draw.circle(surf, border, (opx, opy), obj_r_px, 2)


def _draw_foraging_hud(
    surf: pygame.Surface,
    font_sm,
    font_lg,
    font_title,
    frame_idx: int,
    total_frames: int,
    n_robots: int,
    n_objects: int,
    delivered_count: int,
) -> None:
    """
    Dibuja el HUD específico de foraging: título, iteración y contador de objetos.

    Args:
        surf:            Superficie pygame.
        font_*:          Fuentes pygame.
        frame_idx:       Índice del frame actual.
        total_frames:    Total de frames.
        n_robots:        Número de robots.
        n_objects:       Total de objetos.
        delivered_count: Objetos entregados hasta el frame actual.
    """
    W = surf.get_width()
    H = surf.get_height()

    hud_surf = pygame.Surface((W, 36), pygame.SRCALPHA)
    hud_surf.fill((240, 240, 240, 210))
    surf.blit(hud_surf, (0, 0))

    title_txt = font_title.render(
        "RAOI Swarm Simulator — Foraging Task", True, TEXT_COLOR)
    surf.blit(title_txt, (10, 6))

    status = f"Iteration {frame_idx+1:>4} / {total_frames}   |   " \
             f"N = {n_robots}   |   Objects {delivered_count}/{n_objects}"
    iter_txt = font_lg.render(status, True, (80, 80, 80))
    surf.blit(iter_txt, (W - iter_txt.get_width() - 12, 10))

    # Barra de progreso de tarea (objetos entregados)
    bar_h = 6
    bar_y = H - bar_h - 2
    bar_w = int(W * delivered_count / max(n_objects, 1))
    pygame.draw.rect(surf, (210, 210, 210), (0, bar_y, W, bar_h))
    pygame.draw.rect(surf, NEST_COL, (0, bar_y, bar_w, bar_h))


def animate_foraging(
    report: np.ndarray,
    objects_report: np.ndarray,
    carrying_report: np.ndarray,
    env: dict,
    interval: int = 100,
    show_zones: bool = False,
    show_trail: bool = False,
    trail_length: int = 15,
    save_path: str = "foraging.mp4",
    screen_size: int = 800,
) -> None:
    """
    Anima la tarea de foraging con Pygame y graba el video con OpenCV.

    Extiende animate_report() añadiendo: objetos en movimiento, zona de
    nest (verde), zona de objectbox (azul) y HUD con contador de entregas.

    Args:
        report:          Estado del enjambre, shape (T, N, 8).
        objects_report:  Posiciones de objetos, shape (T, O, 2).
        carrying_report: Índice de objeto cargado por cada robot, shape (T, N).
                         -1 si el robot no lleva objeto. Permite dibujar el
                         objeto sobre el robot correcto sin duplicación.
        env:             Dict del escenario con claves:
                           'area_limits', 'nest_position', 'nest_radius',
                           'nest_area_side', 'objectbox_center', 'box_radius',
                           'box_center', 'box_limits'.
        interval:        ms entre frames (default 100 ms ≈ 10 fps).
        show_zones:      Mostrar radios de percepción RAOI.
        show_trail:      Mostrar rastro de trayectoria.
        trail_length:    Pasos del rastro.
        save_path:       Ruta del video de salida (.mp4). None para no guardar.
        screen_size:     Tamaño de la ventana en píxeles (cuadrada).
    """
    iterations = report.shape[0]
    n_robots   = report.shape[1]
    n_objects  = objects_report.shape[1]

    # ── Inicializar Pygame ────────────────────────────────────────────────────
    os.environ.setdefault("SDL_VIDEODRIVER", "")
    pygame.init()
    pygame.display.set_caption("RAOI Swarm Simulator — Foraging")

    try:
        screen  = pygame.display.set_mode((screen_size, screen_size))
        headless = False
    except Exception:
        os.environ["SDL_VIDEODRIVER"] = "offscreen"
        pygame.init()
        screen   = pygame.Surface((screen_size, screen_size))
        headless = True

    w2s   = WorldToScreen(env["area_limits"], screen_size, margin_px=50)
    clock = pygame.time.Clock()

    pygame.font.init()
    try:
        font_sm    = pygame.font.SysFont("DejaVu Sans", 13)
        font_lg    = pygame.font.SysFont("DejaVu Sans", 14)
        font_title = pygame.font.SysFont("DejaVu Sans Bold", 15)
    except Exception:
        font_sm = font_lg = font_title = pygame.font.Font(None, 16)

    trails = [deque(maxlen=trail_length) for _ in range(n_robots)]

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if save_path:
        fps    = max(1, int(1000 / interval))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps,
                                 (screen_size, screen_size))
        print(f"Recording video → '{save_path}'  ({fps} fps)")

    # ── Precalcular entregas acumuladas por frame para el HUD ─────────────────
    # Un objeto se considera "entregado" cuando deja de moverse (posición fija).
    # Aproximación simple: comparar posición en t vs t-1 para objetos que
    # están en la zona del nest.
    nest_pos   = np.array(env["nest_position"])
    nest_r     = env["nest_radius"]
    nest_side  = env["nest_area_side"]

    delivered_per_frame = np.zeros(iterations, dtype=int)
    for t in range(iterations):
        in_nest = 0
        for o in range(n_objects):
            ox, oy = objects_report[t, o]
            if (abs(ox - nest_pos[0]) <= nest_side / 2 + 0.3
                    and abs(oy - nest_pos[1]) <= nest_side / 2 + 0.3):
                in_nest += 1
        delivered_per_frame[t] = in_nest

    # ── Radios en píxeles para zonas RAOI ────────────────────────────────────
    r_rep_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_repulsion"])
    r_ori_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_orientation"])
    r_att_px = w2s.r(config.ROBOT_BODY_RADIUS + config.RAOI_RADII["r_attraction"])
    body_px  = w2s.r(config.ROBOT_BODY_RADIUS * config.ROBOT_VISUAL_SCALE)

    # ── Loop de animación ─────────────────────────────────────────────────────
    running = True
    frame   = 0
    spawn_m = env["area_limits"] * config.SPAWN_FRACTION

    while running and frame < iterations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = True
                    while paused:
                        for e2 in pygame.event.get():
                            if e2.type == pygame.KEYDOWN and e2.key == pygame.K_SPACE:
                                paused = False
                            if e2.type == pygame.QUIT:
                                paused = False
                                running = False
                        clock.tick(10)

        # ── Fondo y grid ──────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        step_g = 1.0
        v = 0.0
        while v <= env["area_limits"]:
            x0, y0 = w2s.xy(v, 0);            x1, y1 = w2s.xy(v, env["area_limits"])
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            x0, y0 = w2s.xy(0, v);            x1, y1 = w2s.xy(env["area_limits"], v)
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            v += step_g

        # Zona de spawn
        sx0, sy0 = w2s.xy(0, spawn_m)
        sx1, sy1 = w2s.xy(spawn_m, 0)
        pygame.draw.rect(screen, SPAWN_COLOR,
                         (sx0, sy0, sx1 - sx0, sy1 - sy0))
        pygame.draw.rect(screen, SPAWN_BORDER,
                         (sx0, sy0, sx1 - sx0, sy1 - sy0), 1)

        # Borde del área
        bx0, by0 = w2s.xy(0, env["area_limits"])
        bx1, by1 = w2s.xy(env["area_limits"], 0)
        pygame.draw.rect(screen, BORDER_COLOR,
                         (bx0, by0, bx1 - bx0, by1 - by0), 2)

        # Zonas de foraging (nest + objectbox)
        _draw_foraging_environment(screen, w2s, env)

        # Objetos: omitir los cargados (se dibujan sobre el robot)
        carried_set = {
            int(carrying_report[frame, i])
            for i in range(n_robots)
            if carrying_report[frame, i] >= 0
        }
        _draw_objects(
            screen, w2s,
            objects_report[frame],
            carried_set,
            nest_pos, env["nest_area_side"],
        )

        # ── Robots (igual que _draw_frame) ────────────────────────────────────
        positions    = report[frame, :, :2]
        orientations = report[frame, :, 3]
        states       = report[frame, :, 7]

        for i in range(n_robots):
            xm, ym = positions[i]
            theta  = orientations[i]
            state  = int(states[i])
            color  = STATE_RGB.get(state, (128, 128, 128))
            cx, cy = w2s.xy(xm, ym)

            trails[i].append((cx, cy))

            if show_zones:
                for (r_px, rgba, fov_key) in [
                    (r_rep_px, ZONE_RGBA["repulsion"],   "fov_repulsion"),
                    (r_ori_px, ZONE_RGBA["orientation"], "fov_orientation"),
                    (r_att_px, ZONE_RGBA["attraction"],  "fov_attraction"),
                ]:
                    fov_v = config.RAOI_FOV[fov_key]
                    zone_s = pygame.Surface((r_px*2+2, r_px*2+2), pygame.SRCALPHA)
                    if fov_v >= 2*math.pi - 0.01:
                        pygame.draw.circle(zone_s, rgba, (r_px+1, r_px+1), r_px)
                    else:
                        start_a = -theta - fov_v / 2
                        pts = [(r_px+1, r_px+1)]
                        steps = max(20, int(math.degrees(fov_v)))
                        for k in range(steps + 1):
                            a = start_a + fov_v * k / steps
                            pts.append((r_px+1 + r_px*math.cos(a),
                                        r_px+1 + r_px*math.sin(a)))
                        if len(pts) > 2:
                            pygame.draw.polygon(zone_s, rgba, pts)
                    screen.blit(zone_s, (cx - r_px - 1, cy - r_px - 1))

            if show_trail and len(trails[i]) >= 2:
                pts_t = list(trails[i])
                n_pts = len(pts_t)
                for k in range(n_pts - 1):
                    alpha  = int((k + 1) / n_pts * TRAIL_ALPHA)
                    tr_s   = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(tr_s, (130, 130, 130, alpha),
                                     pts_t[k], pts_t[k+1], 2)
                    screen.blit(tr_s, (0, 0))

            polys  = _robot_polygons(cx, cy, theta, body_px)
            to_int = lambda pts: [(int(x), int(y)) for x, y in pts]
            bx, by, br = polys["body_center"]

            pygame.draw.circle(screen, (185, 185, 185), (bx + 2, by + 2), br)

            wheel_color = (35, 35, 35)
            for wkey in ("wheel_l", "wheel_r"):
                wpts = polys[wkey]
                pygame.draw.polygon(screen, wheel_color, to_int(wpts))
                pygame.draw.polygon(screen, (60, 60, 60), to_int(wpts), 1)

            pygame.draw.circle(screen, color, (bx, by), br)
            pygame.draw.circle(screen, (25, 25, 25), (bx, by), br, 1)

            r_c, g_c, b_c = color
            nose_color = (min(255, r_c+45), min(255, g_c+45), min(255, b_c+45))
            pygame.draw.polygon(screen, nose_color, to_int(polys["nose"]))
            pygame.draw.polygon(screen, (25, 25, 25), to_int(polys["nose"]), 1)

            hub_r = max(2, br // 5)
            pygame.draw.circle(screen, (245, 245, 245), (bx, by), hub_r)
            pygame.draw.circle(screen, (40,  40,  40),  (bx, by), hub_r, 1)

            arr_s, arr_e = polys["arrow"]
            pygame.draw.line(screen, (15, 15, 15),
                             (int(arr_s[0]), int(arr_s[1])),
                             (int(arr_e[0]), int(arr_e[1])), 2)
            tip_x, tip_y = int(arr_e[0]), int(arr_e[1])
            tip_ang  = -theta
            tip_size = max(5, br // 2)
            tip_pts  = [
                (tip_x, tip_y),
                (int(tip_x - tip_size*math.cos(tip_ang - 0.42)),
                 int(tip_y - tip_size*math.sin(tip_ang - 0.42))),
                (int(tip_x - tip_size*math.cos(tip_ang + 0.42)),
                 int(tip_y - tip_size*math.sin(tip_ang + 0.42))),
            ]
            pygame.draw.polygon(screen, (15, 15, 15), tip_pts)

            # Visual de transporte: objeto naranja en la nariz del robot.
            # Se dibuja DESPUÉS del cuerpo para quedar encima de todo.
            # El color del robot NO cambia — sigue el estado RAOI normal.
            # carrying_report indica qué objeto lleva cada robot (-1 = ninguno).
            if carrying_report[frame, i] >= 0:
                nose_x = int(bx + body_px * 1.45 * math.cos(-theta))
                nose_y = int(by + body_px * 1.45 * math.sin(-theta))
                obj_r  = max(5, body_px // 2)
                # Sombra del objeto
                pygame.draw.circle(screen, (140, 70, 0),
                                   (nose_x + 2, nose_y + 2), obj_r)
                # Objeto naranja brillante
                pygame.draw.circle(screen, OBJECT_COL,    (nose_x, nose_y), obj_r)
                pygame.draw.circle(screen, OBJECT_CARRIED,(nose_x, nose_y), obj_r, 2)

            if config.SHOW_ROBOT_IDS:
                id_surf = font_sm.render(str(i), True, (20, 20, 20))
                id_x    = cx - id_surf.get_width() // 2
                id_y    = cy - br - id_surf.get_height() - 1
                bg_w    = id_surf.get_width()  + 4
                bg_h    = id_surf.get_height() + 2
                bg_     = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
                bg_.fill((255, 255, 255, 160))
                screen.blit(bg_,     (id_x - 2, id_y - 1))
                screen.blit(id_surf, (id_x,     id_y))

        # ── Leyenda ───────────────────────────────────────────────────────────
        leg_x, leg_y = 10, 44
        leg_surf = pygame.Surface((200, len(STATE_RGB)*22 + 10), pygame.SRCALPHA)
        leg_surf.fill((255, 255, 255, 200))
        screen.blit(leg_surf, (leg_x - 4, leg_y - 4))
        for state_id, label in STATE_LABELS.items():
            rgb = STATE_RGB[state_id]
            pygame.draw.circle(screen, rgb, (leg_x + 8, leg_y + 8), 7)
            pygame.draw.circle(screen, (30, 30, 30), (leg_x + 8, leg_y + 8), 7, 1)
            txt = font_sm.render(label, True, TEXT_COLOR)
            screen.blit(txt, (leg_x + 20, leg_y + 1))
            leg_y += 22

        # ── HUD ───────────────────────────────────────────────────────────────
        _draw_foraging_hud(
            screen, font_sm, font_lg, font_title,
            frame, iterations, n_robots, n_objects,
            delivered_per_frame[frame],
        )

        if not headless:
            pygame.display.flip()

        if writer is not None:
            px_array  = pygame.surfarray.array3d(screen)
            frame_bgr = cv2.cvtColor(
                np.transpose(px_array, (1, 0, 2)), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr)

        clock.tick(1000 // max(1, interval))
        frame += 1

        if frame % max(1, iterations // 10) == 0:
            pct = int(frame / iterations * 100)
            print(f"  Animating... {pct}%", end="\r")

    print(f"\nAnimation complete ({frame} frames rendered).")

    if writer is not None:
        writer.release()
        print(f"Video saved: {save_path}")

    pygame.quit()

def save_figure(fig, filename: str = "aggregation_result.png", dpi: int = 600) -> None:
    """Guarda una figura matplotlib en disco."""
    fig.savefig(filename, format="png", dpi=dpi, facecolor="white")
    print(f"Figure saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualización exclusiva de la tarea de FARMING
# ══════════════════════════════════════════════════════════════════════════════

# Colores específicos de farming
PLOT_COLOR      = ( 60, 120,  50)   # verde oscuro — segmento de parcela
PLOT_FILL_COLOR = (200, 230, 190)   # verde claro  — franja de objetos
FRUIT_COLOR     = (220,  80,  30)   # naranja-rojo — fruto disponible
FRUIT_CARRIED   = (255, 140,  60)   # naranja claro — borde fruto cargado


def _draw_farming_environment(
    surf: pygame.Surface,
    w2s: WorldToScreen,
    env: dict,
) -> None:
    """
    Dibuja el entorno de farming: nest, parcelas y franjas de objetos.

    Args:
        surf: Superficie pygame.
        w2s:  Conversor mundo→pantalla.
        env:  Dict con 'nest_position', 'nest_radius', 'nest_area_side',
              'plots', 'plot_repulsion', 'strip_width'.
    """
    # ── Nest (idéntico al de foraging) ────────────────────────────────────────
    nest_pos   = env["nest_position"]
    nest_area  = env["nest_area_side"]
    nest_rad   = env.get("nest_radius", 4.0)

    # Radio de influencia del nest
    nr_px = w2s.r(nest_rad)
    nc_px = w2s.xy(nest_pos[0], nest_pos[1])
    nest_surf = pygame.Surface((nr_px * 2 + 2, nr_px * 2 + 2), pygame.SRCALPHA)
    pygame.draw.circle(nest_surf, (100, 180, 100, 35), (nr_px + 1, nr_px + 1), nr_px)
    surf.blit(nest_surf, (nc_px[0] - nr_px - 1, nc_px[1] - nr_px - 1))

    # Área del nest
    half = nest_area / 2
    n_lo = w2s.xy(nest_pos[0] - half, nest_pos[1] + half)
    n_hi = w2s.xy(nest_pos[0] + half, nest_pos[1] - half)
    n_w  = n_hi[0] - n_lo[0]
    n_h  = n_hi[1] - n_lo[1]
    nest_s = pygame.Surface((max(1, n_w), max(1, n_h)), pygame.SRCALPHA)
    nest_s.fill((80, 160, 80, 90))
    surf.blit(nest_s, n_lo)
    pygame.draw.rect(surf, (50, 130, 50), (n_lo[0], n_lo[1], max(1, n_w), max(1, n_h)), 2)

    # Etiqueta NEST
    try:
        font_nest = pygame.font.SysFont("Arial", 11, bold=True)
        lbl = font_nest.render("NEST", True, (30, 100, 30))
        surf.blit(lbl, (n_lo[0] + 2, n_lo[1] + 2))
    except Exception:
        pass

    # ── Franjas de objetos (fondo semitransparente) ───────────────────────────
    strip_width    = env.get("strip_width", 1.2)
    plot_repulsion = env.get("plot_repulsion", 0.4)
    margin         = plot_repulsion + 0.05

    for plot in env.get("plots", []):
        x0, x1, yp = plot["x0"], plot["x1"], plot["y"]

        # Franja superior
        yf_lo = yp + margin;       yf_hi = yp + margin + strip_width
        px_lo = w2s.xy(x0, yf_hi); px_hi = w2s.xy(x1, yf_lo)
        fw    = max(1, px_hi[0] - px_lo[0])
        fh    = max(1, px_hi[1] - px_lo[1])
        fs    = pygame.Surface((fw, fh), pygame.SRCALPHA)
        fs.fill((*PLOT_FILL_COLOR, 60))
        surf.blit(fs, px_lo)

        # Franja inferior
        yf_lo = yp - margin - strip_width; yf_hi = yp - margin
        px_lo = w2s.xy(x0, yf_hi);         px_hi = w2s.xy(x1, yf_lo)
        fw    = max(1, px_hi[0] - px_lo[0])
        fh    = max(1, px_hi[1] - px_lo[1])
        fs    = pygame.Surface((fw, fh), pygame.SRCALPHA)
        fs.fill((*PLOT_FILL_COLOR, 60))
        surf.blit(fs, px_lo)

    # ── Segmentos de parcela ──────────────────────────────────────────────────
    plot_rep_px = w2s.r(plot_repulsion)

    for plot in env.get("plots", []):
        x0, x1, yp = plot["x0"], plot["x1"], plot["y"]
        p0 = w2s.xy(x0, yp)
        p1 = w2s.xy(x1, yp)

        # Zona de repulsión semitransparente
        rep_w = p1[0] - p0[0]
        rep_h = max(1, plot_rep_px * 2)
        rep_s = pygame.Surface((max(1, rep_w), rep_h), pygame.SRCALPHA)
        rep_s.fill((200, 100, 50, 30))
        surf.blit(rep_s, (p0[0], p0[1] - plot_rep_px))

        # Línea principal de la parcela
        pygame.draw.line(surf, PLOT_COLOR, p0, p1, 4)

        # Marcas de extremo
        for px_pt in (p0, p1):
            pygame.draw.line(surf, PLOT_COLOR,
                             (px_pt[0], px_pt[1] - plot_rep_px),
                             (px_pt[0], px_pt[1] + plot_rep_px), 2)


def _draw_farming_objects(
    surf: pygame.Surface,
    w2s: WorldToScreen,
    obj_positions: np.ndarray,
    carried_set: set,
    nest_pos: list,
    nest_area: float,
) -> None:
    """
    Dibuja los frutos disponibles. Los cargados se omiten (se dibujan en el robot).
    Los entregados (dentro del área del nest) se dibujan en verde.

    Args:
        surf:          Superficie pygame.
        w2s:           Conversor mundo→pantalla.
        obj_positions: Posiciones actuales de objetos, shape (O, 2).
        carried_set:   Índices de objetos actualmente cargados.
        nest_pos:      Posición [x, y] del nest.
        nest_area:     Lado del área del nest (m).
    """
    half    = nest_area / 2
    obj_r   = max(5, w2s.r(0.15))

    for o, pos in enumerate(obj_positions):
        if o in carried_set:
            continue

        cx, cy   = w2s.xy(pos[0], pos[1])
        in_nest  = (abs(pos[0] - nest_pos[0]) <= half
                    and abs(pos[1] - nest_pos[1]) <= half)

        if in_nest:
            # Fruto entregado — verde pequeño
            pygame.draw.circle(surf, (60, 160, 60),  (cx, cy), max(3, obj_r - 2))
            pygame.draw.circle(surf, (30, 100, 30),  (cx, cy), max(3, obj_r - 2), 1)
        else:
            # Fruto disponible — naranja
            pygame.draw.circle(surf, (90, 45, 0),    (cx + 2, cy + 2), obj_r)  # sombra
            pygame.draw.circle(surf, FRUIT_COLOR,    (cx, cy), obj_r)
            pygame.draw.circle(surf, (255, 200, 100),(cx, cy), obj_r, 1)


def _draw_farming_hud(
    surf: pygame.Surface,
    font_sm,
    font_lg,
    font_title,
    frame: int,
    iterations: int,
    n_robots: int,
    n_objects: int,
    n_plots: int,
    delivered: int,
) -> None:
    """
    Dibuja el HUD de farming: frame, progreso de entrega y parámetros.

    Args:
        surf:       Superficie pygame.
        font_*:     Fuentes pygame.
        frame:      Frame actual.
        iterations: Total de frames.
        n_robots:   Número de robots.
        n_objects:  Número de objetos.
        n_plots:    Número de parcelas.
        delivered:  Objetos entregados en este frame.
    """
    W, H = surf.get_size()

    # Título
    title = font_title.render("RAOI — Farming Task", True, TEXT_COLOR)
    surf.blit(title, (W // 2 - title.get_width() // 2, 6))

    # Frame
    frame_txt = font_sm.render(f"Frame {frame + 1} / {iterations}", True, TEXT_COLOR)
    surf.blit(frame_txt, (W - frame_txt.get_width() - 10, 6))

    # Barra de progreso de entrega
    bar_w = 160; bar_h = 14
    bar_x = W - bar_w - 10
    bar_y = 26
    frac  = delivered / max(n_objects, 1)
    pygame.draw.rect(surf, (220, 220, 220), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(surf, FRUIT_COLOR,    (bar_x, bar_y, int(bar_w * frac), bar_h))
    pygame.draw.rect(surf, (80, 80, 80),   (bar_x, bar_y, bar_w, bar_h), 1)
    prog_txt = font_sm.render(f"{delivered}/{n_objects} fruits", True, TEXT_COLOR)
    surf.blit(prog_txt, (bar_x - prog_txt.get_width() - 6,
                          bar_y + (bar_h - prog_txt.get_height()) // 2))

    # Info inferior
    info = font_sm.render(
        f"Robots: {n_robots}   Plots: {n_plots}   Objects: {n_objects}",
        True, TEXT_COLOR,
    )
    surf.blit(info, (10, H - info.get_height() - 6))


def animate_farming(
    report:          np.ndarray,
    objects_report:  np.ndarray,
    carrying_report: np.ndarray,
    env:             dict,
    interval:        int   = 100,
    show_zones:      bool  = False,
    show_trail:      bool  = False,
    trail_length:    int   = 15,
    save_path:       Optional[str] = None,
    screen_size:     int   = 800,
) -> None:
    """
    Reproduce la animación Pygame de la tarea de farming.

    Renderiza robots, frutos, parcelas y nest cuadro a cuadro.
    Si save_path no es None, escribe también un video mp4 con OpenCV.

    Args:
        report:          Estado del enjambre, shape (T, N, 8).
        objects_report:  Posiciones de objetos, shape (T, O, 2).
        carrying_report: Índice del objeto cargado por cada robot, shape (T, N).
        env:             Dict del escenario con claves:
                           'area_limits', 'nest_position', 'nest_radius',
                           'nest_area_side', 'plots', 'plot_repulsion', 'strip_width'.
        interval:        Milisegundos entre frames.
        show_zones:      Mostrar radios RAOI alrededor de cada robot.
        show_trail:      Mostrar rastro de trayectoria.
        trail_length:    Número de pasos en el rastro.
        save_path:       Ruta del video de salida. None → sin grabación.
        screen_size:     Tamaño de la ventana en píxeles.
    """
    iterations, n_robots, _ = report.shape
    n_objects = objects_report.shape[1]
    n_plots   = len(env.get("plots", []))
    nest_pos  = env["nest_position"]
    nest_area = env["nest_area_side"]

    # Conteo de entregas por frame: objeto entregado = dentro del nest area
    half = nest_area / 2
    delivered_per_frame = np.array([
        int(np.sum(
            (np.abs(objects_report[t, :, 0] - nest_pos[0]) <= half) &
            (np.abs(objects_report[t, :, 1] - nest_pos[1]) <= half)
        ))
        for t in range(iterations)
    ])

    # ── Inicializar Pygame ────────────────────────────────────────────────────
    headless = (save_path is not None and not pygame.display.get_init())
    if not pygame.display.get_init():
        pygame.init()

    if save_path:
        screen = pygame.Surface((screen_size, screen_size))
    else:
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("RAOI — Farming Task")

    pygame.font.init()
    font_sm    = pygame.font.SysFont("Arial", 11)
    font_lg    = pygame.font.SysFont("Arial", 13, bold=True)
    font_title = pygame.font.SysFont("Arial", 14, bold=True)
    clock      = pygame.time.Clock()

    w2s      = WorldToScreen(env["area_limits"], screen_size)
    body_px  = max(6, w2s.r(config.ROBOT_BODY_RADIUS * config.ROBOT_VISUAL_SCALE))
    r_rep_px = w2s.r(config.RAOI_RADII["r_repulsion"])
    r_ori_px = w2s.r(config.RAOI_RADII["r_orientation"])
    r_att_px = w2s.r(config.RAOI_RADII["r_attraction"])
    trails   = [deque(maxlen=trail_length) for _ in range(n_robots)]

    # ── OpenCV writer ─────────────────────────────────────────────────────────
    writer = None
    if save_path:
        fps    = max(1, 1000 // max(1, interval))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (screen_size, screen_size))

    frame = 0
    running = True

    while running and frame < iterations:
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break

        # ── Fondo y grid ──────────────────────────────────────────────────────
        screen.fill(BG_COLOR)
        step_g = 1.0
        v = 0.0
        while v <= env["area_limits"]:
            x0, y0 = w2s.xy(v, 0);           x1, y1 = w2s.xy(v, env["area_limits"])
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            x0, y0 = w2s.xy(0, v);           x1, y1 = w2s.xy(env["area_limits"], v)
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            v += step_g

        # Borde del área
        bx0, by0 = w2s.xy(0, env["area_limits"])
        bx1, by1 = w2s.xy(env["area_limits"], 0)
        pygame.draw.rect(screen, BORDER_COLOR, (bx0, by0, bx1 - bx0, by1 - by0), 2)

        # Entorno farming (nest + parcelas + franjas)
        _draw_farming_environment(screen, w2s, env)

        # Frutos: omitir cargados (se dibujan en la nariz del robot)
        carried_set = {
            int(carrying_report[frame, i])
            for i in range(n_robots)
            if carrying_report[frame, i] >= 0
        }
        _draw_farming_objects(
            screen, w2s,
            objects_report[frame],
            carried_set,
            nest_pos, nest_area,
        )

        # ── Robots ────────────────────────────────────────────────────────────
        positions    = report[frame, :, :2]
        orientations = report[frame, :, 3]
        states       = report[frame, :, 7]

        for i in range(n_robots):
            xm, ym = positions[i]
            theta  = orientations[i]
            state  = int(states[i])
            color  = STATE_RGB.get(state, (128, 128, 128))
            cx, cy = w2s.xy(xm, ym)

            trails[i].append((cx, cy))

            if show_zones:
                for (r_px, rgba, fov_key) in [
                    (r_rep_px, ZONE_RGBA["repulsion"],   "fov_repulsion"),
                    (r_ori_px, ZONE_RGBA["orientation"], "fov_orientation"),
                    (r_att_px, ZONE_RGBA["attraction"],  "fov_attraction"),
                ]:
                    fov_v  = config.RAOI_FOV[fov_key]
                    zone_s = pygame.Surface((r_px * 2 + 2, r_px * 2 + 2), pygame.SRCALPHA)
                    if fov_v >= 2 * math.pi - 0.01:
                        pygame.draw.circle(zone_s, rgba, (r_px + 1, r_px + 1), r_px)
                    else:
                        start_a = -theta - fov_v / 2
                        pts = [(r_px + 1, r_px + 1)]
                        steps = max(20, int(math.degrees(fov_v)))
                        for k in range(steps + 1):
                            a = start_a + fov_v * k / steps
                            pts.append((r_px + 1 + r_px * math.cos(a),
                                        r_px + 1 + r_px * math.sin(a)))
                        if len(pts) > 2:
                            pygame.draw.polygon(zone_s, rgba, pts)
                    screen.blit(zone_s, (cx - r_px - 1, cy - r_px - 1))

            if show_trail and len(trails[i]) >= 2:
                pts_t = list(trails[i])
                n_pts = len(pts_t)
                for k in range(n_pts - 1):
                    alpha = int((k + 1) / n_pts * TRAIL_ALPHA)
                    tr_s  = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(tr_s, (130, 130, 130, alpha),
                                     pts_t[k], pts_t[k + 1], 2)
                    screen.blit(tr_s, (0, 0))

            polys  = _robot_polygons(cx, cy, theta, body_px)
            to_int = lambda pts: [(int(x), int(y)) for x, y in pts]
            bx, by, br = polys["body_center"]

            pygame.draw.circle(screen, (185, 185, 185), (bx + 2, by + 2), br)

            for wkey in ("wheel_l", "wheel_r"):
                wpts = polys[wkey]
                pygame.draw.polygon(screen, (35, 35, 35), to_int(wpts))
                pygame.draw.polygon(screen, (60, 60, 60), to_int(wpts), 1)

            pygame.draw.circle(screen, color, (bx, by), br)
            pygame.draw.circle(screen, (25, 25, 25), (bx, by), br, 1)

            r_c, g_c, b_c = color
            nose_color = (min(255, r_c + 45), min(255, g_c + 45), min(255, b_c + 45))
            pygame.draw.polygon(screen, nose_color, to_int(polys["nose"]))
            pygame.draw.polygon(screen, (25, 25, 25), to_int(polys["nose"]), 1)

            hub_r = max(2, br // 5)
            pygame.draw.circle(screen, (245, 245, 245), (bx, by), hub_r)
            pygame.draw.circle(screen, (40, 40, 40), (bx, by), hub_r, 1)

            arr_s, arr_e = polys["arrow"]
            pygame.draw.line(screen, (15, 15, 15),
                             (int(arr_s[0]), int(arr_s[1])),
                             (int(arr_e[0]), int(arr_e[1])), 2)
            tip_x, tip_y = int(arr_e[0]), int(arr_e[1])
            tip_ang  = -theta
            tip_size = max(5, br // 2)
            tip_pts  = [
                (tip_x, tip_y),
                (int(tip_x - tip_size * math.cos(tip_ang - 0.42)),
                 int(tip_y - tip_size * math.sin(tip_ang - 0.42))),
                (int(tip_x - tip_size * math.cos(tip_ang + 0.42)),
                 int(tip_y - tip_size * math.sin(tip_ang + 0.42))),
            ]
            pygame.draw.polygon(screen, (15, 15, 15), tip_pts)

            # Fruto en la nariz: visual de transporte
            if carrying_report[frame, i] >= 0:
                nose_x = int(bx + body_px * 1.45 * math.cos(-theta))
                nose_y = int(by + body_px * 1.45 * math.sin(-theta))
                obj_r  = max(5, body_px // 2)
                pygame.draw.circle(screen, (120, 50, 0),  (nose_x + 2, nose_y + 2), obj_r)
                pygame.draw.circle(screen, FRUIT_COLOR,   (nose_x, nose_y), obj_r)
                pygame.draw.circle(screen, FRUIT_CARRIED, (nose_x, nose_y), obj_r, 2)

            if config.SHOW_ROBOT_IDS:
                id_surf = font_sm.render(str(i), True, (20, 20, 20))
                id_x    = cx - id_surf.get_width() // 2
                id_y    = cy - br - id_surf.get_height() - 1
                bg_w    = id_surf.get_width() + 4
                bg_h    = id_surf.get_height() + 2
                bg_     = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
                bg_.fill((255, 255, 255, 160))
                screen.blit(bg_,     (id_x - 2, id_y - 1))
                screen.blit(id_surf, (id_x, id_y))

        # ── Leyenda ───────────────────────────────────────────────────────────
        leg_x, leg_y = 10, 44
        leg_surf = pygame.Surface((200, len(STATE_RGB) * 22 + 10), pygame.SRCALPHA)
        leg_surf.fill((255, 255, 255, 200))
        screen.blit(leg_surf, (leg_x - 4, leg_y - 4))
        for state_id, label in STATE_LABELS.items():
            rgb = STATE_RGB[state_id]
            pygame.draw.circle(screen, rgb, (leg_x + 8, leg_y + 8), 7)
            pygame.draw.circle(screen, (30, 30, 30), (leg_x + 8, leg_y + 8), 7, 1)
            txt = font_sm.render(label, True, TEXT_COLOR)
            screen.blit(txt, (leg_x + 20, leg_y + 1))
            leg_y += 22

        # ── HUD ───────────────────────────────────────────────────────────────
        _draw_farming_hud(
            screen, font_sm, font_lg, font_title,
            frame, iterations, n_robots, n_objects, n_plots,
            delivered_per_frame[frame],
        )

        if not headless:
            pygame.display.flip()

        if writer is not None:
            px_array  = pygame.surfarray.array3d(screen)
            frame_bgr = cv2.cvtColor(
                np.transpose(px_array, (1, 0, 2)), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr)

        clock.tick(1000 // max(1, interval))
        frame += 1

        if frame % max(1, iterations // 10) == 0:
            pct = int(frame / iterations * 100)
            print(f"  Animating... {pct}%", end="\r")

    print(f"\nAnimation complete ({frame} frames rendered).")

    if writer is not None:
        writer.release()
        print(f"Video saved: {save_path}")

    pygame.quit()