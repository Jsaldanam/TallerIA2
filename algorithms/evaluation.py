import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)
    layout = state.layout
    defender = state.defender_position
    intruder = state.intruder_position
    pending = state.pending_terminals
    max_distance = layout.width + layout.height
    if pending:
        nearest_terminal = min(layout.distance(defender, t) for t in pending)
        if nearest_terminal == float("inf"):
            nearest_terminal = max_distance  
    else:
        nearest_terminal = 0  
        
    intruder_distance = layout.distance(defender, intruder)
    if intruder_distance == float("inf"):
        intruder_distance = max_distance
    mobility = len(state.get_legal_actions(0))
    terminal_term = -20.0 * (nearest_terminal / max(max_distance, 1))
    pending_term = -50.0 * len(pending)
    danger_term = -80.0 if intruder_distance <= 1 else 0.0
    safety_term = 5.0 * (min(intruder_distance, max_distance) / max(max_distance, 1))

    mobility_term = 2.0 * mobility
    raw = state.get_score() + terminal_term + pending_term + danger_term + safety_term + mobility_term
    return 900.0 * math.tanh(raw / 200.0)
    return base_evaluation_function(state)
