from abc import ABC, abstractmethod

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(propio, state: GameState) -> str | None:
        propio.nodes_evaluated = 0
        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        propio.nodes_evaluated += 1  
        alpha, beta = float("-inf"), float("inf")
        best_action = legal_actions[0]
        best_value = float("-inf")

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            value = propio._value(successor, 1, propio.depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

    def _value(
        propio, state: GameState, agent_index: int, depth: int, alpha: float, beta: float
    ) -> float:
        propio.nodes_evaluated += 1

        if state.is_win() or state.is_lose() or depth == 0:
            return evaluation_function(state)

        legal_actions = state.get_legal_actions(agent_index)
        next_agent = (agent_index + 1) % state.get_num_agents()

        if agent_index == 0:  
            value = float("-inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = max(value, propio._value(successor, next_agent, depth - 1, alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value
        else:  # MIN (intruso)
            value = float("inf")
            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)
                value = min(value, propio._value(successor, next_agent, depth - 1, alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value
        raise NotImplementedError("Punto 4: implemente MinimaxAgent.get_action")


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        # TODO: Add your code here
        raise NotImplementedError("Punto 5: implemente AlphaBetaAgent.get_action")
