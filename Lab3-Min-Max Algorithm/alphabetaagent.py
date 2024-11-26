import math
import copy
from exceptions import AgentException

class AlphaBetaAgent:
    def __init__(self, my_token='o', use_heuristic=True):
        # Initialize the agent with its token and whether to use heuristic
        self.my_token = my_token
        self.opponent_token = 'x' if my_token == 'o' else 'o'
        self.use_heuristic = use_heuristic

    def decide(self, connect4):
        # Initialize the agent with its token and whether to use heuristic
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')
        _, column = self.alphabeta(connect4, 2, -math.inf, math.inf, True)
        return column

    def alphabeta(self, connect4, depth, alpha, beta, maximizing):
        # Alpha-beta pruning algorithm to decide the next move
        # If the maximum depth is reached or the game is over, return the heuristic score
        if depth == 0:
            if self.use_heuristic:
                return self.heuristic(connect4), None
            else:
                return 0, None
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1, None
            elif connect4.wins == self.opponent_token:
                return -1, None
            else:
                return 0, None

        # If the agent is maximizing, find the move with the maximum score
        if maximizing:
            max_value = -math.inf
            best_column = None
            for n_column in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(n_column)
                value = self.alphabeta(connect4_copy, depth - 1, alpha, beta, False)[0]
                if value > max_value:
                    max_value = value
                    best_column = n_column
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return max_value, best_column
        else:
            # If the agent is minimizing, find the move with the minimum score
            min_value = math.inf
            best_column = None
            for n_column in connect4.possible_drops():
                connect4_copy = copy.deepcopy(connect4)
                connect4_copy.drop_token(n_column)
                value = self.alphabeta(connect4_copy, depth - 1, alpha, beta, True)[0]
                if value < min_value:
                    min_value = value
                    best_column = n_column
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value, best_column

    def heuristic(self, connect4):
        # Heuristic function to evaluate the state of the game
        score = 0
        for four in connect4.iter_fours():
            if four.count(self.my_token) == 3 and four.count('_') == 1:
                score += 5
            elif four.count(self.my_token) == 2 and four.count('_') == 2:
                score += 2
            elif four.count(self.opponent_token) == 3 and four.count('_') == 1:
                score -= 4
            elif four.count(self.opponent_token) == 2 and four.count('_') == 2:
                score -= 2
        return score / 100