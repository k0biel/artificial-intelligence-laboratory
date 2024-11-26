import time

from exceptions import GameplayException
from connect4 import Connect4
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from alphabetaagent import AlphaBetaAgent

connect4 = Connect4(width=7, height=6)
# agent1 = MinMaxAgent('o', False)
# agent2 = MinMaxAgent('x', True)
# agent1 = MinMaxAgent('o', True)
# agent2 = AlphaBetaAgent('x', True)
agent1 = AlphaBetaAgent('o', True)
agent2 = MinMaxAgent('x', True)
round_number = 1
start_time = time.time()
while not connect4.game_over:
    print("=" * 20)
    print(f"Round: {round_number}")
    connect4.draw()
    try:
        if connect4.who_moves == agent1.my_token:
            n_column = agent1.decide(connect4)
        else:
            n_column = agent2.decide(connect4)
        connect4.drop_token(n_column)
        round_number += 1
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()

end_time = time.time()
game_duration = end_time - start_time

print(f"Total_rounds: {round_number - 1}")
print(f"Time: {game_duration} s")