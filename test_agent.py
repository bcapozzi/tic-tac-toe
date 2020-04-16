from episode import Episode
from environment import Environment
import play as p
from rl_player import RLPlayer
from random_player import RandomPlayer

def test1():

    env = Environment(RandomPlayer('O'))
    agent = RLPlayer('X')
    episode = Episode(agent, env)

    board = p.empty_board()
    agent, final_board = episode.execute(board)

    return agent, final_board
