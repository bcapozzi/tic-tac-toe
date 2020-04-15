# create an episode
from random_player import RandomPlayer
from rl_player import RLPlayer
import play as p
import numpy as np


def test_random_game():
    p1 = RandomPlayer('X')
    p2 = RandomPlayer('O')
    play_game(p1,p2)

def test_rl_player(num_samples):
    p1 = RLPlayer('X')
    p2 = RandomPlayer('O')

#    p1 = RandomPlayer('X')
#    p2 = RLPlayer('O')

    # play a number of games
    winners = []

    for i in range(0,num_samples):
        # flip a coin as to who gets to go first
        coin = np.random.uniform()

        file = open("games/game_" + str(i) + ".txt","w")
        if (coin < 0.5):
            winner = play_game(p1,p2,file)
        else:
            winner = play_game(p2,p1,file)

        file.close()

        print("AFTER GAME: ", i)
        p1.dump_q_values()
        winners.append(winner)


    x_wins = 0.0
    o_wins = 0.0
    cat_games = 0.0
    num_games = 0.0
    for i in range(0,len(winners)):
        num_games += 1.0
        winner = winners[i]
        if winner is None:
            print("NO WINNER FOR GAME ", i)
            cat_games += 1
        else:
            print("WINNER FOR GAME ", i, ": ", winner.player)

            if (winner.player == 'X'):
                x_wins += 1
            else:
                o_wins += 1

        if (num_games > 0) and (num_games % 100 == 0):
            x_percent = x_wins / num_games
            o_percent = o_wins / num_games
            cat_percent = cat_games / num_games

            # reset statistics
            num_games = 0
            x_wins = 0.0
            o_wins = 0.0
            cat_games = 0.0

            print("X: ", x_percent, " O: ", o_percent, " CAT: ", cat_percent)

    p1.finish()

def to_display_string(board):
    result = ""
    for row in board:
        result += str(row)
        result += "\n"

    return result

def play_game(p1, p2, file=None):

    board = p.empty_board()

    players = [p1, p2]
    current_player_index = 0
    winner = None
    move_count = 0
    while (True):

        print("Current move is for player: ", players[current_player_index].player)

        if (file is not None):
            file.write("PRIOR TO MOVE " + str(move_count) + " ------------\n")
            file.write(to_display_string(board))

        if p.is_cat_game(board):
            if (file is not None):
                file.write("RESULT IS CAT GAME")
            break

        m = players[current_player_index].pick_next_move(board)

        board = p.add_move(m[0], m[1], board)

        p.display_board(board)

        move_count += 1

        if p.is_winner(board, players[current_player_index].player):
            winner = players[current_player_index]
            if (file is not None):
                file.write("FINAL BOARD AFTER MOVE " + str(move_count) + " WINNER IS: " + winner.player + "\n")
                file.write(to_display_string(board))
            break

        # alternate players
        if (current_player_index == 0):
            print("Switching to player 1...")
            current_player_index = 1
        else:
            print("Switching to player 0...")
            current_player_index = 0


    if (winner is None):
        print("CAT GAME")
    else:
        print("WINNER IS PLAYER: ", winner.player)

    return winner
