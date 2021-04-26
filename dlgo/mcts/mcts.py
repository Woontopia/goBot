import random
import math

from dlgo import agent
from dlgo.gotypes import Player

# Rollout = an instance of a random simulated game
class MCTSNode(object):
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0
        }
        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    # If there is any legal moves that we havent yet visited
    def can_add_child(self):
        return len(self.unvisited_moves > 0)

    # Returns if the game is over at this node
    def is_terminal(self):
        return self.game_state.is_over()

    # returns the fraction of rollouts that were won by a given player
    def winning_frac(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)


def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + temperature * exploration


class MCTSAgent(agent.Agent):
    def __init__(self, num_rounds, temperature):
        agent.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state):
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            # Adds a new child node to the tree
            if node.can_add_child():
                node = node.add_random_child()

            # Simulates a random game from that node
            winner = self.simulate_random_game(node.game_state)

            # Sends the score back up the tree
            while node is not None:
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_pct(game_state.next_player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        return best_move

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(total_rollouts, child.num_rollouts, child.winning_pct(node.game_state.next_player), self.temperature)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    @staticmethod
    def simulate_random_game(game):
        bots = {
            Player.black: agent.naive.RandomBot(),
            Player.white: agent.naive.RandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()

