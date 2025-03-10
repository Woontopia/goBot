# Board functionalitys

import copy
from dlgo.gotypes import Player, Point

# contains all the actions a player can do in a turn (place stone(play), pass or resign
class Move():
    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign

    # Places a stone on the board
    @classmethod
    def play(cls, point):
        return Move(point=point)

    # Passes
    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)

    # Resigns
    @classmethod
    def resign(cls):
        return Move(is_resign=True)


# GO Strings are a chain of connected stones of the same color p.33
class GoString():
    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)

    def remove_liberty(self, point):
        self.liberties.remove(point)

    def add_liberty(self, point):
        self.liberties.add(point)

    # Returns a new GoString with all the stones in both strings
    # Called when a player places a stone to connect 2 of his string
    def merged_with(self, go_string):
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones
        )

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties


# # Board initialized as empty grid with specified dimensions
# class Board():
#     def __init__(self, num_rows, num_cols):
#         self.num_rows = num_rows
#         self.num_cols = num_cols
#         self._grid = {}
#
#     def place_stone(self, player, point):
#         assert self.is_on_grid(point)
#         assert self._grid.get(point) is None
#         adjacent_same_color = []
#         adjacent_opposite_color = []
#         liberties = []
#
#         # First, examine direct neighbors of point
#         for neighbor in point.neighbors():
#             if not self.is_on_grid(neighbor):
#                 continue
#             neighbor_string = self._grid.get(neighbor)
#             if neighbor_string is None:
#                 liberties.append(neighbor)
#             elif neighbor_string.color == player:
#                 if neighbor_string not in adjacent_same_color:
#                     adjacent_same_color.append(neighbor_string)
#             else:
#                 if neighbor_string not in adjacent_opposite_color:
#                     adjacent_opposite_color.append(neighbor_string)
#         new_string = GoString(player, [point], liberties)
#
#         # Merge any adjacent goStrings of the same color
#         for same_color_string in adjacent_same_color:
#             new_string = new_string.merged_with(same_color_string)
#         for new_string_point in new_string.stones:
#             self._grid[new_string_point] = new_string
#
#         # Reduce liberties of adjacent string of the opposite color
#         for other_color_string in adjacent_opposite_color:
#             other_color_string.remove_liberty(point)
#
#         # Remove any opposite color goStrings that not have zero liberties
#         for other_color_string in adjacent_opposite_color:
#             if other_color_string.num_liberties == 0:
#                 self._remove_string(other_color_string)
#
#     def is_on_grid(self, point):
#         return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols
#
#     # Returns a Player if stone is on that point, or else None
#     def get(self, point):
#         string = self._grid.get(point)
#         if string is None:
#             return None
#         return string.color
#
#     # Returns the entire string of stones at a point: GoString if a stone is on that point
#     def get_go_string(self, point):
#         string = self._grid.get(point)
#         if string is None:
#             return None
#         return string
#
#     def _remove_string(self, string):
#         for point in string.stones:
#             # Removing a string can create liberties for other strings
#             for neighbor in point.neighbors():
#                 neighbor_string = self._grid.get(neighbor)
#                 if neighbor_string is None:
#                     continue
#                 if neighbor_string is not string:
#                     neighbor_string.add_liberty(point)
#             self._grid[point] = None

class Board():  # <1>
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid = {}

# <1> A board is initialized as empty grid with the specified number of rows and columns.
# end::board_init[]

# tag::board_place_0[]
    def place_stone(self, player, point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        adjacent_same_color = []
        adjacent_opposite_color = []
        liberties = []
        for neighbor in point.neighbors():  # <1>
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)
        new_string = GoString(player, [point], liberties)
# <1> First, we examine direct neighbors of this point.
# end::board_place_0[]
# tag::board_place_1[]
        for same_color_string in adjacent_same_color:  # <1>
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string
        for other_color_string in adjacent_opposite_color:  # <2>
            other_color_string.remove_liberty(point)
        for other_color_string in adjacent_opposite_color:  # <3>
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)
# <1> Merge any adjacent strings of the same color.
# <2> Reduce liberties of any adjacent strings of the opposite color.
# <3> If any opposite color strings now have zero liberties, remove them.
# end::board_place_1[]

# tag::board_remove[]
    def _remove_string(self, string):
        for point in string.stones:
            for neighbor in point.neighbors():  # <1>
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    neighbor_string.add_liberty(point)
            del(self._grid[point])
# <1> Removing a string can create liberties for other strings.
# end::board_remove[]

# tag::board_utils[]
    def is_on_grid(self, point):
        return 1 <= point.row <= self.num_rows and \
            1 <= point.col <= self.num_cols

    def get(self, point):  # <1>
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point):  # <2>
        string = self._grid.get(point)
        if string is None:
            return None
        return string
# <1> Returns the content of a point on the board:  a Player if there is a stone on that point or else None.
# <2> Returns the entire string of stones at a point: a GoString if there is a stone on that point or else None.
# end::board_utils[]

    def __eq__(self, other):
        return isinstance(other, Board) and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._grid == other._grid


# # Class to know the current state of the game
# class GameState():
#     def __init__(self, board, next_player, previous, move):
#         self.board = board
#         self.next_player = next_player
#         self.previous_state = previous
#         self.last_move = move
#
#     # Returns the new GameState after applying the move
#     def apply_move(self, move):
#         if move.is_play:
#             next_board = copy.deepcopy(self.board)
#             next_board.place_stone(self.next_player, move.point)
#         else:
#             next_board = self.board
#         return GameState(next_board, self.next_player.other, self, move)
#
#     # Initiates a new Game
#     @classmethod
#     def new_game(cls, board_size):
#         if isinstance(board_size, int):
#             board_size = (board_size, board_size)
#         board = Board(*board_size)
#         return GameState(board, Player.black, None, None)
#
#     # Checks if the game is over
#     def is_over(self):
#         if self.last_move is None:
#             return False
#         # Checks if the player resigns
#         if self.last_move.is_resign:
#             return True
#         second_last_move = self.previous_state.last_move
#         if second_last_move is None:
#             return False
#         # Game is over if the 2 players pass one after the other
#         return self.last_move.is_pass and second_last_move.is_pass
#
#     def is_move_self_capture(self, player, move):
#         if not move.is_play:
#             return False
#         next_board = copy.deepcopy(self.board)
#         next_board.place_stone(player, move.point)
#         new_string = next_board.get_go_string(move.point)
#         return new_string.num_liberties == 0
#
#     # For the Ko Rule
#     @property
#     def situation(self):
#         return (self.next_player, self.board)
#
#     def does_move_violate_ko(self, player, move):
#         if not move.is_play:
#             return False
#         next_board = copy.deepcopy(self.board)
#         next_board.place_stone(player, move.point)
#         next_situation = (player.other, next_board)
#         past_state = self.previous_state
#         while past_state is not None:
#             if past_state.situation == next_situation:
#                 return True
#             past_state = past_state.previous_state
#         return False
#
#     def is_valid_move(self, move):
#         if self.is_over():
#             return False
#         if move.is_pass or move.is_resign:
#             return True
#         return (
#             self.board.get(move.point) is None and
#             not self.is_move_self_capture(self.next_player, move) and
#             not self.does_move_violate_ko(self.next_player, move)
#         )

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        self.last_move = move

    def apply_move(self, move):  # <1>
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)
# <1> Return the new GameState after applying the move.
# end::game_state[]

# tag::self_capture[]
    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0
# end::self_capture[]

# tag::is_ko[]
    @property
    def situation(self):
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board)
        past_state = self.previous_state
        while past_state is not None:
            if past_state.situation == next_situation:
                return True
            past_state = past_state.previous_state
        return False
# end::is_ko[]

# tag::is_valid_move[]
    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        return (
            self.board.get(move.point) is None and
            not self.is_move_self_capture(self.next_player, move) and
            not self.does_move_violate_ko(self.next_player, move))
# end::is_valid_move[]

# tag::is_over[]
    def is_over(self):
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass
# end::is_over[]














