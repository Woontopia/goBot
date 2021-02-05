from dlgo.gotypes import Point


# def is_point_an_eye(board, point, color):
#     # An Eye is an empty point on the board
#     if board.get(point) is not None:
#         return False
#     for neighbor in point.neighbors():
#         if board.is_on_grid(neighbor):
#             neighbor_color = board.get(neighbor)
#             if neighbor_color != color:
#                 return False
#
#     # Must control 3 out of 4 corners if point in the middle
#     # If on edge, must control all corners
#     friendly_corners = 0
#     off_board_corners = 0
#     corners = [
#         Point(point.row - 1, point.col - 1),
#         Point(point.row - 1, point.col + 1),
#         Point(point.row + 1, point.col - 1),
#         Point(point.row + 1, point.col + 1)
#     ]
#     for corner in corners:
#         if board.is_on_grid(corner):
#             corner_color = board.get(corner)
#             if corner_color == color:
#                 friendly_corners += 1
#         else:
#             off_board_corners += 1
#     if off_board_corners > 0:
#         # Point is on edge or the corner
#         return off_board_corners + friendly_corners == 4
#     # Point is in the middle
#     return friendly_corners >= 3

# tag::eye[]
def is_point_an_eye(board, point, color):
    if board.get(point) is not None:  # <1>
        return False
    for neighbor in point.neighbors():  # <2>
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False

    friendly_corners = 0  # <3>
    off_board_corners = 0
    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4  # <4>
    return friendly_corners >= 3  # <5>
