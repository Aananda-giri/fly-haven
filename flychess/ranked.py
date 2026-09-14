"""V5.5's engine-free ranked lookahead, encoded through fly sensory neurons.

Two plies of material minimax are an intentionally limited external helper.
Rank-one and fly-input interventions must be measured to attribute improvement.
"""
from functools import lru_cache
import chess
import numpy as np

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
RANKED_K = 4
RANKED_FEATURES = 788 + RANKED_K * (780 + 128 + 2)


def material_score(board, us):
    if board.is_checkmate():
        return -10000 if board.turn == us else 10000
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
        return 0
    return sum(value * (len(board.pieces(piece, us)) - len(board.pieces(piece, not us)))
               for piece, value in PIECE_VALUES.items())


@lru_cache(maxsize=8192)
def ranked_lookahead(fen):
    """Every legal root move, sorted by its worst reply's material outcome.

    No network/Stockfish, no claimable-draw heuristic, fixed two-ply depth.
    FEN caching omits prior repetition history; the fly separately receives it.
    """
    board = chess.Board(fen)
    us = board.turn
    ranked = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_game_over(claim_draw=False):
            score = material_score(board, us)
        else:
            scores = []
            for reply in list(board.legal_moves):
                board.push(reply)
                scores.append(material_score(board, us))
                board.pop()
            score = min(scores)
        board.pop()
        ranked.append((move.uci(), score))
    return tuple(sorted(ranked, key=lambda item: (-item[1], item[0])))


def encode_ranked_board(board, encode_history_board, encode_base_board, intervention=None):
    result = np.zeros(RANKED_FEATURES, np.float32)
    if intervention == "zero_all":
        return result
    result[:788] = encode_history_board(board)
    if intervention == "no_candidates":
        return result
    ranked = list(ranked_lookahead(board.fen(en_passant="fen"))[:RANKED_K])
    if intervention == "reverse_candidates":
        ranked.reverse()
    us = board.turn
    for rank, (uci, score) in enumerate(ranked):
        move = chess.Move.from_uci(uci)
        child = board.copy(stack=False)
        child.push(move)
        # Candidate boards are encoded in the ROOT mover's coordinate frame.
        child.turn = us
        offset = 788 + rank * 910
        result[offset:offset+780] = encode_base_board(child)
        frm = move.from_square if us else chess.square_mirror(move.from_square)
        to = move.to_square if us else chess.square_mirror(move.to_square)
        result[offset+780+frm] = 1
        result[offset+844+to] = 1
        result[offset+908] = np.clip(score / 1000, -1, 1)
        result[offset+909] = 1
    return result
