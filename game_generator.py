import random

# Get a list of randomly generated gameSize x gameSize sudoku boards
def getSudokuValidBoards(nBoards, gameSize):
    return [shuffleBoard(initBoard(gameSize)) for _ in xrange(nBoards)]

# Initialize a known-valid sudoku board deterministically
def _initBoard(n):
    nums = [e + 1 for e in range(n)]
    init = [nums]
    for i in range(1, n):
        row = nums[-i:] + nums[:n - i]
        init.append(row)
    return init

# Shuffle rows and columns of an n x n list of lists
def _shuffleBoard(board):
    random.shuffle(board)
    boardT = transposeBoard(board)
    random.shuffle(boardT)
    return transposeBoard(boardT)

def _transposeBoard(board):
    return [list(*row) for row in zip(zip(*board))]
