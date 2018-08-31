from collections import defaultdict
from Game import *
import copy
import random
import numpy as np
def getBoard(answer, counts, nClues, cluesPerGame, gameSize):
    available = []
    for r in range(gameSize):
        for c in range(gameSize):
            available.append((r,c))

    clueList = []

    divFrac = counts['/'][2] / float(nClues)
    divClues = np.random.binomial(cluesPerGame, divFrac)
    spots = findDivisionSpots(answer, gameSize)
    random.shuffle(spots)
    while divClues > 0 and spots:
        spot = spots.pop()
        c1 = spot[0]
        c2 = spot[1]
        if c1 in available and c2 in available:
            divClues -= 1
            available.remove(c1)
            available.remove(c2)
            val = obtainValue(spot, '/', answer)
            clueList.append(Clue(val,'/',spot))


    while available:
        (op, length) = randomClue(counts, nClues, ['/'])
        cells = getClueCells(available, length, gameSize)
        if len(cells) == 1:
            op = EQL
        val = obtainValue(cells, op, answer)
        clueList.append(Clue(val,op,cells))

    return Game(clueList, answer, gameSize)

def makeBoards(nBoards, gameSize):
    counts = defaultdict(lambda: defaultdict(int))
    games = getGamesFromXML('data/gamedata%d.xml' % gameSize, gameSize)
    nClues = 0
    for game in games:
        for clue in game.clues:
            counts[clue.operation][clue.length] += 1
            nClues += 1

    boards = []

    answers = getSudokuValidBoards(nBoards, gameSize)
    cluesPerGame = nClues / len(games)

    for answer in answers:
        boards.append(getBoard(answer, counts, nClues, cluesPerGame, gameSize))

    return boards

def getClueCells(available, desiredLength, gameSize):
    cells = [available.pop(0)]
    currentCell = cells[-1]

    while len(cells) < desiredLength:
        neighbors = getAvailableNeighbors(currentCell, available, gameSize)
        if not neighbors:
            break
        currentCell = random.choice(neighbors)
        cells.append(currentCell)
        available.remove(currentCell)

    return cells

# Initialize a known-valid sudoku board deterministically
def initBoard(n):
    nums = [e + 1 for e in range(n)]
    init = [nums]
    for i in range(1, n):
        row = nums[-i:] + nums[:n - i]
        init.append(row)
    return init

# Get a list of randomly generated gameSize x gameSize sudoku boards
def getSudokuValidBoards(nBoards, gameSize):
    initialBoard = initBoard(gameSize)
    return [shuffleBoard(initialBoard) for _ in xrange(nBoards)]

# Shuffle rows and columns of an n x n list of lists
def shuffleBoard(board):
    random.shuffle(board)
    boardT = transposeBoard(board)
    random.shuffle(boardT)
    return transposeBoard(boardT)

def transposeBoard(board):
    return [list(*row) for row in zip(zip(*board))]

def obtainValue(cells, operation, answer):
    # cells: list of tuples
    # operation: string
    # answer: matrix of numbers
    if operation is EQL:
        (r,c) = cells[0]
        val = answer[r][c]
    elif operation is ADD:
        val = 0
        for cell in cells:
            (r,c) = cell
            val += answer[r][c]
    elif operation is MUL:
        val = 1
        for cell in cells:
            (r,c) = cell
            val *= answer[r][c]
    elif operation is SUB:
        (r1, c1) = cells[0]
        (r2, c2) = cells[1]
        val = answer[r1][c1] - answer[r2][c2]
        if val < 0:
            val = -val
    elif operation is DIV:
        (r1, c1) = cells[0]
        (r2, c2) = cells[1]
        val = answer[r1][c1] / float(answer[r2][c2])
        if val < 1:
            val = 1 / val
    return val


def getAvailableNeighbors(cell, available, gameSize):
    # cell: tuple
    # available: set of cells
    # gameSize: width of board
    neighbors = []
    (r,c) = cell
    if r > 0:
        if (r - 1, c) in available:
            neighbors.append((r - 1, c))
    if r < gameSize - 1:
        if (r + 1, c) in available:
            neighbors.append((r + 1, c))
    if c > 0:
        if (r, c - 1) in available:
            neighbors.append((r, c - 1))
    if c < gameSize - 1:
        if (r, c + 1) in available:
            neighbors.append((r, c + 1))

    return neighbors

def findDivisionSpots(answer, gameSize):
    # answer: matrix of numbers
    # gamesize: width of board
    spots = []
    for r in range(gameSize):
        for c in range(gameSize):
            val1 = answer[r][c]
            if r < gameSize - 1:
                val2 = answer[r+1][c]
                if val1 % val2 is 0 or val2 % val1 is 0:
                    spots.append([(r,c),(r+1,c)])
            if c < gameSize - 1:
                val2 = answer[r][c+1]
                if val1 % val2 is 0 or val2 % val1 is 0:
                    spots.append([(r,c),(r,c+1)])
    return spots


def randomClue(inputCounts,nClues,exclude=[]):
    # inputCounts is a doubledict [operation][length]
    counts = copy.deepcopy(inputCounts)
    for op in exclude:
        for length in counts[op]:
            nClues -= counts[op][length]
        counts[op] = defaultdict(int) # reset to zero
    clueNum = random.randrange(nClues)
    for op in counts:
        for length in counts[op]:
            if clueNum < counts[op][length]:
                return (op,length)
            clueNum -= counts[op][length]
