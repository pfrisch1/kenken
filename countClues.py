from collections import defaultdict
from Game import *
from random import randrange
import copy

gameSize = 4

#def makeBoard(gameSize):
counts = defaultdict(lambda: defaultdict(int))
games = getGamesFromXML('data/gamedata%d.xml' % gameSize, gameSize)
nClues = 0
for game in games:
    for clue in game.clues:
        counts[clue.operation][clue.length] += 1
        nClues += 1

available = set()
for r in range(gameSize):
    for c in range(gameSize):
        available.add((r,c))

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
    elif: operation is DIV:
        (r1, c1) = cells[0]
        (r2, c2) = cells[1]
        val = answer[r1][c1] / float(answer[r2][c2])
        if val < 1:
            val = 1 / val
    return val


def getAvailableNeighbors(cell,available, gameSize):
    # cell: tuple
    # available: set of cells
    # gameSize: width of board
    neighbors = []
    (r,c) = cell
    if r > 0:
        if (r-1, c) in available:
            neighbors.append((r - 1, c))
    if r < gameSize - 1:
        if (r + 1, c) in available:
            neighbors.append((r + 1, c))
    if c > 0:
        if (r, c - 1) in available:
            neighbors.append((r - 1), c)
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
            val1 = float(answer[r][c])
            if r < gameSize - 1:
                val2 = float(answer[r+1][c])
                if (val1 / val2 % 0) or (val2 / val1 % 0):
                    spots.append([(r,c),(r+1,c)])
            if c < gameSize - 1:
                val2 = float(answer[r][c+1])
                if (val1 / val2 % 0) or (val2 / val1 % 0):
                    spots.append([(r,c),(r,c+1)])
    return spots


def randomClue(inputCounts,nClues,exclude=[]):
    # inputCounts is a doubledict [operation][length]
    counts = copy.deepcopy(inputCounts)
    for op in exclude:
        for length in counts[op]:
            nClues -= counts[op][length]
        counts[op] = defaultdict(int) # reset to zero
    clueNum = randrange(nClues)
    for op in counts:
        for length in counts[op]:
            if clueNum < counts[op][length]:
                return '%s%d' % (op,length)
            clueNum -= counts[op][length]

