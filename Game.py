from operations import *
import xml.etree.ElementTree as ET
from pprint import pprint


class Clue:
    def __init__(self, value, operation, cells):
        self.text = '%d%s' % (value, operation)
        self.value = value
        self.operation = operation
        self.cells = cells
        self.length = len(cells)

    def check(guess):
        # guess is a list of numbers
        if operation is DIV:
            return (guess[0] / float(guess[1]) == self.value) \
                or (guess[1] / float(guess[0]) == self.value)
        if operation is SUB:
            return (guess[0] - guess[1] == self.value) \
                or (guess[1] - guess[0] == self.value)
        if operation is MUL:
            return reduce(lambda x, y: x * y, guess) == self.value
        if operation is ADD:
            return sum(guess) == self.value
        if operation is EQL:
            return guess[0] == self.value

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text



class Game:
    def __init__(self, clues, answers, gameSize):
        self.clues = clues
        self.gameSize = gameSize
        self.answers = answers
        self.guesses = [[0 for i in range(gameSize)] for j in range(gameSize)]
        self.clueMap = {}

        for clue in clues:
            for cell in clue.cells:
                self.clueMap[cell] = clue
        assert len(self.clueMap) == gameSize ** 2

    def visualizeCluesId(self):
        textArray = [[0 for i in range(self.gameSize)]
                     for j in range(self.gameSize)]
        currId = 0
        clueIDs = {}
        for cell in self.clueMap:
            (r, c) = cell
            clue = self.clueMap[cell]
            if clue not in clueIDs:
                currId += 1
                clueIDs[clue] = currId
            textArray[r][c] = clueIDs[clue]

        pprint(clueIDs)
        for row in textArray:
            print row

    def visualizeClues(self):
        textArray = [[0 for i in range(self.gameSize)]
                     for j in range(self.gameSize)]
        for cell in self.clueMap:
            (r, c) = cell
            textArray[r][c] = self.clueMap[cell]

        pprint(textArray)

    def visualizeAnswers(self):
        for row in self.answers:
            print row


class Cell:
    # 0, 1, 2, 3 -> N, E, S, W
    def __init__(self, row, col, gameSize):
        self.row = row
        self.col = col
        walls = [False] * 4

        if row == 0:
            walls[0] = True
        if row == gameSize - 1:
            walls[2] = True
        if col == 0:
            walls[3] = True
        if col == gameSize - 1:
            walls[1] = True

        self.walls = walls

    def setWall(self, direc):
        self.walls[direc] = True

    def hasWall(self, direc):
        return self.walls[direc]

# Note: because the files are small (< 300 games)
# it is almost instantaneous to just get all of the games


def getGamesFromXML(xmlPath, gameSize):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    games = []
    for game in root:
        maze = []
        for rownum in range(gameSize):
            row = []
            for colnum in range(gameSize):
                row.append(Cell(rownum, colnum, gameSize))
            maze.append(row)

        verts = game.find('vertical').text.split(',')
        for i in range(gameSize ** 2 - gameSize, -1, -gameSize):
            verts.pop(i)
        for (i, v) in enumerate(verts):
            v = v.strip('"')
            wall = int(v)
            row = i / (gameSize - 1)
            col = i % (gameSize - 1)
            if wall == 1:
                maze[row][col].setWall(1)
                maze[row][col + 1].setWall(3)

        horz = game.find('horizontal').text.split(',')
        horz = horz[:-gameSize]
        for (i, h) in enumerate(horz):
            h = h.strip('"')
            wall = int(h)
            row = i / gameSize
            col = i % gameSize
            if wall == 1:
                maze[row][col].setWall(2)
                maze[row + 1][col].setWall(0)

        clueTexts = game.find('clue').text.split(',')
        clueTexts.pop()
        clues = addClues(maze, clueTexts)

        answers = [[0 for i in range(gameSize)] for j in range(gameSize)]

        answersText = game.find('answer').text.split(',')
        for (i, a) in enumerate(answersText):
            a = a.strip('"')
            answer = int(a)
            row = i / gameSize
            col = i % gameSize
            answers[row][col] = answer

        gameObject = Game(clues, answers, gameSize)
        games.append(gameObject)
    return games


def addClues(maze, clueTexts):
    gameSize = len(maze)
    clues = []
    for (i, c) in enumerate(clueTexts):
        row = i / gameSize
        col = i % gameSize
        c = c.strip('"')
        if c == '0':
            continue
        if len(c) == 1:
            val = int(c)
            op = '='
        else:
            val = int(c[:-1])
            op = c[-1]
        tiles = []
        toVisit = [(row, col)]
        visited = set()
        visited.add((row, col))

        while toVisit:
            curr = toVisit.pop()
            r = curr[0]
            c = curr[1]
            tiles.append(curr)
            for direc in range(4):
                # N,E,S,W
                if not maze[r][c].hasWall(direc):
                    if direc == 0:
                        neighbor = (r - 1, c)
                    elif direc == 1:
                        neighbor = (r, c + 1)
                    elif direc == 2:
                        neighbor = (r + 1, c)
                    elif direc == 3:
                        neighbor = (r, c - 1)
                    if neighbor not in visited:
                        toVisit.append(neighbor)
                        visited.add(neighbor)
        clues.append(Clue(val, op, tiles))
    return clues
