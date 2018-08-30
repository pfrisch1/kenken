from collections import defaultdict
from Game import *
from random import randrange

gameSize = 4

#def makeBoard(gameSize):
counts = defaultdict(lambda: defaultdict(int))
games = getGamesFromXML('data/gamedata%d.xml' % gameSize, gameSize)
nClues = 0
for game in games:
    for clue in game.clues:
        counts[clue.operation][clue.length] += 1
        nClues += 1

clueCounts = []
countMap = {}
i = 0
for op in counts:
    for length in counts[op]:
        clueCounts.append(counts[op][length])
        countMap['%s%d' % (op,length)] = i
        i += 1

clueNum = randrange(nClues)
    
