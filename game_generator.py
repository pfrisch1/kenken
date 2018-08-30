import random

# Make an n x n board adhering to sudoku rules
def sudokuValidBoard(n):
    nums = [i + 1 for i in range(n)]
    random.shuffle(nums)

    board = []
    while len(board) < n:
        while not _newRowIsValid(board, nums):
            random.shuffle(nums)
        board.append(nums[:])

    return board

def _newRowIsValid(partialBoard, nums):
    valid = True
    for row in partialBoard:
        for i, num in enumerate(nums):
            if row[i] == num:
                valid = False
                break
    return valid
