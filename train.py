from game_generator import makeBoards
import argparse
import matplotlib.pyplot as plt
import numpy as np
from models import SimpleCNN
import torch
import torch.nn.functional as F

from game_generator import getSudokuValidBoards

# run train.py --bsz 40 --epochs 100000 --lr 0.001 --gsz 4 --pr 100 --n 50

def get_args():
    parser = argparse.ArgumentParser(description='train something to play kenken')
    parser.add_argument('--bsz', type=int,
                        help='batch size')
    parser.add_argument('--epochs', type=int,
                        help='number of gradient steps')
    parser.add_argument('--lr', type=float,
                        help='learning rate')
    parser.add_argument('--gsz', type=int,
                        help='game size')
    parser.add_argument('--pr', type=int,
                        help='print rate')
    parser.add_argument('--n', type=int,
                        help='number of unseen examples to test on')

    args = parser.parse_args()
    print args
    return args

def get_model(args):
    return SimpleCNN(args.gsz)

def get_xy(bsz, gsz):
    # boards = makeBoards(bsz, gsz)

    # x = torch.LongTensor([board.answers for board in  boards]) - 1
    x = torch.LongTensor(getSudokuValidBoards(bsz, gsz)) - 1
    for i in range(bsz/2):
        change_ix = [np.random.randint(gsz), np.random.randint(gsz)]
        current = x[i,change_ix[0], change_ix[1]]
        newone = (current + np.random.randint(1,gsz)) % gsz
        x[i,change_ix[0],change_ix[1]] = newone
    ygt = torch.Tensor([[0] if i<bsz/2 else [1] for i in range(bsz)])    

    return x, ygt

def get_opt(args):
    return torch.optim.Adam(model.parameters(), lr = args.lr)

def eval_model(n, model, gsz):
    x = torch.LongTensor(getSudokuValidBoards(n, gsz)) - 1
    for i in range(n/2):
        for r in range(gsz):
            for c in range(gsz):
                if np.random.rand() < (1 / float(gsz)):
                    current = x[i,r,c]
                    newone = (current + np.random.randint(1,gsz)) % gsz
                    x[i,r,c] = newone

    ygt = torch.Tensor([[0] if i<n/2 else [1] for i in range(n)]) 

    y = model(x)
    return get_acc(y, ygt)

def get_acc(ypred, ygt):
    ypred = torch.Tensor([0 if pred<0 else 1 for pred in ypred.squeeze()])
    return (ypred == ygt.squeeze()).sum().item() / float(len(ygt))

args = get_args()
model = get_model(args)
optimizer = get_opt(args)

for batch_ix in range(args.epochs):
    x, ygt = get_xy(args.bsz, args.gsz)

    optimizer.zero_grad()
    model.train()
    y = model(x)
    l = F.binary_cross_entropy_with_logits(y,ygt)
    l.backward()
    optimizer.step()

    if (batch_ix +1)%args.pr ==0:
        print "batch:", batch_ix, "loss:", l.item(), "acc:", get_acc(y, ygt)
        print "EVALUDATION:", eval_model(args.n, model, args.gsz)




    # fig = plt.figure()
    # ax = plt.gca()
    # for i in xrange(len(board.answers)):
    #     for j in xrange(len(board.answers[0])):
    #         c = board.answers[i][j]
    #         ax.text(j+0.5, len(board.answers[0]) - i-0.5, str(c), va='center', ha='center')
    # for cell in board.clueMap:
    #     r, c = cell
    #     clue = board.clueMap[cell]
    #     ax.text(c, len(board.answers[0]) - r, str(clue.value) + str(clue.operation), va='center', ha='center')
    # plt.xlim((0, len(board.answers[0])))
    # plt.ylim((0, len(board.answers)+1))
    # plt.axis('off')
    # plt.show(block=False)
    # plt.pause(.1)
    # raw_input()
    # plt.close(fig)