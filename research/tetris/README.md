# Tetris

This is a program that uses the Deep Q algorithm to train Tetris ai. Currently only tested on the mac.

game.py The logic of the game, without UI

model.py Model files

robot.py ai game logic and training logic

play.py Game UI, and control flow

## Design ideas

The overall idea of using deepQ algorithm, later reference to these articles in the algorithm:

Including prioritized sweeping, heuristic reward functions, heuristic reward functions to score reward migration, etc.

http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf

https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

## Instructions

play.py

    Directly to the game

play.py -a

    Use ai game

play.py -A0

    Use ai for interfaceless games, statistical average score, can be used to evaluate ai
    -Ax, x is the number of games, if you enter 0, 10 times

play.py -t0 [-n [-g]] [-m] [-l0] [-u0]

    Training model
    -tx, x is the number of training, enter 0 to 10000 times
    -n, Create a new model training (without this option will automatically load the previously saved model to continue training)
    -g, Used with -n to load a model from the golden directory as the initial value of the new model (you can start training in the middle of an archive)
    -m, Use master training mode for later adjustments. The logic of "random action" and "reward function" in master mode is different
    -lx, The specified learning rate, if you do not specify, then use the exponential declining learning rate
    -ux, The training process shows ui, generally used for debugging. x indicates the UI frame interval (milliseconds)

## My training steps:

play.py -n140000   ——This result is saved as golden by me

play.py -n10000 -m -l0.0001

play.py -n10000 -m -l0.00008

The back of the steps can be sustained, the test results at any time.

My training results, you can achieve an average of 70 points or more, up to a maximum of 340 points.

My trained model is saved here, there are also some historical models and training code here.

https://github.com/yaoGreat/tetris
