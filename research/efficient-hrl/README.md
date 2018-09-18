Code for performing Hierarchical RL based on
"Data-Efficient Hierarchical Reinforcement Learning" by
Ofir Nachum, Shixiang (Shane) Gu, Honglak Lee, and Sergey Levine
(https://arxiv.org/abs/1805.08296).


This library currently includes three of the environments used:
Ant Maze, Ant Push, and Ant Fall.

The training code is planned to be open-sourced at a later time.


Requirements:
* TensorFlow (see http://www.tensorflow.org for how to install/upgrade)
* OpenAI Gym (see http://gym.openai.com/docs, be sure to install MuJoCo as well)
* NumPy (see http://www.numpy.org/)


Quick Start:

Run a random policy on AntMaze (or AntPush, AntFall):

```
python environments/__init__.py --env=AntMaze
```


Maintained by Ofir Nachum (ofirnachum).
