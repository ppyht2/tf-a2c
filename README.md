![Ms Pacman](imgs/1518718878.gif)
![Breakout](imgs/1518717941.gif)
![Space Invaders](imgs/1518718521.gif)


# Advantage Actor-Critic
Minimal TensorFlow implementation of the Advantage Actor-Critic model for Atari games.

> As an alternative to the asynchronous implementation, researchers found you can write asynchronous, deterministic implementation that waits for each actor to finish its segment of experience before performing an update, averaging over all of the actors. One advantage of this method is that it can more effectively use of GPUs, which perform best with large batch sizes. This algorithm is naturally called A2C, short for advantage actor-critic.

* [Original A3C Paper](https://arxiv.org/abs/1602.01783)
* [A2C Blog Post](https://blog.openai.com/baselines-acktr-a2c/)

The gym environment wrappers used are from [Open AI baseline](https://github.com/openai/baselines)
