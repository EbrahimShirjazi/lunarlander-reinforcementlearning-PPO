# LunarLander Reinforcement Learning — PPO

A Proximal Policy Optimization (PPO) agent trained with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) to solve the LunarLander environment from Gymnasium. Built as part of Hugging Face's [Deep RL Course](https://huggingface.co/learn/deep-rl-course) (Unit 1).

## Result

**Mean reward: 287.04 ± 15.50** (over 20 evaluation episodes, deterministic policy)

LunarLander is considered "solved" at a mean reward of 200+.

## Model

- **Algorithm**: PPO (MlpPolicy)
- **Environment**: LunarLander-v2
- **Training steps**: 2,000,000
- **Hyperparameters**:
  - `n_steps`: 2048
  - `batch_size`: 128
  - `n_epochs`: 10
  - `gamma`: 0.999
  - `gae_lambda`: 0.98
  - `ent_coef`: 0.01
  - `learning_rate`: 3e-4
  - `clip_range`: 0.2
  - 16 parallel training environments

Full trained model and model card: **[huggingface.co/Ebishj/ppo-LunarLander-v2-clean](https://huggingface.co/Ebishj/ppo-LunarLander-v2-clean)**

## Replay

See the agent in action on the [Hugging Face model page](https://huggingface.co/Ebishj/ppo-LunarLander-v2-clean/blob/main/replay.mp4).

## Notebook

The full training pipeline — environment setup, training, evaluation, and Hugging Face Hub deployment — is in [`ppo_lunarlander.ipynb`](./ppo_lunarlander.ipynb).

## What this covers

- Setting up Gymnasium + Box2D environments in a Colab environment
- Training with vectorized (parallel) environments for faster, more stable learning
- PPO hyperparameter tuning
- Evaluating a trained policy
- Publishing a model, model card, and replay video to the Hugging Face Hub
