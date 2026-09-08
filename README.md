# brawlgym-ppo
A vectorized implementation of PPO for use with [brawlgym](https://github.com/brawlgym/brawlgym), adapted from
[rlgym-ppo](https://github.com/AechPro/rlgym-ppo).

## INSTALLATION
1. Install [brawlgym](https://github.com/brawlgym/brawlgym).
2. If you would like to use a GPU install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
3. Install this project via `pip install git+https://github.com/brawlgym/brawlgym-ppo`

## USAGE
Import the learner with `from brawlgym_ppo import Learner`, pass it a function that takes a game port and returns a
brawlgym environment, and run the learning algorithm. The learner patches the game, boots one instance per process
and hands each process its port:
```
from brawlgym_ppo import Learner

def my_brawlgym_function(port):
    import brawlgym
    return brawlgym.make(port=port)

learner = Learner(my_brawlgym_function)
learner.learn()
```
See `example.py` for a function that configures the environment's components.
