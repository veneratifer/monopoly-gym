# Monopoly Gym
Monopoly simulator for reinforcement learning following approach described in https://arxiv.org/pdf/2103.00683.pdf  
A hybrid agent is employed, where the Q-Network is mostly utilized for actions, complemented by a rule-based policy (mirroring that of opponent agents) for a select few actions.

## Installation
```shell
git clone https://github.com/veneratifer/monopoly-gym
cd monopoly-gym
pip install -r requirements.txt
```

## Usage
Call repo directory `$ python monopoly-gym --num_episodes=<N>` to start training session with N game episodes.  
For default trained network will be saved to the file named "model.pt", this can be changed with option `--out_path=<OUTPUT_PATH>`. To evaluate model pass file with serialized state_dict `$ python monopoly-gym --eval --model_path=<MODEL_PATH>`, you can add flag `--vis` to run in a visual mode.

## Configuration
For training configuration look at TrainConfig class from model/trainer.py. I set hyperparameters similar to these proposed as optimal for DDQN in a paper.

## Acknowledgements
The authors of the paper provided their own [simulator](https://github.com/mayankkejriwal/GNOME-p3).    
I found a lot of inspiration there, but indeed I wrote everything from the beginning. I aimed to keep the implementation simple and concise.

