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
Call repo directory `$ python monopoly-gym --num-episodes=<N>` to start training session with N game episodes.  
For default trained network weights will be saved to the file "model.h5", to change it use `--out-path=<OUTPUT_PATH>`. To evaluate model pass file with model weights `$ python monopoly-gym --eval --model-path=<MODEL_PATH>`, you can add flag `--vis` to run in a visual mode.

## Acknowledgements
The authors of the paper provided their own [simulator](https://github.com/mayankkejriwal/GNOME-p3).    
I found a lot of inspiration there, but indeed I wrote everything from the beginning. I aimed to keep the implementation simple and concise.
