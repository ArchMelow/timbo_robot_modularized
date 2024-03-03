# Reinforcement Learning Pipeline using dm_control(MuJoCo based) and PPO algorithm (implemented in Tensorflow)

Ultimate objective of this project is to use RL-trained actor weights on the ESP32, to make a reindeer-shaped toy walk properly.

1. Install dm_control using below command. You might also have to install MuJoCo.
```bash
pip install dm_control
```

2. Replace the files in your conda environment's dm_control/suite folder with the files in this repository's ../edit folder. In my case, it was:
```bash
C:\Users\{USERNAME}\anaconda3\envs\{CONDA_ENV_NAME}\Lib\site-packages\dm_control\suite
```

testing on : Tensorflow 2.11.0, dm_control 1.0.16
