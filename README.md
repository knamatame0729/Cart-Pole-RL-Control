# Cart Pole Swing-Up Reinforcement Learning
**Originally I used ```conda``` to manage the virtual environment with python3.11. However, conda was found to have compatibility issues with ROS2. I switched to using ```venv``` to create a Python 3.10 environment.  
See [ros2_cartpole branch](https://github.com/knamatame0729/Cart-Pole-RL-Control/tree/ros2_cartpole) for the setup**
## Overview
In this final project, reinforcement learning is used to implement control for **swinging a pole on a cart from a downward position to a vertically upright position** and then maintaining balance while keeping the cart at the center of the track.  

Swing-up control requires more advanced control than simple balance maintenance, and since it is a system that cannot be controlled with linear control, reinforcement learning is effective. The Proximal Policy Optimization (PPO) algorithm, which can handle continuous action spaces, is adopted as the policy learning algorithm.  

The implementation utilizes the **RSL RL** library and the **Genesis** environment.

## Testing Environment
- AMD Ryzen 7 5700X
- RTX 3060 Ti
- CUDA 11.8
- RAM 32GB
- Ubuntu 22.04

## Demo Video
![Demo](media/cart_pole_rl.gif)  

## Installation and Usage

```bash
conda create -n cart-pole-rl python3.11
conda activate cart-pole-rl
```
Install pytorch with matching CUDA version (CUDA 11.8 is utilized in this repo)
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install Genesis
```bash
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```

Clone Genesis reopsitory and install locally
```bash
cd
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
```
Install rsl_rl library locally
```bash
cd
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```
Clone this repository
```bash
cd
git clone https://github.com/knamatame0729/Cart-Pole-RL-Control.git cart_pole_rl_control
pip install tensorboard
```

Run train script
```bash
cd cart_pole_rl_control
python3 cart_pole_train.py
```
Run this on the other terminal and follow discription to see train detail
```bash
cd cart_pole_rl_control
tensorboard --logdir logs
```
After complete training, run this to watch the training result
```bash
python3 cart_pole_eval.py
```
## Reward Functions

1. **Reward for Pole Upright**  
Encourage the pole to swing from the downward position to the upright position (Swing-Up)

<div align="center">

![alt text](media/Screenshot%20from%202025-07-18%2021-46-52.png)  

![alt text](media/Screenshot%20from%202025-07-18%2021-47-16.png)  

</div>
  


2. **Reward for Upright Stability**  
To remain stable near the upright position

<div align="center">

![alt text](media/Screenshot%20from%202025-07-18%2021-47-37.png)

</div>

3. **Penalty for Action Rate**  
To ensure smooth and stable control, reducing the oscillations in the pole and cart during the swing-up phase  

<div align="center">

![alt text](media/Screenshot%20from%202025-07-18%2021-48-02.png)

</div>

4. **Penalty for Cart Position Deviation**  
Add a penalty for the cart moving away from x = 0 to keep the cart centered

<div align="center">

![alt text](media/Screenshot%20from%202025-07-18%2021-48-24.png)

</div>


## Evaluation Metrics
#### Cart Position (m)
![](media/cart_pos.png)
#### Cart Velocity (m/s)
![](media/cart_vel.png)  

#### Pole Angle (rad)
![](media/pole_angle.png)  

#### Pole Velocity (rad/s)
![](media/pole_vel.png)  

#### Control Force (N)
![](media/actions.png)  

## References
- Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter.  
**"Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning."**  
Proceedings of the 5th Conference on Robot Learning (CoRL 2022), PMLR 164:91–100, 2022.  
[Link to paper](https://proceedings.mlr.press/v164/rudin22a.html)  

- Genesis Authors. **Genesis: A Generative and Universal Physics Engine for Robotics and Beyond**, December 2024.  
[https://github.com/Genesis-Embodied-AI/Genesis](https://github.com/Genesis-Embodied-AI/Genesis)

## License
MIT Liense
