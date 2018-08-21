# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python 3
 * Numpy
 * TensorFlow
 * MuJoCo version **1.31** and mujoco-py **0.5.7**
 * OpenAI Gym version **0.9.1**

Once Python 3 is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: The MuJoCo and Gym versions above are **not** the latest releases.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.