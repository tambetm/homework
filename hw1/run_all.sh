#!/bin/bash

python bc.py Ant-v2
python bc.py HalfCheetah-v2
python bc.py Hopper-v2
python bc.py Humanoid-v2
python bc.py Reacher-v2
python bc.py Walker2d-v2

python dagger.py Ant-v2
python dagger.py HalfCheetah-v2
python dagger.py Hopper-v2
python dagger.py Humanoid-v2
python dagger.py Reacher-v2
python dagger.py Walker2d-v2
