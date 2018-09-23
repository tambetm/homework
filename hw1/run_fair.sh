#!/bin/bash

python bc_fair.py Ant-v2 $*
python bc_fair.py HalfCheetah-v2 $*
python bc_fair.py Hopper-v2 $*
python bc_fair.py Humanoid-v2 $*
python bc_fair.py Reacher-v2 $*
python bc_fair.py Walker2d-v2 $*

python dagger.py Ant-v2 $*
python dagger.py HalfCheetah-v2 $*
python dagger.py Hopper-v2 $*
python dagger.py Humanoid-v2 $*
python dagger.py Reacher-v2 $*
python dagger.py Walker2d-v2 $*
