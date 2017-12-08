# Simple game

The simple game environment demonstrates how to make a game for openai.
I use python3 because it is required by the openai benchmarks.

Make sure to add:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/gym-sumo/
```
to ~/.bashrc if you want to import outside of this folder.

# Installation

INSTALL SUMO
sudo apt-get install build-essential
sudo apt-get install autotools-dev
sudo apt-get install autoconf
sudo apt-get install automake
sudo apt-get install libtool
sudo apt-get install subversion
sudo apt-get install libgdal1-dev
sudo apt-get install libgl1-mesa-dev
sudo apt-get install libglu1-mesa-dev
sudo apt-get install ffmpeg
sudo apt-get install libfox-1.6-dev
sudo apt-get install python-dev

svn co https://svn.code.sf.net/p/sumo/code/trunk/sumo

cd sumo
make -f Makefile.cvs
./configure --with-python --without-ffmpeg

make
sudo make install
export PYTHONPATH=$PYTHONPATH:/<path to sumo>/sumo/tools

sudo apt install python-pip
pip install numpy
sudo pip install matplotlib
sudo pip install scipy
sudo apt-get install python-tk

INSTALL OPENAI
sudo apt-get install xvfb libav-tools xorg-dev libsdl2-dev swig cmake
git clone https://github.com/openai/gym
cd gym
sudo pip install -e .
sudo pip install -e .[all]

```bash
cd gym-sumo
sudo pip3 install -e .
```

# Run

```bash
python3 run_sumo.py
```

