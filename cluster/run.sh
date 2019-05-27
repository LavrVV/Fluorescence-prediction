#!/bin/sh
# Installation Section
  # in this section, you can install required packages (programs)
  # before installing, you must update the package index
  sudo apt-get update
  sudo apt-get -y upgrade
  # INSTALLATION EXAMPLE
  # sudo apt-get install -yqq cmake wget gfortran \
  #                           sed locales \
  #                           nodejs  
  sudo apt-get install -y curl
  sudo apt-get install -y git
  sudo apt-get install -y cmake
  sudo apt-get install -y python3
  sudo apt-get install -y python3-pip --fix-missing
  sudo apt-get install -y gfortran libopenblas-dev liblapack-dev
  sudo apt-get update gcc
  sudo apt-get install -y zlib1g-dev
  sudo apt-get install -y libjpeg-dev
  
  echo ##############
  #curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
  #sh Anaconda3-5.2.0-Linux-x86_64.sh
  echo ##############
  
  export LC_ALL=C
  #sudo export LC_ALL="en_US.UTF-8"
  #sudo export LC_CTYPE="en_US.UTF-8"
  #sudo dpkg-reconfigure locales
  sudo pip3 install --upgrade pip
  #sudo pip3 install pandas
  #sudo pip3 install numpy

  #sudo pip3 install scipy
  #sudo pip3 install scikit-learn
  #sudo pip3 install Pillow
  #sudo pip3 install pyyaml
  #sudo pip3 install git+https://github.com/pytorch/pytorch
  #sudo pip3 install torchvision
  #sudo pip3 install -U pydicom
  sudo pip3 install xgboost
  
  #sudo pip3 install torch torchvision
  #sudo conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# Configuration Section
  # in this section, you can configure installed packages, start necessary services, copy files, etc.

# Execution Section
  # this section contains the executable code for the calculation, or a link to the file containing the executable code for the calculation
  python3 -c "import xgboost"
  nvcc --version
  openssl version
  pip3 --version
  gcc --version
  #python3 imports.py
  #python3 cluster.py