cd ~
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6
sudo rm /usr/bin/python3
sudo ln -s /usr/bin/python3.6 /usr/bin/python3
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.6 get-pip.py
sudo apt-get install build-essential
export LC_ALL="en_US.UTF-8" 
export LC_CTYPE="en_US.UTF-8" 
sudo dpkg-reconfigure locales
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb 
mv cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc
sudo dpkg -i libcudnn7_7.0.5.15–1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15–1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.0.5.15–1+cuda9.0_amd64.deb
sudo apt-get install python3.6-gdbm
sudo apt-get install htop
