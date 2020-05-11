# MBTI type personality detection with BERT and RoBERT
Detection of MBTI-type personality with state-of-art language models

**Software version: Ubuntu version 16.04 | Tensorflow-gpu version 1.11 | Keras 1.0.8 | cuda 9.0 | python 3.6.10**

*Minimum server requirements: 30 Gb CPU, 15 Gb NVIDIA P100 GPU*

Step-by-step instructions for setting up the environment:
1. Download libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb, libcudnn7-doc_7.4.2.24-1+cuda9.0_amd64.deb, libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb to your home folder from Nvidia developer website. Please note the versions (the module may not be compatible with any other version)
2. Clone the git repository to your home
3. ``` cd /$repository/setup/```
4. Run setup.sh from setup dir with command ```sh setup.sh ```
5. Naviagte back to home dir ```cd ../``` 
6. Run ```sh requirement.sh```
7. Import bash file by ```source ~/.bashrc```
8. copy mbti9k_comments.csv file (2GB) to ./data/


Step-by-step instructions for running the data preparation modules and models:
1. cd ```./src```. Run ```python3 src/data_prep.py``` to create training samples for both the models
