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
8. copy mbti9k_comments.csv file (~2GB) to ./data/ (This file can be obtained on request)



Step-by-step instructions for running the codes:
1. ```cd ./src``` 
2. To quickly check a summary of the data run ```python3 data_summary.py```. You may need to change the default parameters *n_sample = 999999 all_class = True*. This will output a file in data directory with sentence length for each user. Please note this may take long time if you run for the entire dataset. *999999* is the default value that runs for the whole sample.
2. Run ```python3 src/data_prep.py``` to create training samples for both the models. You may want to change the default parameters *seq_length = 150, overlap_length = 25, n_sample = 999999, all_class = False*. This will create files in the data directory for BERT and RoBERT. The default value for *n_sample* took almost three hours in our machine.
3. Run ```python3 train_bert.py``` with default parameters *TRAIN_BATCH_SIZE = 4, EVAL_BATCH_SIZE = 2, LEARNING_RATE = 1e, NUM_TRAIN_EPOCHS = 1.0, WARMUP_PROPORTION = 0.1, MAX_SEQ_LENGTH = 150, NUM_SAMPLE = 4500, uncased = True #False, all_class = True* This will fine tune the BERT model for the classification task. It will *create model.ckpt-NUM* files in the model_folder/outputs directory. 
4. Run ```python3 predict_bert.py```. If you set the parameter to *FLAG = 'H'* then it will prepare a dataset with grouped embeddings for the entire document of each individual user and pass it to the RoBERT model. Otherwise it will run the fine tuned BERT model on a test dataset. Also you need to set the *BERT_NUM* parameter from the *ckpt* file from step 3. Please refer to the script for other parameters. 
5. Run ```python3 run_model_roBert.py``` to run the RoBERT model.

*Please note some of the functions of this repository was taken from Google Research's BERT repository*

**Results**: We achieved 41% accuracy on an 8-class and 43% accuracy on a 4-class classification problem with MBTI type personality detection in our research. We also ran the 4-class classification task with RoBERT for the full document and achieved the best accuracy of 29.8%.

For 4-class classification the fine-tuning and training BERT-base model took 2 hr 25 mins with 230,428 examples (each of length 150 words). For the same classification task with RoBERT using 10 epochs, batch size of 5 took 2 hr 43 minutes with 7218 samples (whole document). Other computation times for data preparation and other experiments can be found in the log files.
