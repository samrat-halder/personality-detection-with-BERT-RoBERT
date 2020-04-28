mkdir model_folder
#mkdir data
cd model_folder
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py
wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py
wget https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py
wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py
sudo apt-get intall python3-dev
sudo apt-get install build-essential
sudo apt-get install ipython3
sudo apt-get install python3-pip
python3 -m pip install pandas
python3 -m pip install nltk
