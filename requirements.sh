mkdir model_folder
cd model_folder
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py
wget https://raw.githubusercontent.com/google-research/bert/master/optimization.py
wget https://raw.githubusercontent.com/google-research/bert/master/run_classifier.py
wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py

pip3 install pandas
pip3 install nltk
