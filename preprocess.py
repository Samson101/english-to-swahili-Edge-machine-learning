import pandas as pd
from colabcode import ColabCode
from sklearn.model_selection import  train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.colab import files
uploaded = files.upload()
import io
data = pd.read_csv(io.StringIO(uploaded['eng-swahili.csv'].decode('utf-8')))
english_sentences = data['english'].values
swahili_sentences = data['swahili'].values

#tokenizing the sentences
tokenizer=Tokenizer()
#fit the tokenizer on english sentences
tokenizer.fit_on_texts(english_sentences)
english_token=tokenizer.texts_to_sequences(english_sentences)
#swahili tokenizer
tokenizer.fit_on_texts(swahili_sentences)
swahili_token=tokenizer.texts_to_sequences(swahili_sentences)
#split the data into training and test sets
english_train, english_test, swahili_train, swahili_test=train_test_split(english_token,swahili_token, test_size=0.2)
#truncating the sequence
max_length=20
english_train=pad_sequences(english_train, maxlen=max_length, padding='post')
english_test=pad_sequences(english_test, maxlen=max_length, padding='post')
swahili_train=pad_sequences(swahili_train, maxlen=max_length, padding='post')
swahili_test=pad_sequences(swahili_test, maxlen=max_length, padding='post')