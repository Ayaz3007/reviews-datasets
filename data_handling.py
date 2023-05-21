import nltk
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import fastparquet
import re
import pymorphy2
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from pyarrow.parquet import ParquetFile

#приводим в стандартной форме слова
morph = pymorphy2.MorphAnalyzer()
def morph_analyzer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([morph.parse(word)[0].normal_form for word in text.split()])
    return text

#оставляем только слова
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^\u0401\u0451\u0410-\u044f\s]'
    text=re.sub(pattern,'',text)
    return text

tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('russian')

#удаляем ненужные слова
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

data_reviews = pd.read_parquet(r"train.parquet")
data_reviews['text']=data_reviews['text'].apply(remove_special_characters)
data_reviews['text']=data_reviews['text'].apply(morph_analyzer)
data_reviews['text']=data_reviews['text'].apply(remove_stopwords)

norm_train_reviews = data_reviews
norm_train_reviews.to_parquet('norm_train_reviews.parquet', engine='fastparquet')

data_test_reviews = pd.read_parquet(r"test.parquet")
data_test_reviews['text']=data_test_reviews['text'].apply(remove_special_characters)
data_test_reviews['text']=data_test_reviews['text'].apply(morph_analyzer)
data_test_reviews['text']=data_test_reviews['text'].apply(remove_stopwords)

norm_test_reviews = data_test_reviews
norm_test_reviews.to_parquet('norm_test_reviews.parquet', engine='fastparquet')

