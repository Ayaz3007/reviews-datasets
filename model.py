import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_parquet('norm_train_reviews.parquet', engine='pyarrow')
data_test = pd.read_parquet('norm_train_reviews.parquet', engine='pyarrow')

# Объединяем тренировочные и тестовые данные
all_data = pd.concat([data, data_test], ignore_index=True)

# Разделим данные на признаки (X) и целевую переменную (y)
X = all_data['text']
y = all_data['labels']

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_features, y_train)

# Прогнозы на тестовых данных
y_pred = model.predict(X_test_features)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall:", recall)
print("F1 score: ", f1)
print("AUC-ROC: ", auc_roc)

example_review = ["Ужасный фильм, не знаю, как такое вообще можно смотреть. Отвратительно!"]
example_vector = vectorizer.transform(example_review)
example_prediction = model.predict(example_vector)

if example_prediction == 1:
    print("Негативный отзыв")
else:
    print("Положительный отзыв")

example_review = ["«American Psycho» — без сомнения, один из лучших психологических слэшеров; Введение титров идеально, представление персонажа ещё лучше, в нём лучшие диалоги, лучшая музыка и самый чёрный юмор, это очень оригинально. Игра Бэйла фантастическая, другой актёр, наверно, не смог бы сыграть лучше, декорации фильма идеальны, любой готов поклясться, что «Американский психопат» действительно был снят в 80-х... Одним словом, это шедевр, отражающий материализм, мачизм и тщеславие, которое жило в то время в окружении этих персонажей; Это определённо не фильм для тех, кто привык смотреть лёгкие фильмы, если вы не из тех, кто любит, чтобы его заставили задуматься, лучше не смотрите его. Невероятный фильм, который граничит с чертой, отделяющей обычное кино от шедевров. Кроме того, интерпретируется по-разному."]
example_vector = vectorizer.transform(example_review)
example_prediction = model.predict(example_vector)

if example_prediction == 1:
    print("Негативный отзыв")
else:
    print("Положительный отзыв")

text = ' '.join(data['text'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
