import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# 데이터 불러오기
raw_text = pd.read_csv(r"D:\학습\250529\NLP-practice-main\250529_NLP3\q4\emotions_train.txt", delimiter=';', header=None, names=['sentence','emotion'])
train_data = raw_text['sentence']
train_emotion = raw_text['emotion']

# 1. CountVectorizer 객체 생성 후 학습 데이터 변환
cv = CountVectorizer()
transformed_text = cv.fit_transform(train_data)

# 2. MultinomialNB 객체 생성 후 학습
clf = MultinomialNB()
clf.fit(transformed_text, train_emotion)

# 3. 테스트 문장들
test_data = ['i am curious', 'i feel gloomy and tired', 'i feel more creative', 'i feel a little mellow today']

# 테스트 데이터를 벡터화하여 감정 예측
test_transformed = cv.transform(test_data)
test_result = clf.predict(test_transformed)

print(test_result)
