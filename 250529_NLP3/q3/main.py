import pandas as pd
import numpy as np

# 단어의 감정별 빈도수 계산
def cal_partial_freq(texts, emotion):
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    partial_freq = dict()

    for sentence in filtered_texts:
        for word in sentence.split():
            partial_freq[word] = partial_freq.get(word, 0) + 1

    return partial_freq

# 감정별 전체 단어 수 계산
def cal_total_freq(partial_freq):
    return sum(partial_freq.values())

# 감정의 사전 확률 (Prior) 계산
def cal_prior_prob(data, emotion):
    total_docs = len(data)
    emotion_docs = len(data[data['emotion'] == emotion])
    return np.log(emotion_docs / total_docs)

# 감정 예측 함수 (나이브 베이즈)
def predict_emotion(sent, data_path):
    emotions = ['anger', 'love', 'sadness', 'fear', 'joy', 'surprise']
    predictions = []
    train_txt = pd.read_csv(data_path, delimiter=';', header=None, names=['sentence', 'emotion'])

    vocabulary = set()
    for sentence in train_txt['sentence']:
        vocabulary.update(sentence.split())
    vocab_size = len(vocabulary)
    smoothing = 10

    for emotion in emotions:
        log_prob = cal_prior_prob(train_txt, emotion)

        partial_freq = cal_partial_freq(train_txt, emotion)
        total_freq = cal_total_freq(partial_freq)

        for word in sent.split():
            word_freq = partial_freq.get(word, 0)
            likelihood = (word_freq + smoothing) / (total_freq + smoothing * vocab_size)
            log_prob += np.log(likelihood)

        predictions.append((emotion, log_prob))

    # 확률이 가장 높은 감정을 반환
    return max(predictions, key=lambda x: x[1])

# 예측할 문장
test_sent = "i really want to go and enjoy this party"
data_path = r"D:\학습\250529\NLP-practice-main\250529_NLP3\q3\emotions_train.txt"
predicted = predict_emotion(test_sent, data_path)
print(predicted)

