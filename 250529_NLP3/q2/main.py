import pandas as pd

# 감정 내 단어별 빈도 계산 함수
def cal_partial_freq(texts, emotion):
    partial_freq = dict()
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    
    for sentence in filtered_texts:
        words = sentence.split()
        for word in words:
            partial_freq[word] = partial_freq.get(word, 0) + 1
    
    return partial_freq

# 감정 내 전체 단어 빈도 합계 계산 함수
def cal_total_freq(partial_freq):
    total = sum(partial_freq.values())
    return total

# 데이터 불러오기 (정확한 경로 지정)
data = pd.read_csv(r"D:\학습\250529\NLP-practice-main\250529_NLP3\q2\emotions_train.txt", delimiter=';', header=None, names=['sentence','emotion'])

# happy가 joy라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
joy_freq = cal_partial_freq(data, "joy")
joy_total = cal_total_freq(joy_freq)
joy_likelihood = joy_freq.get("happy", 0) / joy_total
print("joy_likelihood:", joy_likelihood)

# happy가 sadness라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
sad_freq = cal_partial_freq(data, "sadness")
sad_total = cal_total_freq(sad_freq)
sad_likelihood = sad_freq.get("happy", 0) / sad_total
print("sad_likelihood:", sad_likelihood)

# can이 surprise라는 감정을 표현하는 문장에서 발생할 가능도를 구하세요.
sup_freq = cal_partial_freq(data, "surprise")
sup_total = cal_total_freq(sup_freq)
sup_likelihood = sup_freq.get("can", 0) / sup_total
print("surprise_likelihood (can):", sup_likelihood)
