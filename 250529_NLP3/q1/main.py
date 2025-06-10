from sklearn.model_selection import train_test_split

data = []
file_path = r'D:\학습\250529\NLP-practice-main\250529_NLP3\q1\emotions_train.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                print(f"[빈 줄] {idx}번째 줄 건너뜀")
                continue
            if ';' not in line:
                print(f"[형식 오류] {idx}번째 줄: {line}")
                continue
            sentence, emotion = line.split(';', 1)
            data.append((sentence, emotion))
except FileNotFoundError:
    print(f"[오류] 파일을 찾을 수 없습니다: {file_path}")

print(f"[결과] data에 저장된 튜플 수: {len(data)}")

if data:
    train, test = train_test_split(data, test_size=0.2, random_state=7)

    Xtrain = [x[0] for x in train]
    Ytrain = [x[1] for x in train]

    print("학습 데이터 문장 개수:", len(Xtrain))
    print("학습 데이터 감정 종류:", set(Ytrain))

    Xtest = [x[0] for x in test]
    Ytest = [x[1] for x in test]

    print("평가 데이터 문장 개수:", len(Xtest))
    print("평가 데이터 감정 종류:", set(Ytest))
else:
    print("🚫 데이터가 비어 있어 train_test_split을 수행할 수 없습니다.")
