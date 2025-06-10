from flask import Flask, request, jsonify
import pickle
import warnings
warnings.filterwarnings(action='ignore')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query = json_['infer_texts']  # 문자열 리스트

    # 전달받은 query를 cv로 벡터화 후 clf로 감정 예측
    transformed_query = cv.transform(query)
    preds = clf.predict(transformed_query)

    # 결과를 "문서 인덱스: 예측된 감정" 형태로 딕셔너리에 저장
    response = {str(i): pred for i, pred in enumerate(preds)}

    return jsonify(response)

if __name__ == '__main__':
    with open('nb_model.pkl', 'rb') as f:
        cv, clf = pickle.load(f)

    app.run(host='0.0.0.0', port=8080)
