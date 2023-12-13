import os
import json
import re
import pandas as pd
from collections import Counter
from models.preprocessor import load_data, preprocess_data, clean_and_replace,json_to_df
from models.word2vec_layers import Word2Vec
from models.lstm import LSTM
import numpy as np
import matplotlib.pyplot as plt
import pickle

####################################################
# def json_to_df(json_data):
#     # 필요한 정보를 추출하여 DataFrame으로
#     data = {
#         "essay_id": json_data["info"]["essay_id"],
#         "essay_type": json_data["info"]["essay_type"],
#         "essay_level": json_data["info"]["essay_level"],
#         "essay_length": json_data["info"]["essay_len"],
#         "essay_text": json_data["paragraph"][0]["paragraph_txt"],
#         "student_grade": json_data["student"]["student_grade"],
#         "essay_score_avg": json_data["score"]["essay_scoreT_avg"],
#         "essay_score_detail": json_data["score"]["essay_scoreT"]
#     }
#     return pd.DataFrame([data])
#
#
# # 데이터가 저장된 폴더 경로
# data_folder_path = "./data/Training/라벨링데이터"
# categories = ["글짓기", "대안제시", "설명글", "주장", "찬성반대"]
#
# all_dataframes = []
# # for category in categories:
# #     category_path = data_folder_path+'/'+ category
# #     print(f"Processing category: {category}")
# #
# #     # 각 카테고리 폴더 내의 모든 파일에 대해
# #     for file in os.listdir(category_path):
# #         if file.endswith('.json'):
# #             file_path = os.path.join(category_path, file)
# #             print(f"Reading file: {file_path}")
# #             with open(file_path, 'r', encoding='utf-8') as f:
# #                 json_data = json.load(f)
# #                 df = json_to_df(json_data)
# #                 all_dataframes.append(df)
#
# category_path = data_folder_path + '/' + "글짓기"
# # 각 카테고리 폴더 내의 모든 파일에 대해
# for file in os.listdir(category_path):
#     if file.endswith('.json'):
#         file_path = os.path.join(category_path, file)
#         with open(file_path, 'r', encoding='utf-8') as f:
#             json_data = json.load(f)
#             df = json_to_df(json_data)
#             all_dataframes.append(df)
#
#
#
# # DataFrame 병합
# if all_dataframes:
#     combined_df = pd.concat(all_dataframes, ignore_index=True)
#     print("\nCombined DataFrame:")
#     print(combined_df.head())
# else:
#     print("No dataframes were created. Check the file paths and JSON structure.")
#
#
# ## 1. 데이터 분석과 전처리
# import matplotlib.pyplot as plt
#
# # 1. 기본적인 통계 분석
# print("기본 통계 분석:")
# print(combined_df.describe())
#
# # 2. 결측치 확인
# print("\n결측치 확인:")
# print(combined_df.isnull().sum())
#
#
# # 3. 데이터 전처리
# # "#@문장구분#" 문자열을 공백으로 대체
# combined_df['essay_text'] = combined_df['essay_text'].str.replace('#@문장구분#', ' ')
#
# # 정규 표현식을 사용하여 비문자를 공백으로 대체
# combined_df['essay_text'] = combined_df['essay_text'].str.replace('[^a-zA-Z0-9]', ' ', regex=True)
#
# # 여러 개의 공백을 하나의 공백으로 대체
# combined_df['essay_text'] = combined_df['essay_text'].str.replace('\s+', ' ', regex=True)
#
# # 앞뒤 공백 제거 - 문자열의 앞뒤에 불필요한 공백이 생기는 것을 방지
# combined_df['essay_text'] = combined_df['essay_text'].str.strip()
#
#
# # 4. 데이터 탐색 및 시각화
# # 에세이 길이의 분포
# plt.figure(figsize=(10, 6))
# plt.hist(combined_df['essay_length'], bins=50, color='blue', alpha=0.7)
# plt.title('Essay Length Distribution')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.show()
#
# # 에세이 평균 점수의 분포
# plt.figure(figsize=(10, 6))
# plt.hist(combined_df['essay_score_avg'], bins=50, color='green', alpha=0.7)
# plt.title('Essay Average Score Distribution')
# plt.xlabel('Average Score')
# plt.ylabel('Frequency')
# plt.show()
#
# # 에세이 길이와 평균 점수의 상관관계
# plt.figure(figsize=(10, 6))
# plt.scatter(combined_df['essay_length'], combined_df['essay_score_avg'], alpha=0.5)
# plt.title('Correlation between Essay Length and Average Score')
# plt.xlabel('Essay Length')
# plt.ylabel('Average Score')
# plt.show()
#
# # 학년 레벨 별 분포 시각화
# plt.figure(figsize=(10, 6))
# combined_df['essay_level'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
# plt.title('Essay Levels Distribution')
# plt.xlabel('Level')
# plt.ylabel('Count')
# plt.xticks(rotation=0)  # 레벨 레이블이 가로로 표시되도록 설정
# plt.grid(True)
# plt.show()
#
# level_counts = combined_df['essay_level'].value_counts()
# print(level_counts)


##############################################################
# 데이터 전처리

# 데이터가 저장된 폴더 경로
data_folder_path = "./data/Training/원천데이터"
categories = ["글짓기", "대안제시", "설명글", "주장", "찬성반대"]

# 데이터 로딩
# essays, ids = load_data(data_folder_path, categories)
essays, ids = load_data(data_folder_path, ["글짓기"])

# 데이터 전처리
processed_essays = preprocess_data(essays)

# 데이터 클리닝 및 토큰화
cleaned_essays = clean_and_replace(processed_essays)

# 각 에세이의 id와 클리닝된 단어들을 매핑
mapped_results = dict(zip(ids, cleaned_essays))

# 5개의 에세이에 대해 첫 10단어 출력
for i, (essay_id, cleaned_words) in enumerate(mapped_results.items()):
    if i < 5:  # 처음 5개의 에세이에 대해서만 작동
        print(f"id = {essay_id}, 텍스트 = {cleaned_words[:10]}")
    else:
        break  # 5개 이후는 반복 중단

# Word2Vec 모델 학습
# 임베딩된 데이터 추출
# Pandas DataFrame으로 변환

# 단어-인덱스 매핑 생성
word_set = set()
for essay in cleaned_essays:
    word_set.update(essay)

word_to_index = {word: i for i, word in enumerate(word_set)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(word_to_index)

# 훈련 데이터 준비 (Skip-gram)
training_data = []
window_size = 1  # 윈도우 크기

for essay in cleaned_essays:
    essay_indices = [word_to_index[word] for word in essay]
    for i in range(len(essay_indices)):
        target = essay_indices[i]
        context = [essay_indices[j] for j in
                   range(max(0, i - window_size), min(len(essay_indices), i + window_size + 1)) if j != i]
        for context_word in context:
            training_data.append((target, context_word))

# word2vec 모델 저장 경로 설정
model_path = 'word2vec_embeddings.npy'

# 저장된 모델이 존재하는지 확인하고, 존재하면 로드
if os.path.exists(model_path):
    loaded_embeddings = np.load(model_path)
else:
    # 하이퍼파라미터 설정
    vocab_size = len(word_to_index)  # 단어장 크기
    embedding_dim = 100  # 임베딩 차원
    learning_rate = 0.01  # 학습률
    epochs = 50  # 학습 에포크 수

    # Word2Vec 모델 초기화
    model = Word2Vec(vocab_size, embedding_dim)

    losses = []  # 각 에포크의 평균 손실을 저장할 리스트

    # 훈련 루프
    for epoch in range(epochs):
        total_loss = 0
        for target_index, context_index in training_data:
            # Forward pass
            prediction = model.forward(target_index, context_index)

            # 여기서는 간단히 모든 타겟-컨텍스트 쌍을 긍정적 예제(1)로 가정
            label = 1
            loss = model.compute_loss(prediction, label)
            total_loss += loss

            # 그라디언트 계산 (여기서는 손실 함수의 도함수를 사용)
            gradient = -(label - prediction)

            # Backward pass
            model.backward(target_index, context_index, gradient, learning_rate)

        avg_loss = total_loss / len(training_data)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(training_data)}")

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()

    # 모델의 임베딩 행렬 저장
    np.save(model_path, model.embedding_matrix)


################
#   LSTM 학습
###
# 임베딩 벡터 변환
def words_to_vectors(words, word_to_index, embedding_matrix):
    return [embedding_matrix[word_to_index[word]] for word in words if word in word_to_index]

# cleaned_essays에서 각 에세이를 임베딩 벡터로 변환
embedded_essays = [words_to_vectors(essay, word_to_index, loaded_embeddings) for essay in cleaned_essays]

# 임베딩 데이터를 DataFrame으로 변환
embedding_df = pd.DataFrame({
    "essay_id": ids,
    "embedding": embedded_essays
})

# train_데이터가 저장된 폴더 경로
data_folder_labeling_path = "./data/Training/라벨링데이터/글짓기"


# JSON 파일로부터 추가 데이터 추출 및 병합
dataframes = []
for file_name in os.listdir(data_folder_labeling_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(data_folder_labeling_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            df = json_to_df(json_data)
            dataframes.append(df)

combined_df = pd.concat(dataframes)

# combined_df와 embedding_df를 essay_id를 기준으로 결합
final_df = pd.merge(combined_df, embedding_df, on="essay_id")

pd.set_option('display.max_columns', None)  # 모든 열을 출력하도록 설정
print(final_df.head())


# 데이터프레임에서 열 추출
selected_columns = ['embedding', 'essay_score_avg', 'essay_level', 'essay_length']
df = final_df[selected_columns]

# 임베딩 벡터와 평균 점수, essay_level, essay_length를 추출
embeddings = df['embedding'].values
scores = df['essay_score_avg'].values
levels = df['essay_level'].values
lengths = df['essay_length'].values

# 데이터 분할 비율 설정
test_size = 0.2
total_samples = len(embeddings)
test_samples = int(total_samples * test_size)

# 데이터를 랜덤하게 섞음
np.random.seed(42)
indices = np.arange(total_samples)
np.random.shuffle(indices)

# 훈련 데이터와 테스트 데이터로 나눔
train_indices = indices[test_samples:]
test_indices = indices[:test_samples]

X_train = embeddings[train_indices]
X_test = embeddings[test_indices]
y_train = scores[train_indices]
y_test = scores[test_indices]
level_train = levels[train_indices]
level_test = levels[test_indices]
length_train = lengths[train_indices]
length_test = lengths[test_indices]


def pad_sequences(sequences, maxlen, dtype='float32'):
    """주어진 시퀀스를 maxlen 길이에 맞춰 패딩 적용하는 함수"""
    num_samples = len(sequences)
    num_features = sequences[0].shape[1]
    padded_sequences = np.zeros((num_samples, maxlen, num_features), dtype=dtype)
    for i, sequence in enumerate(sequences):
        if len(sequence):
            padded_sequences[i, -len(sequence):] = sequence[:maxlen]
    return padded_sequences

X_train = [np.array(essay) for essay in X_train]
X_test = [np.array(essay) for essay in X_test]

# 가장 긴 시퀀스 길이 계산
maxlen = max(len(essay) for essay in X_train)

# 훈련 데이터와 테스트 데이터 패딩 적용
X_train_padded = pad_sequences(X_train, maxlen)
X_test_padded = pad_sequences(X_test, maxlen)


# 추가 특성을 패딩된 시퀀스와 결합
def add_features_to_sequences(sequences, features):
    num_samples = sequences.shape[0]
    num_features = sequences.shape[2] + features.shape[1]
    new_sequences = np.zeros((num_samples, sequences.shape[1], num_features))

    for i in range(num_samples):
        new_sequences[i, :, :sequences.shape[2]] = sequences[i]
        new_sequences[i, -1, sequences.shape[2]:] = features[i]

    return new_sequences


# 추가 특성 통합
features_train = np.hstack((level_train[:, np.newaxis], length_train[:, np.newaxis]))
features_test = np.hstack((level_test[:, np.newaxis], length_test[:, np.newaxis]))
X_train_combined = add_features_to_sequences(X_train_padded, features_train)
X_test_combined = add_features_to_sequences(X_test_padded, features_test)

# LSTM 모델 초기화
input_dim = X_train_combined.shape[2]  # 입력 데이터의 특성 수
hidden_dim = 10  # 은닉 상태의 차원

Wx = np.random.randn(4 * hidden_dim, input_dim + hidden_dim)
Wh = np.random.randn(4 * hidden_dim, hidden_dim)
b = np.random.randn(4 * hidden_dim)
lstm = LSTM(Wx, Wh, b)


# 학습 파라미터 설정
epochs = 10
learning_rate = 0.01

# 손실 기록을 위한 리스트 초기화
losses = []

# 예측을 위한 가중치 벡터 초기화
prediction_weights = np.random.randn(hidden_dim)

for epoch in range(epochs):
    total_loss = 0
    lstm.reset_grads()
    for i in range(len(X_train_combined)):
        x = X_train_combined[i]  # 훈련 데이터
        y = y_train[i]  # 실제 점수

        # x가 2차원인 경우, 3차원으로 변환
        if len(x.shape) == 2:
            x = x[np.newaxis, :]

        # 순전파
        h, c = lstm.forward(x, np.zeros((1, hidden_dim)), np.zeros((1, hidden_dim)))

        # 가중 합을 통한 단일 예측 값 계산
        prediction = np.dot(h[-1], prediction_weights)

        # 손실 계산 (MSE)
        loss = np.mean((prediction - y) ** 2)

        # 손실에 대한 그라디언트
        dh_pred = 2 * (prediction - y)  # dh는 이제 스칼라 값

        # dh_pred를 은닉 상태 그라디언트 dh에 전파
        dh = np.zeros_like(h[-1])
        dh = dh_pred * prediction_weights

        # 각 시간 단계에 대해 역전파
        for t in reversed(range(len(x))):
            dx, dh, dc = lstm.backward(dh, np.zeros_like(c))

        # 파라미터 업데이트
        for param, grad in zip(lstm.params, lstm.grads):
            param -= learning_rate * grad

        total_loss += loss

    avg_loss = total_loss / len(X_train_combined)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

# 손실 그래프 그리기
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

# 모델 파라미터 저장
model_params = lstm.params
np.save('lstm_model_params.npy', model_params)











# 모델 파라미터 저장
# model_params = {
#     'W_in': in_layer.params[0],
#     'W_out': out_layer.params[0]
# }

# with open(model_file, 'wb') as f:
#     pickle.dump(model_params, f)


# 이상치 제거 필요하다면 진행.

#
# 예시로 첫 번째 에세이의 길이를 출력
# print("첫 번째 에세이의 길이:", len(essays[0]))
#
# # Numpy를 사용하여 에세이 길이의 평균과 표준편차 계산
# lengths = np.array([len(essay) for essay in essays])
# print("평균 길이:", np.mean(lengths))
# print("표준편차:", np.std(lengths))
# 이상치를 제거하는 함수
# def remove_outliers(essays, mean, std, threshold=2):
#     filtered_essays = [essay for essay in essays if mean - threshold*std <= len(essay) <= mean + threshold*std]
#     return filtered_essays
#
# # 이상치 제거
# filtered_essays = remove_outliers(essays, np.mean(lengths), np.std(lengths))
#
# # 정규화를 위한 길이 계산
# lengths = np.array([len(essay) for essay in filtered_essays])
#
# # Min-Max 정규화
# normalized_lengths = (lengths - lengths.min()) / (lengths.max() - lengths.min())
#
# # 결과 출력
# print("정규화된 길이:", normalized_lengths[:5])  # 처음 5개의 정규화된 길이 출력
