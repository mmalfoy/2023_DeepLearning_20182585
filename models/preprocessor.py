import os
import re
import json
import pandas as pd


# 데이터 로드 함수
def load_data(folder_path, categories):
    essays = []
    ids = []

    for category in categories:
        category_path = os.path.join(folder_path, category)
        for file in os.listdir(category_path):
            if file.endswith(".json"):
                file_path = os.path.join(category_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    essay_text = data['essay_txt']  # JSON 데이터의 "essay_txt" 항목
                    essay_id = data['essay_id']  # JSON 데이터의 "essay_id" 항목
                    essays.append(essay_text)
                    ids.append(essay_id)

    return essays, ids


# 데이터 전처리 및 토큰화 함수
def preprocess_data(essays):
    processed_essays = []

    removal_list = ['은', '라는', '라고', '라는', '는', '이', '가', '을', '를', '에', '에게', '에게서', '이다', '만', '밖에', '도', '까지', '대로',
                    '조차', '다']
    sentence_separator = '#@문장구분#'

    for essay in essays:
        processed_words = []
        # '#@문장구분#'을 공백으로 대체
        essay = essay.replace(sentence_separator, ' ')

        # 여러 개의 공백을 하나로 축소
        essay = re.sub('\s+', ' ', essay)

        # 문장을 단어 단위로 분할
        words = essay.split()

        # 단어를 처리하여 결과 리스트에 추가
        for word in words:
            # 특수문자 제거
            word = re.sub('[^가-힣A-Za-z0-9\s]', '', word)

            # 제거할 단어들을 찾아 제거 (단어 끝에 있을 때만)
            for remove_word in removal_list:
                if word.endswith(remove_word):
                    word = word[:-len(remove_word)].strip()

            # 공백 제거
            word = word.strip()

            # 빈 문자열 제거
            if len(word) > 0:
                processed_words.append(word)

        # 현재 에세이의 처리된 단어 리스트를 추가
        processed_essays.append(processed_words)

    return processed_essays


# 데이터 클리닝 함수
def clean_and_replace(processed_essays):
    cleaned_essays = []
    # 변환할 단어들의 매핑
    word_mapping = {
        '했': '하다',
        '한': '하다',
        '보았': '보다',
        '보였': '보다',
        '있': '있다',
        '혔': '히다',
        '됐': '되다',
        '된': '되다',
        '지': '지다',
        '졌': '지다',
        '왔': '오다',
        '렸': '리다',
        '났': '나다',
        '냈': '내다',

        # 추가 매핑
    }

    for processed_words in processed_essays:
        cleaned_words = []  # 현재 에세이의 클리닝된 단어들을 저장할 리스트

        for word in processed_words:
            # 단어 끝에 매핑할 단어가 있는지 확인하고 변환
            for mapping_word, replacement in word_mapping.items():
                if word.endswith(mapping_word):
                    word = word[:-len(mapping_word)] + replacement

            cleaned_words.append(word)  # 현재 에세이의 클리닝된 단어 추가

        cleaned_essays.append(cleaned_words)  # 현재 에세이의 클리닝된 단어 리스트를 추가

    return cleaned_essays

# JSON 데이터를 데이터 프레임으로 변환
def json_to_df(json_data):
    data = {
        "essay_id": json_data["info"]["essay_id"],
        "essay_level": json_data["info"]["essay_level"],
        "essay_length": json_data["info"]["essay_len"],
        "essay_score_avg": json_data["score"]["essay_scoreT_avg"]
    }
    return pd.DataFrame([data])
