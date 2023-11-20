import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import emoji
from soynlp.normalizer import repeat_normalize

# 모델과 토크나이저 로드
model_path = 'kc_electra_model.pt'  # 모델 파일 경로
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# 데이터 전처리 함수
def clean(x):
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(r'(http|https):\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    x = pattern.sub(' ', str(x))
    x = emoji.replace_emoji(x, replace='')
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

# 악성 여부 판단 함수
def predict_toxicity(text):
    text = clean(text)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return '악성' if prediction == 1 else '비악성'

# 예시 사용
text = "여기에 분석할 텍스트를 입력하세요."
print(f"입력 텍스트: {text}\n분석 결과: {predict_toxicity(text)}")
