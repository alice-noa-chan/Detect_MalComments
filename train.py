import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import emoji
from soynlp.normalizer import repeat_normalize
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# KcELECTRA 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=2)

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

# PyTorch Dataset
class CommentDataset(Dataset):
    def __init__(self, comments, labels):
        self.comments = comments
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments[idx], self.labels[idx]

# 데이터 로드
df = pd.read_csv('ko_comments.tsv', sep='\t')
df['content'] = df['content'].apply(clean)
comments = df['content'].tolist()
labels = df['label'].tolist()

# 데이터를 학습 및 검증 데이터셋으로 분리
train_comments, val_comments, train_labels, val_labels = train_test_split(comments, labels, test_size=0.2, random_state=42)

# 오버샘플링
ros = RandomOverSampler(random_state=42)
train_comments_res, train_labels_res = ros.fit_resample(np.array(train_comments).reshape(-1, 1), train_labels)
train_comments_res = train_comments_res.flatten().tolist()
print("Resampled dataset shape:", Counter(train_labels_res))

# 데이터를 PyTorch Dataset으로 변환
train_dataset = CommentDataset(train_comments_res, train_labels_res)
val_dataset = CommentDataset(val_comments, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# 모델 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# early stopping criteria
early_stop = False
best_loss = np.inf
patience = 2  # stop training if no improvement in loss for 2 consecutive epochs
counter = 0

# 모델 학습
model.train()
for epoch in range(5):  # 5 epoch 학습
    if early_stop:
        print("Early stopping.")
        break
    epoch_loss = 0.0
    correct = 0
    total = 0

    # Training Phase
    data_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Train epoch {epoch}")
    for i, (texts, labels) in data_iter:
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        labels = labels.to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        data_iter.set_postfix({'loss': epoch_loss / (i + 1), 'accuracy': correct / total})

    # Validation Phase
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for texts, labels in val_dataloader:
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f"\nValidation loss: {val_loss:.4f}, accuracy: {correct / total:.4f}")
    model.train()

    # Check early stopping conditions
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'kc_electra_model.pt')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            early_stop = True

    # Adjust learning rate
    scheduler.step(val_loss)

# 모델 저장
torch.save(model.state_dict(), 'kc_electra_model.pt')