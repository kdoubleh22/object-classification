# object-classification

## 📌 목차  
0. 요약
1. 실행 방법
2. API 예시
3. 주요 코드 설명 
4. 레퍼런스

## 0. 요약

![image](https://github.com/user-attachments/assets/b2441497-597f-4857-bf54-28f3c1583a03)

AI를 활용한 영어학습 앱 프로젝트 중 사물 분류 API 부분입니다. OpenAI CLIP 모델 활용했습니다.

## 1. 실행 방법

```jsx
1. git clone.
git clone https://github.com/kdoubleh22/object-classification.git

2. 프로젝트 폴더로 이동.
cd object-classification

3. docker 빌드.
docker build -t name .

4. docker run.
docker run -p 80:80 name
```

## 2. API 예시

![image](https://github.com/user-attachments/assets/bc9dd4c5-0060-4ddc-9bbf-8d7d5e029868)

## 3. 주요 코드 설명

### 3-1. main.py [1]

```python
model, preprocess = clip.load("ViT-B/32", device=device)
```

CLIP의 여러 모델 중 Vision Transformer 구조, base 크기, 32 패치 크기의 모델을 load합니다. clip.available_models()를 통해 ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'] 상황에 맞는 다른 모델을 선택할 수 있습니다.

```python
labels = ["airplane", "apple", "ball", "banana", "bicycle", "book", "broccoli", "burger", "bus",
            "cake", "candy", "cap", "cat", "chair", "chopsticks", "cookie", "crayon", "cup", "dinosaur",
            "dog", "duck", "eraser", "firetruck", "flower", "fork", "glasses", "grape", "icecream",
            "milk", "orange", "pencil", "penguin", "piano", "pizza", "policecar", "scissors",
            "socks", "spoon", "strawberry", "table", "tiger", "toothbrush", "tree", "television", "window"]
```

labels에 분류를 원하는 단어를 넣습니다. 프로젝트의 단어장에 들어갈 단어를 지정해줬습니다.

```python
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)
```

“a photo of a”를 붙이는 이유는 성능을 높이기 위함입니다.

“

we found that using the prompt template “A photo of a {label}.” to be a good default that helps specify the text is about the content of the image. This often improves performance over the baseline of using only the label text. For instance, just using this prompt improves accuracy on ImageNet by 1.3%.

“ [2]

CLIP이 사전학습한 데이터는 (word, image)가 아닌 (text, image) 데이터 쌍으로 학습했기 때문입니다.

```python
with torch.no_grad():
    image_feature = model.encode_image(image)
    text_features = model.encode_text(text_inputs)
```

특징 벡터를 추출합니다.

```python
image_feature /= image_feature.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(similarity.size(-1))
```

정규화한 후 코사인 유사도를 구하고 내림차순합니다.

```python
for value, index in zip(values, indices):
print(f"{labels[index]:>15s} : {100 * value.item():.2f}%")
```

결과를 출력합니다.

![image](https://github.com/user-attachments/assets/a5f5075c-a946-4ee9-bcd7-67bfa416984c)

## 4. 레퍼런스

[1] https://github.com/openai/CLIP

[2] https://arxiv.org/abs/2103.00020
