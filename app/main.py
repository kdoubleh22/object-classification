from fastapi import FastAPI, File, UploadFile

import torch
import clip
from PIL import Image
from io import BytesIO

app = FastAPI()

@app.post("/clip")
async def predictions_objects_clip(file: UploadFile = File(...)):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = Image.open(BytesIO(await file.read()))
    labels = ["airplane", "apple", "ball", "banana", "bicycle", "book", "broccoli", "burger", "bus",
            "cake", "candy", "cap", "cat", "chair", "chopsticks", "cookie", "crayon", "cup", "dinosaur",
            "dog", "duck", "eraser", "firetruck", "flower", "fork", "glasses", "grape", "icecream",
            "milk", "orange", "pencil", "penguin", "piano", "pizza", "policecar", "scissors",
            "socks", "spoon", "strawberry", "table", "tiger", "toothbrush", "tree", "television", "window"]

    image = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)

    with torch.no_grad():
        image_feature = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_feature @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(similarity.size(-1))

    for value, index in zip(values, indices):
        print(f"{labels[index]:>15s} : {100 * value.item():.2f}%")

    return {'prediction' : labels[indices[0]]}