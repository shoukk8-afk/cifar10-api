import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image

#1 モデルの定義
model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

#2 重みのロード
model.load_state_dict(torch.load("cifar10_resnet18.pth", map_location="cpu"))
model.eval()

#transformの準備
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#CIFAR-10のラベル
categories = ["airplane", "automobile", "bird", "cat", "deer", 
              "dog", "frog", "horse", "ship", "truck"]

def get_prediction(image_bytes):
    #1. 生データをPillowイメージに変換
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    #2. 前処理を適用
    tensor = preprocess(image).unsqueeze(0) #バッチサイズを追加するため

    #3. 推論
    with torch.no_grad():
        outputs = model(tensor)

    #4. 最も確率の高いインデックスを取得
    _, predicted = outputs.max(1)
    return categories[predicted.item()]

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #1. 届いた画像ファイルをバイトデータとして読み込む
    image_bytes = await file.read()

    #2. 前処理 + 推論
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)

    #3. 結果をJSON形式で返す
    return {
        "prediction": categories[predicted.item()],
        "confidence": float(conf.item())
    }