from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import shutil
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc danh sách các nguồn cụ thể
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các tiêu đề
)


UPLOAD_DIRECTORY = "uploads"


if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


def handle_img(img_path):
    np.set_printoptions(suppress=True)
    model = load_model("./models/keras_model.h5", compile=False)
    class_names = open("./models/labels.txt", "r", encoding="utf8").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
    return class_name[2:]


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    response = handle_img(file_location)
    data = {"info": f"{response}"}
    print(data)
    return {"info": f"{response}"}