from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from predictor import *
from typing import Optional


app_desc = """<h2>Try this app by uploading any image</h2>
<h2>API for Face Spoofing</h2>"""

class Item(BaseModel):
    file: UploadFile = File(...)
    threshold: float


app = FastAPI(title='Face Spoofing API', description=app_desc)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...), threshold: Optional[float] = None ):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "JPG")
    if not extension:
        return "Image must be jpg or png format!"

    image = read_imagefile(await file.read())
    prediction = predict(image,threshold)
    return prediction
