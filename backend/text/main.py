from typing import Optional
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from text_translation import pipeline as translate_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3001",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:5501",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def read_root():
    return {"hello":"word"}

@app.get("/text/{text}")
def read_item(text: str):
    output = translate_text(text)
    print(output)
    return {"signs": output}


# note انا وانت should be equal to انا و انت
# انا اسمى & انا اسمي
#  need to handle errors 


# uvicorn main:app --port 8002 --reload