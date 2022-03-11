from typing import Optional
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from text_translation import pipeline as translate_text

app = FastAPI()


@app.get('/')
def read_root():
    return {"hello":"word"}

@app.get("/text/{text}")
def read_item(text: str):
    output = translate_text(text)
    print(output)
    return {"signs": output}



# uvicorn main:app --port 8001 --reload