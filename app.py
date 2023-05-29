from generate import abs_summary
from pydantic import BaseModel
from fastapi import FastAPI
app = FastAPI()

class Article(BaseModel):
    content: str

@app.get("/api/sum") 
async def get_menu_rec(idx : int):
    return abs_summary(input_text = None,idx = idx)

@app.post("/api/sum")
async def get_menu_rec(art : Article):
    art_dict = art.dict()
    return abs_summary(input_text = art_dict['content'])

    