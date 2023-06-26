from generate_demo import abs_summary,generate_from_index
from pydantic import BaseModel
from fastapi import FastAPI
app = FastAPI()

class Article(BaseModel):
    content: str

@app.get("/api/summarize") 
async def get_menu_rec(dataset_name: str, idx : int):
    return generate_from_index(dataset_name, idx = idx)

@app.post("/api/summarize")
async def get_menu_rec(art : Article):
    art_dict = art.dict()
    return abs_summary(input_text = art_dict['content'])

    