from fastapi import FastAPI
from pydantic import BaseModel
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from tokenizer.core.mt_translate_core import translate
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizer.core.mt_translate_core import translate
app = FastAPI()

class Req(BaseModel):
    text: str
    max_len: int = 40

class Res(BaseModel):
    translation: str

@app.post("/translate", response_model=Res)
def t(req: Req):
    return Res(translation=translate(req.text, req.max_len))

@app.get("/health")
def h():
    return {"ok": True}
