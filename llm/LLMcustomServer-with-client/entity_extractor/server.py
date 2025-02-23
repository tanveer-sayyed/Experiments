from fastapi import Body, FastAPI, Response

from utils import performExtraction

app = FastAPI()

@app.get('/ask')
def ask(prompt:str = Body(..., embed=True)) -> Response:
    content = performExtraction(prompt=prompt)
    return Response(content=content, media_type="application/json")
