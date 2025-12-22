from fastapi import FastAPI
from src.app.main import router
import uvicorn

app = FastAPI(title="Face Mask Detector API", version="1.0")
app.include_router(router, prefix="/api/v1")

if __name__=="__main__":
    uvicorn.run("src.main:app", host="0.0.0.0",port=7001, reload=True)