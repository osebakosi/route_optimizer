from fastapi import FastAPI
import uvicorn
from app.router import router as app_router

app = FastAPI()
app.include_router(app_router)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
