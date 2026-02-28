from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.routes import router

app = FastAPI()

app.include_router(router)

Instrumentator().instrument(app).expose(app)
