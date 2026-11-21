
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.squat_api import router as squat_router
from backend.app.api.vertical_jump_api import router as vertical_jump_router
from backend.app.api.sprint_api import router as sprint_router
from backend.app.api.history_api import router as history_router
from backend.app.api.pushup_api import router as pushup_router
from backend.app.api.lunge_api import router as lunge_router
from backend.app.api.vertical_jump_comparison_api import (
    router as vertical_jump_comparison_router,
)
from backend.app.api.pushup_comparison_api import router as pushup_comparison_router

# Existing Squat Comparison
from backend.app.api.comparison_api import router as comparison_router

# New Sprint Comparison
from backend.app.api.sprint_comparison_api import router as sprint_comparison_router
from backend.app.api.lunge_comparison_api import (
    router as lunge_comparison_router
)


app = FastAPI(
    title="AI Sports Talent Assessment Platform"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/processed",
    StaticFiles(directory="data/processed"),
    name="processed"
)


# Existing API routers
app.include_router(squat_router)
app.include_router(vertical_jump_router)
app.include_router(sprint_router)
app.include_router(history_router)
app.include_router(pushup_router)
app.include_router(lunge_router)

# Comparison routers
app.include_router(comparison_router)
app.include_router(sprint_comparison_router)
app.include_router(vertical_jump_comparison_router)
app.include_router(pushup_comparison_router)
app.include_router(lunge_comparison_router)


@app.get("/")
def home():
    return {
        "message": "AI Sports Talent Assessment Platform is running!"
    }
