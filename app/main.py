from fastapi import FastAPI
import uvicorn
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from rout import login_route, chatbot_rout, logout_route
from starlette.middleware.cors import CORSMiddleware
app = FastAPI()
origins = ["*"]
app.add_middleware(

    CORSMiddleware,
    allow_origins=origins,  # Allows only the specified origins
   
    allow_credentials=True,

    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE)

    allow_headers=["*"],  # Allows all headers
)
 
# Secure Session Middleware
app.add_middleware(
    SessionMiddleware,
    secret_key="supersecretkey",  # Keep this secret
    session_cookie="session_id",  # The session ID remains the same
    max_age=86400,  # Session persists for 1 day
    same_site="lax",  
    https_only=False,  
)
# /datadisk0/chatbot/aman/ChatBot

import os

STATIC_DIR = os.path.join(os.path.dirname(__file__), "rout/static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(login_route.route)
app.include_router(chatbot_rout.route)
app.include_router(logout_route.route)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8112, reload=True)
