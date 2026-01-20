from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm

from .. import schemas
from ..database import get_db
from ..models import User
from ..auth import verify_password, create_access_token
from ..dependencies import get_db

import uuid
from datetime import timedelta

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/token")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    # Generate or reuse session_id
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id

    request.session["user"] = user.username
    request.session["authenticated"] = True

    access_token = create_access_token(
        data={"user": user.username, "session_id": session_id},
        expires_delta=timedelta(days=7)
    )

    response = JSONResponse({
        "access_token": access_token,
        "token_type": "bearer",
        "session_id": session_id,
        "username": user.username
    })

    response.set_cookie(
        key="sessid",
        value=session_id,
        httponly=True,
        secure=False,          # True + HTTPS in production
        samesite="lax",
        max_age=86400 * 7
    )

    return response