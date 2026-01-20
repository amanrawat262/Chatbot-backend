from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from .. import schemas
from ..database import get_db
from ..dependencies import get_current_user, oauth2_scheme
from ..utils import decode_access_token
from ..chatbot import handle_user_input   

import logging

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(
    user_query: schemas.UserQuery,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(401, "Invalid/expired token")

    username = payload.get("user")
    session_id = payload.get("session_id")

    if not username or not session_id:
        raise HTTPException(401, "Token missing required fields")

    question = user_query.question.strip()
    if not question:
        raise HTTPException(400, "Question is required")

    try:
        response_data = handle_user_input(question, session_id)

        graph_url = ""
        response_text = ""

        if isinstance(response_data, dict):
            if "graph" in response_data:
                graph_url = response_data["graph"]  # or format path
            if "response" in response_data:
                response_text = response_data["response"]
        elif isinstance(response_data, str):
            response_text = response_data
        else:
            raise ValueError("Unexpected response format from handle_user_input")

        # Save to database
        entry_time = datetime.now(timezone.utc)
        expiry = entry_time + relativedelta(months=2)

        # ← call your insert function here
        # process_user_query(username, session_id, question, response_text, graph_url, expiry, entry_time)

        result = {
            "response": response_text,
            "graph_url": graph_url if graph_url else None
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Error in /chat endpoint")
        raise HTTPException(500, f"Internal server error: {str(e)}")