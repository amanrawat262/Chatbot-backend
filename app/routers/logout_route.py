from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()

    response = RedirectResponse(url="/auth/", status_code=302)
    response.delete_cookie("sessid")
    response.delete_cookie("access_token")

    return response