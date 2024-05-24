from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.route_handler import RouteHandler


router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/optimize-route/")
async def optimize_route_handler(
    addresses: list[str] = Form(...),
    optimization_method: str = Form(...),
    transportation_mode: str = Form(...),
):

    route_handler = RouteHandler()

    return HTMLResponse(
        await route_handler.process_optimize_route(
            addresses, transportation_mode, method=optimization_method
        )
    )
