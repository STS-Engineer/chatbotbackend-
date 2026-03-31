from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.api.routes import router

app = FastAPI(title=settings.APP_NAME)

origins = [
    "https://avo-kms-rag-knowledge.azurewebsites.net"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://avo-kms-rag-knowledge.azurewebsites.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all so unhandled errors still return a proper JSON response.
    Without this, the connection drops before CORS headers are written and
    the browser reports a CORS error instead of a 500."""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )


prefix = settings.API_V1_PREFIX.strip()
if not prefix.startswith("/"):
    prefix = "/" + prefix

print("API_V1_PREFIX loaded as:", repr(settings.API_V1_PREFIX))
print("API_V1_PREFIX normalized:", repr(prefix))

app.include_router(router, prefix=prefix)
