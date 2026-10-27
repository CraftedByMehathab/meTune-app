"""Single evaluator account. All auth logic lives here; the client only renders login + redirects on 401."""
import hmac
import time
from collections import defaultdict, deque

from fastapi import APIRouter, Depends, HTTPException, Request
from pwdlib import PasswordHash
from pydantic import BaseModel

from .settings import Settings, get_settings

hasher = PasswordHash.recommended()  # argon2
router = APIRouter(prefix="/api/auth", tags=["auth"])

_attempts: dict[str, deque[float]] = defaultdict(deque)
MAX_ATTEMPTS, WINDOW_S = 5, 60


class LoginIn(BaseModel):
    username: str
    password: str


class Me(BaseModel):
    username: str


def _rate_limit(ip: str) -> None:
    q, t = _attempts[ip], time.monotonic()
    while q and t - q[0] > WINDOW_S:
        q.popleft()
    if len(q) >= MAX_ATTEMPTS:
        raise HTTPException(429, "Too many login attempts, try again in a minute")
    q.append(t)


@router.post("/login", response_model=Me)
def login(body: LoginIn, request: Request, settings: Settings = Depends(get_settings)) -> Me:
    _rate_limit(request.client.host if request.client else "unknown")
    if not settings.admin_password_hash:
        raise HTTPException(503, "Login not configured (ADMIN_PASSWORD_HASH missing)")
    user_ok = hmac.compare_digest(body.username, settings.admin_user)
    pass_ok = hasher.verify(body.password, settings.admin_password_hash)
    if not (user_ok and pass_ok):
        raise HTTPException(401, "Invalid username or password")
    request.session.clear()
    request.session["user"] = body.username
    return Me(username=body.username)


@router.post("/logout", status_code=204)
def logout(request: Request) -> None:
    request.session.clear()


def require_user(request: Request) -> str:
    if user := request.session.get("user"):
        return user
    raise HTTPException(401, "Not authenticated")


@router.get("/me", response_model=Me)
def me(user: str = Depends(require_user)) -> Me:
    return Me(username=user)
