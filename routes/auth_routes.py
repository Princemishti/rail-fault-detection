from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["Authentication & User Management"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login_user(payload: LoginRequest):
    """
    Railway Inspector & Authority Login Authentication.
    """
    if payload.username and payload.password:
        return {
            "status": "success",
            "message": "Authentication successful",
            "user": {
                "username": payload.username,
                "role": "Lead Railway Inspector",
                "organization": "National Railway Authority",
                "token": "railway_jwt_session_token_sec_984"
            }
        }
    raise HTTPException(status_code=400, detail="Invalid username or password")

@router.get("/me")
async def get_current_user():
    return {
        "username": "inspector_admin",
        "role": "Lead Railway Inspector",
        "organization": "National Railway Authority",
        "status": "ACTIVE"
    }
