"""
auth.py — registration and login logic.

Passwords are hashed with bcrypt before ever touching the database.
"""

import bcrypt
import db


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if db.get_user_by_username(username):
        return False, "That username is already taken."

    password_hash = hash_password(password)
    db.create_user(username, password_hash)
    return True, "Account created. You can log in now."


def login_user(username: str, password: str) -> int | None:
    user = db.get_user_by_username(username.strip())
    if user and verify_password(password, user["password_hash"]):
        return user["id"]
    return None
