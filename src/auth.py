"""
Lightweight email/password authentication for InterviewTwin.
Stores users in a local JSON file with salted, hashed passwords.
Suitable for a student project demo - not production-grade auth.
"""

import json
import os
import hashlib
import secrets

USERS_FILE = "data/users.json"


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def sign_up(email: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if email in users:
        return False, "An account with this email already exists. Please sign in instead."

    salt = secrets.token_hex(16)
    users[email] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
    }
    _save_users(users)
    return True, "Account created successfully!"


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    email = email.strip().lower()
    users = _load_users()

    if email not in users:
        return False, "No account found with this email. Please sign up first."

    user = users[email]
    if _hash_password(password, user["salt"]) != user["password_hash"]:
        return False, "Incorrect password."

    return True, "Signed in successfully!"
