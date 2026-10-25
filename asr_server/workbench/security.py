from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import OrderedDict
from urllib.parse import urlsplit

from fastapi import Request

from asr_server.errors import AsrError
from asr_server.workbench.executor import BoundedExecutor


COOKIE_NAME = "asr_workbench_session"
password_executor = BoundedExecutor(2, 8, "asr-password")
_password_slots = threading.BoundedSemaphore(2)


class LoginFailures:
    """Event-loop-owned failure table with bounded addresses and a one-minute TTL."""
    def __init__(self, capacity: int = 4096, ttl: float = 60) -> None:
        self.capacity = capacity
        self.ttl = ttl
        self.addresses: OrderedDict[str, list[float]] = OrderedDict()

    def recent(self, address: str) -> list[float]:
        cutoff = time.monotonic() - self.ttl
        while self.addresses and next(iter(self.addresses.values()))[-1] < cutoff:
            self.addresses.popitem(last=False)
        return [stamp for stamp in self.addresses.get(address, []) if stamp >= cutoff]

    def failed(self, address: str) -> None:
        self.addresses[address] = [*self.recent(address), time.monotonic()][-10:]
        self.addresses.move_to_end(address)
        while len(self.addresses) > self.capacity:
            self.addresses.popitem(last=False)

    def clear(self, address: str) -> None:
        self.addresses.pop(address, None)


def password_hash(password: str) -> str:
    if len(password) < 8 or len(password) > 1024:
        raise AsrError(400, "bad_request", "password must contain 8 to 1024 characters")
    salt = secrets.token_hex(16)
    with _password_slots:
        key = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
    return f"scrypt${salt}${key.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, salt, expected = stored.split("$")
        if algorithm != "scrypt" or len(password) > 1024:
            return False
        with _password_slots:
            actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1)
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def check_same_origin(request: Request) -> None:
    """Reject browser cross-site mutations, including login CSRF."""
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise AsrError(403, "cross_origin_request", "cross-site requests are not allowed")
    origin = request.headers.get("origin")
    if origin is not None:
        parsed = urlsplit(origin)
        actual = urlsplit(str(request.url))
        if (parsed.scheme, parsed.hostname, parsed.port) != (actual.scheme, actual.hostname, actual.port):
            raise AsrError(403, "cross_origin_request", "request origin does not match this server")
