"""
Enkel rate-limiting i minnet (glidende vindu).

Holder for MVP med én maskin. Ved flere Fly-maskiner er telleren per maskin -
den reelle grensen blir da N x limit. Naar §3.1 (soek etter telefon/bruker-ID
med karantene) skal implementeres, maa telleren flyttes til databasen eller
Redis; se backend_spec §3.1.
"""
import time
from collections import defaultdict, deque

from fastapi import Request


class SlidingWindow:
    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def client_ip(request: Request) -> str:
    """Fly setter Fly-Client-IP; ellers foerste ledd i X-Forwarded-For."""
    for header in ("fly-client-ip", "x-forwarded-for"):
        value = request.headers.get(header)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "ukjent"
