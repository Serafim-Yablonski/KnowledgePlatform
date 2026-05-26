from dataclasses import dataclass


@dataclass(frozen=True)
class UserCreationInput:
    email: str
    display_name: str | None = None


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
