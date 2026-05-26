from dataclasses import dataclass


@dataclass(frozen=True)
class UserCreationInput:
    email: str
    display_name: str | None = None
