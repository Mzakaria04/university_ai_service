from dataclasses import dataclass
from enum import Enum

class UserRole(str, Enum):
    STUDENT = "STUDENT"
    INSTRUCTOR = "INSTRUCTOR"
    ADMIN = "ADMIN"

@dataclass(frozen=True)
class UserContext:
    user_id: str        # Maps to User.id (text UUID)
    university_id: str  # Maps to User.universityId (text, e.g. "20260001")
    full_name: str      # Maps to User.fullName (text)
    role: UserRole      # Maps to User.role (enum)
