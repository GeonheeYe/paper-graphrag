import uuid

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}" # 12자리의 랜덤한 문자열을 생성함.