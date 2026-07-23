from datetime import datetime

def should_retrain(now_utc: datetime, newly_matured: int, cfg: dict) -> tuple[bool, str]:
    if now_utc.weekday() != int(cfg["weekday_utc"]):
        return False, "NOT_SCHEDULED_DAY"
    if newly_matured < int(cfg["minimum_new_matured_candidates"]):
        return False, "INSUFFICIENT_NEW_MATURED_CANDIDATES"
    return True, "ELIGIBLE"
