"""Frozen, versioned teacher-candidate ranking rule shared by all shards."""

import math

WEIGHTS = dict(identity=0.25, pose=0.20, expression=0.20, va=0.10, musiq=0.20, valid=0.05)
SCORING_VERSION = "teacher_composite_v1"


def composite_score(record):
    if not record.get("face_valid"):
        return 0.0
    def clip(value):
        return min(1.0, max(0.0, value))
    values = [record[k] for k in ("identity_cosine", "pose_mae_deg", "hsemotion_cosine",
                                  "hsemotion_va_mae", "musiq_generated")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite candidate metric")
    terms = dict(identity=clip((.35-values[0])/.35), pose=math.exp(-values[1]/15),
                 expression=clip((values[2]-.2)/.8), va=math.exp(-values[3]/.5),
                 musiq=clip((values[4]-20)/60), valid=1.)
    return sum(WEIGHTS[name]*value for name, value in terms.items())


def best_candidate(records):
    valid = [row for row in records if row.get("face_valid")]
    return max(valid, key=lambda row: (row["quality_score"], -row["round"])) if valid else None
