# =============================================================================
# data_generation/session_config.py
# Session variation parameters for synthetic data generation.
# Controls emotional states, disclosure stages, and session distribution
# to ensure diverse, balanced training data for model fine-tuning.
# =============================================================================

import random

# =============================================================================
# EMOTIONAL STATES
# Weights sum to 1.0. Distribution designed to produce:
#   ~25% crisis, ~30% anxious, ~20% stable, ~15% hopeful, ~10% regressing
# across 1000 sessions — giving the model exposure to the full risk spectrum.
# =============================================================================

EMOTIONAL_STATES = {
    "crisis": {
        "weight": 0.25,
        "label_ar": "أزمة حادة",
        "risk_band_hint": "high_to_critical",
        "instruction_ar": (
            "أنتِ في حالة أزمة حادة اليوم. حدث شيء مؤخراً — حادثة، تهديد، يوم لا تحتملينه. "
            "أنتِ في ذروة الضيق: مرعوبة، أو محطمة، أو غير قادرة على التفكير بوضوح. "
            "قد تكون رسائلكِ غير منظمة أو عاجلة أو مقطوعة."
        ),
    },
    "anxious": {
        "weight": 0.30,
        "label_ar": "قلق وتوتر",
        "risk_band_hint": "medium",
        "instruction_ar": (
            "أنتِ قلقة جداً اليوم — قلقة على المستقبل، أو على الأطفال، أو بسبب إجراءات قانونية، "
            "أو من ما قد يحدث لاحقاً. لستِ في خطر فوري، لكنكِ مضطربة وخائفة ولا تستطيعين الراحة."
        ),
    },
    "stable": {
        "weight": 0.20,
        "label_ar": "نسبياً هادئة",
        "risk_band_hint": "low_to_medium",
        "instruction_ar": (
            "أنتِ تمرين بيوم هادئ نسبياً. لا تعانين من أزمة، لكن لا يزال لديكِ ألم وتحديات "
            "تريدين التحدث عنها. يمكنكِ التفكير بوضوح أكثر من المعتاد، وأنتِ منفتحة على النقاش."
        ),
    },
    "hopeful": {
        "weight": 0.15,
        "label_ar": "بصيص أمل",
        "risk_band_hint": "low",
        "instruction_ar": (
            "تشعرين اليوم ببصيص من الأمل — ربما حدث شيء صغير إيجابي، أو أنكِ أحسستِ بلحظة قوة. "
            "لا تزالين تعانين وتواجهين صعوبات، لكن هناك لحظات ضوء يمكنكِ التحدث عنها."
        ),
    },
    "regressing": {
        "weight": 0.10,
        "label_ar": "انتكاسة مؤقتة",
        "risk_band_hint": "medium_to_high",
        "instruction_ar": (
            "أنتِ تمرين بانتكاسة اليوم — شيء ما أعاد تشغيل الذكريات المؤلمة: ذكرى قديمة، "
            "اتصال من الشخص المسيء، أو موقف صعب مع الأطفال. تشعرين أنكِ رجعتِ للوراء رغم كل شيء."
        ),
    },
}

# =============================================================================
# DISCLOSURE STAGES
# How much the user trusts the service and how openly they share.
# Weights: 30% early (guarded) / 45% mid (opening up) / 25% late (open).
# =============================================================================

DISCLOSURE_STAGES = {
    "early": {
        "weight": 0.30,
        "label_ar": "تواصل مبكر — حذرة",
        "instruction_ar": (
            "هذه من أولى مراتك في التواصل مع هذه الخدمة. أنتِ حذرة وغير متأكدة من مقدار ما "
            "ستشاركين، وتختبرين ما إذا كان يمكنكِ الوثوق بهذا الشخص. تشاركين المعلومات ببطء "
            "وبتردد، وأحياناً تتراجعين عما قلتِه."
        ),
    },
    "mid": {
        "weight": 0.45,
        "label_ar": "منتصف العلاقة — تنفتحين تدريجياً",
        "instruction_ar": (
            "تواصلتِ مع هذه الخدمة عدة مرات من قبل وبدأتِ تثقين بها نوعاً ما. أنتِ أكثر "
            "انفتاحاً من المرة الأولى، وتشاركين تفاصيل أعمق — لكنكِ لا تزالين تحتفظين ببعض "
            "الأشياء لنفسكِ ولم تقوليها بعد."
        ),
    },
    "late": {
        "weight": 0.25,
        "label_ar": "علاقة متقدمة — ثقة عالية",
        "instruction_ar": (
            "تتواصلين مع هذه الخدمة منذ فترة وتثقين بها ثقة كبيرة. تتحدثين بانفتاح تام، "
            "وتعملين بشكل تعاوني على الحلول والتعافي. أنتِ لا تترددين في البوح بمشاعرك الحقيقية."
        ),
    },
}

# =============================================================================
# SESSION LENGTH DISTRIBUTION
# Uniform random between 5 and 20 user turns per session.
# (Each user turn is paired with one assistant turn → 10–40 total turns)
# =============================================================================

SESSION_LENGTH_MIN = 5
SESSION_LENGTH_MAX = 20

# =============================================================================
# DISTRIBUTION TARGETS (for 1000 sessions, 100 per persona)
# These are informational — actual selection uses weighted random choices.
# =============================================================================

DISTRIBUTION_TARGETS = {
    "sessions_total": 1000,
    "sessions_per_persona": 100,
    "num_personas": 10,
    "emotional_state_approximate_counts": {
        "crisis":     250,   # 25% — High/critical risk sessions
        "anxious":    300,   # 30% — Medium risk sessions
        "stable":     200,   # 20% — Low-medium risk sessions
        "hopeful":    150,   # 15% — Low risk sessions
        "regressing": 100,   # 10% — Variable risk sessions
    },
    "disclosure_stage_approximate_counts": {
        "early": 300,        # 30%
        "mid":   450,        # 45%
        "late":  250,        # 25%
    },
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def weighted_choice(options_dict: dict) -> str:
    """Select a random key weighted by the 'weight' values in the dict."""
    keys = list(options_dict.keys())
    weights = [options_dict[k]["weight"] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def random_session_length() -> int:
    """Return a random session turn count between SESSION_LENGTH_MIN and MAX."""
    return random.randint(SESSION_LENGTH_MIN, SESSION_LENGTH_MAX)


def build_session_plan(personas: dict, sessions_per_persona: int = 100) -> list:
    """
    Build a shuffled list of session configurations.
    Returns a list of dicts, each describing one session to generate.
    """
    plan = []
    for persona in personas.values():
        for _ in range(sessions_per_persona):
            plan.append({
                "persona_id": persona["id"],
                "emotional_state": weighted_choice(EMOTIONAL_STATES),
                "disclosure_stage": weighted_choice(DISCLOSURE_STAGES),
                "num_turns": random_session_length(),
            })
    random.shuffle(plan)
    return plan
