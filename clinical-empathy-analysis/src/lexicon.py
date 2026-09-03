"""A weighted empathy lexicon, grounded in a coding framework.

The construct is not "nice words". Clinical empathy research measures specific
behaviours, and this lexicon is organised by the behaviour each cue signals,
following the structure of established consultation-coding schemes (Empathic
Communication Coding System; the Roter Interaction Analysis System's
socio-emotional categories).

Weights are the analyst's judgement about how strongly a cue indicates the
behaviour. They are not learned, and they are not validated against clinician
ratings -- both of which are stated on the site rather than implied away.

Bilingual because the source corpus this replaces was Mandarin consultation
transcripts, and empathy cues do not survive translation intact.
"""

from __future__ import annotations

# category -> {cue: weight}
LEXICON: dict = {
    # Naming or reflecting the patient's emotion back to them.
    "acknowledgement": {
        "i understand": 1.0, "i can see": 0.9, "that sounds": 1.0,
        "it makes sense": 1.0, "i hear you": 1.0, "must be": 0.8,
        "understandably": 0.9, "of course": 0.4,
        "我理解": 1.0, "我明白": 1.0, "能理解": 1.0, "听起来": 0.9,
        "确实": 0.5, "难怪": 0.8,
    },
    # Explicit emotional support.
    "validation": {
        "difficult": 0.7, "hard for you": 1.0, "frustrating": 0.9,
        "worrying": 0.9, "frightening": 1.0, "not your fault": 1.0,
        "many people": 0.6, "perfectly normal": 0.9,
        "不容易": 1.0, "辛苦": 0.9, "担心": 0.7, "害怕": 0.8,
        "不是你的错": 1.0, "很正常": 0.8,
    },
    # Inviting the patient to say more; the strongest behavioural signal.
    "exploration": {
        "tell me more": 1.0, "how are you feeling": 1.0, "what worries": 1.0,
        "can you describe": 0.9, "what matters most": 1.0, "anything else": 0.7,
        "how has that": 0.9,
        "再说说": 1.0, "感觉怎么样": 1.0, "还有什么": 0.8,
        "担心什么": 1.0, "能描述": 0.9,
    },
    # Checking the patient followed; shared decision-making.
    "partnership": {
        "we can": 0.8, "let us": 0.7, "together": 0.8, "what do you think": 1.0,
        "your choice": 1.0, "does that make sense": 0.9, "shall we": 0.8,
        "我们一起": 1.0, "你觉得呢": 1.0, "你来决定": 1.0, "明白吗": 0.6,
    },
    # Warmth markers that are weak on their own but real in aggregate.
    "warmth": {
        "thank you for": 0.6, "please": 0.3, "take your time": 0.9,
        "no rush": 0.8, "sorry to hear": 1.0,
        "谢谢": 0.5, "别着急": 0.9, "慢慢来": 0.9, "很遗憾": 1.0,
    },
    # Negative markers. Empathy is not only presence of warmth; dismissal is
    # its own signal and a lexicon without it scores curt consultations as
    # merely neutral.
    "dismissal": {
        "calm down": -1.0, "nothing to worry": -0.7, "just": -0.2,
        "obviously": -0.6, "as i said": -0.7, "you should have": -1.0,
        "that is not": -0.5, "no time": -0.9,
        "别紧张": -0.6, "没什么好担心": -0.8, "我说过了": -0.9,
        "你应该": -0.6, "没时间": -1.0,
    },
}

CATEGORIES = tuple(LEXICON.keys())
POSITIVE_CATEGORIES = tuple(c for c in CATEGORIES if c != "dismissal")


def all_cues() -> dict:
    """Flat cue -> (category, weight) map."""
    out = {}
    for cat, cues in LEXICON.items():
        for cue, w in cues.items():
            out[cue] = (cat, w)
    return out


def validate() -> list:
    """Structural problems a lexicon acquires as it is edited."""
    problems = []
    seen: dict = {}
    for cat, cues in LEXICON.items():
        for cue, w in cues.items():
            if cue in seen:
                problems.append(f"'{cue}' appears in {seen[cue]} and {cat}")
            seen[cue] = cat
            if cat == "dismissal" and w >= 0:
                problems.append(f"dismissal cue '{cue}' has non-negative weight {w}")
            if cat != "dismissal" and w <= 0:
                problems.append(f"{cat} cue '{cue}' has non-positive weight {w}")
            if abs(w) > 1.0:
                problems.append(f"'{cue}' weight {w} outside [-1, 1]")
    return problems
