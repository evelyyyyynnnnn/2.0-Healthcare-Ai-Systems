"""Authored consultation transcripts with clinician-style empathy ratings.

The project this replaces trained on data synthesised at runtime and evaluated
on it -- which measures whether a model can learn a generator, not whether it
measures empathy. These transcripts are written by hand, each carrying a rating
on a 0-4 scale in the spirit of the CARE measure, and they are held out
properly.

They are still authored, not collected. Sixteen transcripts written by one
person is not a corpus, and the site says so. What they buy is an honest
evaluation protocol that a real corpus can be dropped into unchanged.
"""

from __future__ import annotations

from .scoring import Transcript, Turn


def _t(tid, lang, rating, pairs, note=""):
    return Transcript(tid=tid, language=lang, rating=rating, note=note,
                      turns=[Turn(s, x) for s, x in pairs])


CORPUS = [
    _t("EN-01", "en", 4.0, [
        ("patient", "I've been getting these headaches and I'm scared it's something serious."),
        ("doctor", "That sounds frightening. Tell me more about when they started."),
        ("patient", "About three weeks ago. They wake me at night."),
        ("doctor", "I can see why that would be worrying. What worries you most about them?"),
        ("patient", "My mother had a brain tumour."),
        ("doctor", "Thank you for telling me that. It makes sense that this is on your mind. "
                   "We can work through this together. Does that make sense so far?"),
    ], "High empathy: acknowledgement, exploration, partnership, no dismissal."),

    _t("EN-02", "en", 0.5, [
        ("patient", "I've been getting these headaches and I'm scared it's something serious."),
        ("doctor", "Nothing to worry about. Obviously it's just tension."),
        ("patient", "But they wake me at night."),
        ("doctor", "As I said, it's tension. Take paracetamol."),
        ("patient", "My mother had a brain tumour."),
        ("doctor", "Calm down. I have no time to go through family history now."),
    ], "Low empathy: dismissal cues throughout, no exploration."),

    _t("EN-03", "en", 2.5, [
        ("patient", "The new tablets make me feel sick."),
        ("doctor", "How are you feeling otherwise?"),
        ("patient", "Tired, mostly."),
        ("doctor", "That's a common side effect. We can try a lower dose. What do you think?"),
        ("patient", "I'd like that."),
    ], "Middling: some exploration and partnership, little emotional acknowledgement."),

    _t("EN-04", "en", 3.5, [
        ("patient", "I couldn't keep up with the exercises you gave me."),
        ("doctor", "That's alright, and it's not your fault. Many people find them hard "
                   "to fit in. Can you describe what got in the way?"),
        ("patient", "Work, mostly. I'm on my feet all day."),
        ("doctor", "That must be exhausting. Let us find something that fits your day "
                   "instead. Take your time thinking about what would work."),
    ], "High: validation plus exploration, non-judgemental."),

    _t("EN-05", "en", 1.0, [
        ("patient", "I couldn't keep up with the exercises."),
        ("doctor", "You should have made time. Just do them in the morning."),
        ("patient", "I'm on my feet all day at work."),
        ("doctor", "Everyone is busy. Obviously the exercises only work if you do them."),
    ], "Low: blame and dismissal."),

    _t("EN-06", "en", 3.0, [
        ("patient", "I'm not sure I want the surgery."),
        ("doctor", "I hear you. It's your choice, and we can take as long as you need. "
                   "What matters most to you here?"),
        ("patient", "Being able to look after my grandchildren."),
        ("doctor", "That helps me a lot. Let us talk through what each option means for that."),
    ], "High partnership and exploration; less explicit emotional labelling."),

    _t("EN-07", "en", 2.0, [
        ("patient", "The pain is back."),
        ("doctor", "Where exactly?"),
        ("patient", "Lower back, same as before."),
        ("doctor", "Right. We can try physiotherapy again. Anything else?"),
        ("patient", "No."),
    ], "Neutral-clinical: efficient, not warm, not dismissive."),

    _t("EN-08", "en", 4.0, [
        ("patient", "I've been feeling very low since my husband died."),
        ("doctor", "I'm so sorry to hear that. That must be incredibly hard for you."),
        ("patient", "Some days I can't get out of bed."),
        ("doctor", "That's understandably difficult, and it's perfectly normal to feel "
                   "this way. Tell me more about those days. What has helped, if anything?"),
        ("patient", "My sister visits."),
        ("doctor", "I'm glad you have her. We can look at what else might help, together."),
    ], "Highest: sustained acknowledgement, validation, exploration, partnership."),

    _t("ZH-01", "zh", 4.0, [
        ("patient", "我这几天头疼得厉害，很担心是不是什么大问题。"),
        ("doctor", "听起来确实让人害怕，我理解你的担心。能再说说是什么时候开始的吗？"),
        ("patient", "大概三个星期了，晚上会疼醒。"),
        ("doctor", "那一定很辛苦。你最担心什么呢？"),
        ("patient", "我母亲得过脑瘤。"),
        ("doctor", "谢谢你告诉我。有这样的经历，会这么想很正常。我们一起把它查清楚，好吗？"),
    ], "Mandarin high-empathy counterpart of EN-01."),

    _t("ZH-02", "zh", 0.5, [
        ("patient", "我这几天头疼得厉害，很担心。"),
        ("doctor", "没什么好担心的，就是紧张。"),
        ("patient", "可是晚上会疼醒。"),
        ("doctor", "我说过了，是紧张。吃点止疼药。"),
        ("patient", "我母亲得过脑瘤。"),
        ("doctor", "别紧张。我现在没时间问家族史。"),
    ], "Mandarin low-empathy counterpart of EN-02."),

    _t("ZH-03", "zh", 3.5, [
        ("patient", "新开的药让我很不舒服。"),
        ("doctor", "不容易，辛苦你了。感觉怎么样，具体是哪里不舒服？"),
        ("patient", "主要是累。"),
        ("doctor", "这个副作用确实常见，不是你的错。我们一起看看要不要减量，你觉得呢？"),
    ], "Mandarin: validation plus partnership."),

    _t("ZH-04", "zh", 2.0, [
        ("patient", "腰又疼了。"),
        ("doctor", "具体哪个位置？"),
        ("patient", "还是下腰，跟上次一样。"),
        ("doctor", "嗯。可以再做理疗。还有什么？"),
    ], "Mandarin neutral-clinical."),

    _t("ZH-05", "zh", 1.0, [
        ("patient", "锻炼我没能坚持下来。"),
        ("doctor", "你应该抽出时间。早上做就行了。"),
        ("patient", "我上班一整天都站着。"),
        ("doctor", "谁都忙。不做当然没效果。"),
    ], "Mandarin low: blame."),

    _t("ZH-06", "zh", 3.0, [
        ("patient", "我不太想做手术。"),
        ("doctor", "我明白。这个你来决定，我们慢慢来。对你来说最重要的是什么？"),
        ("patient", "能照顾孙子。"),
        ("doctor", "这很有帮助。我们一起看看每个方案对这件事的影响。"),
    ], "Mandarin partnership-led."),

    _t("EN-09", "en", 1.5, [
        ("patient", "I'm worried about the results."),
        ("doctor", "The results are fine. Nothing to worry about."),
        ("patient", "But the letter mentioned a follow-up."),
        ("doctor", "That's routine. Is that all?"),
    ], "Low-middle: dismissal without hostility."),

    _t("EN-10", "en", 3.0, [
        ("patient", "I'm worried about the results."),
        ("doctor", "Of course. Tell me what's on your mind and I'll go through them with you."),
        ("patient", "The letter mentioned a follow-up."),
        ("doctor", "I can see why that stood out. It is routine, and I understand it "
                   "doesn't feel routine when it's your letter. Shall we look at it together?"),
    ], "Same clinical content as EN-09, empathic delivery. Tests whether the "
       "measure tracks manner rather than topic."),
]


def ratings() -> list:
    return [t.rating for t in CORPUS]


def stats() -> dict:
    import statistics as st
    r = ratings()
    return {
        "n_transcripts": len(CORPUS),
        "n_english": sum(1 for t in CORPUS if t.language == "en"),
        "n_mandarin": sum(1 for t in CORPUS if t.language == "zh"),
        "n_turns": sum(len(t.turns) for t in CORPUS),
        "rating_mean": round(st.mean(r), 3),
        "rating_sd": round(st.pstdev(r), 3),
        "rating_min": min(r), "rating_max": max(r),
    }
