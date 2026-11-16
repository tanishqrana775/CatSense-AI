"""
Biological Insight Engine for Feline Emotion Recognition.
Provides ethologically grounded biological insights, environmental guidance,
and confidence-calibrated assessments without asserting biological feature detection.
"""
from typing import Dict, Any, Optional, List, Union

INSIGHTS: Dict[str, Dict[str, Any]] = {
    "Angry": {
        "biology": (
            "In feline ethology, signs of aggressive arousal often correlate with ears rotated backwards or flattened, "
            "pupil shifts reflecting autonomic nervous system arousal, and localized facial tension. "
            "Acoustic markers such as low growls, snarls, or sudden hissing serve as evolutionary distance-increasing signals."
        ),
        "tips": [
            "Provide immediate physical space; avoid prolonged direct eye contact, looming postures, or sudden movements.",
            "Eliminate perceived environmental triggers and safely separate other household pets until baseline calm returns."
        ],
        "disclaimer": (
            "This assessment reflects learned statistical patterns from audio-visual datasets and does not replace "
            "professional veterinary or certified feline behavioral consultation."
        )
    },
    "Fearful": {
        "biology": (
            "Feline defensive threat responses frequently manifest as dilated pupils, ears retracted tightly against the skull, "
            "and crouched body silhouettes designed to minimize exposure. Vocal signatures may include plaintive distress calls, "
            "high-pitched yowls, or defensive spitting when retreat options feel constrained."
        ),
        "tips": [
            "Ensure unimpeded access to elevated perches or enclosed hiding sanctuaries without attempting to corner or restrain the cat.",
            "Speak softly in gentle, unhurried cadences and allow the cat to initiate any subsequent physical interaction at its own pace."
        ],
        "disclaimer": (
            "This assessment reflects learned statistical patterns from audio-visual datasets and does not replace "
            "professional veterinary or certified feline behavioral consultation."
        )
    },
    "Happy": {
        "biology": (
            "Contentment and relaxed affect in domestic felines are typically accompanied by upright, forward-facing pinnae, "
            "soft unconstricted gaze with occasional slow-blinking, and relaxed whisker spreads. Acoustic expressions such as "
            "rhythmic low-frequency purring or chirping trills commonly indicate positive social engagement."
        ),
        "tips": [
            "Reinforce positive bonding through gentle interactive play, interactive grooming, or affectionate slow-blinks.",
            "Offer affection targeting preferred feline scent gland areas, notably the temporal regions, cheeks, and under the chin."
        ],
        "disclaimer": (
            "This assessment reflects learned statistical patterns from audio-visual datasets and does not replace "
            "professional veterinary or certified feline behavioral consultation."
        )
    },
    "Sad": {
        "biology": (
            "While feline affect differs from human depressive states, reduced social engagement, slumped posture, and subdued facial "
            "tonus often indicate physiological discomfort, chronic environmental stress, or illness. Vocal output may become "
            "uncharacteristically muted or express prolonged plaintive meows."
        ),
        "tips": [
            "Closely monitor routine indicators including appetite, hydration, litter box frequency, and grooming consistency.",
            "Schedule a clinical evaluation with a veterinarian if withdrawal, lethargy, or altered eating habits persist beyond 24 hours."
        ],
        "disclaimer": (
            "This assessment reflects learned statistical patterns from audio-visual datasets and does not replace "
            "professional veterinary or certified feline behavioral consultation."
        )
    }
}

ANTHROPOMORPHISM_WARNING = (
    "Notice: Ambiguous behavioral signals or lower model certainty detected. Projecting human emotional interpretations "
    "(anthropomorphism) onto felines can inadvertently overlook underlying clinical discomfort or subtle environmental distress. "
    "Evaluate the cat's full body posture, appetite, and environmental context holistically."
)


def get_insight(
    emotion: str,
    confidence_pct: float,
    probabilities: Optional[Union[List[float], Dict[str, float]]] = None,
    modalities_agree: bool = True
) -> Dict[str, Any]:
    """
    Computes confidence-calibrated biological insight, returning level, badges,
    ethological biological context, actionable tips, and disclaimers.

    Parameters:
        emotion: Canonical emotion class ('Angry', 'Fearful', 'Happy', 'Sad').
        confidence_pct: Confidence as a float percentage (0.0 to 100.0).
        probabilities: Probability vector or dictionary of class probabilities.
        modalities_agree: Whether multi-modal predictions concurred.

    Returns:
        Dictionary with level ('high', 'moderate', 'conflicted'), css classes,
        biological context, tips, and disclaimer.
    """
    # Normalize confidence to 0-100 range if 0-1 float passed
    conf_norm = float(confidence_pct)
    if 0.0 < conf_norm <= 1.0:
        conf_norm *= 100.0

    # Ensure emotion is recognized
    canonical_emotion = emotion.capitalize() if emotion else "Happy"
    insight_data = INSIGHTS.get(canonical_emotion, INSIGHTS["Happy"])

    # Determine probability margin if provided
    prob_values: List[float] = []
    if isinstance(probabilities, dict):
        prob_values = sorted(probabilities.values(), reverse=True)
    elif isinstance(probabilities, (list, tuple)):
        prob_values = sorted([float(p) for p in probabilities], reverse=True)

    narrow_margin = False
    if len(prob_values) >= 2:
        top_diff = prob_values[0] - prob_values[1]
        # Top-2 probability gap < 0.15 indicates ambiguous classifier decision
        if top_diff < 0.15:
            narrow_margin = True

    # Assess insight level
    if not modalities_agree or narrow_margin or conf_norm < 60.0:
        level = "conflicted"
        css_class = "insight-conflicted"
        badge_class = "badge-warning"
        badge_label = "Ambiguous / Conflicted Signals"
        conflict_note = ANTHROPOMORPHISM_WARNING
    elif conf_norm > 80.0:
        level = "high"
        css_class = "insight-high"
        badge_class = "badge-success"
        badge_label = "High Confidence Assessment"
        conflict_note = None
    else:
        level = "moderate"
        css_class = "insight-moderate"
        badge_class = "badge-info"
        badge_label = "Moderate Confidence Assessment"
        conflict_note = None

    return {
        "emotion": canonical_emotion,
        "confidence_pct": round(conf_norm, 1),
        "level": level,
        "css_class": css_class,
        "badge_class": badge_class,
        "badge_label": badge_label,
        "biology": insight_data["biology"],
        "tips": insight_data["tips"],
        "disclaimer": insight_data["disclaimer"],
        "conflict_note": conflict_note,
        "modalities_agree": modalities_agree
    }
