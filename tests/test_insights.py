"""
Unit tests for Biological Insight Engine (backend/insights.py).
Verifies confidence stratification, multi-modal conflict handling,
copy guidelines, and failure fallbacks.
"""
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.insights import get_insight, INSIGHTS, ANTHROPOMORPHISM_WARNING


class TestBiologicalInsightEngine:

    def test_all_canonical_classes_exist(self):
        canonical_classes = ["Angry", "Fearful", "Happy", "Sad"]
        for cls in canonical_classes:
            assert cls in INSIGHTS
            data = INSIGHTS[cls]
            assert "biology" in data and len(data["biology"]) > 30
            assert "tips" in data and len(data["tips"]) >= 2
            assert "disclaimer" in data and len(data["disclaimer"]) > 20

    def test_copy_rules_no_false_feature_detection_claims(self):
        forbidden_phrases = [
            "detected your cat's",
            "detected ears",
            "measured whisker",
            "measured pupil",
            "facial recognition found",
            "we scanned the pupil",
            "we detected flattened ears"
        ]
        for cls, data in INSIGHTS.items():
            full_text = (data["biology"] + " " + " ".join(data["tips"]) + " " + data["disclaimer"]).lower()
            for phrase in forbidden_phrases:
                assert phrase not in full_text, f"Found forbidden phrase '{phrase}' in class {cls}"

    def test_high_confidence_stratification(self):
        insight = get_insight("Happy", confidence_pct=92.5, probabilities=[0.925, 0.03, 0.025, 0.02], modalities_agree=True)
        assert insight["level"] == "high"
        assert insight["css_class"] == "insight-high"
        assert insight["badge_class"] == "badge-success"
        assert insight["conflict_note"] is None
        assert insight["emotion"] == "Happy"
        assert insight["confidence_pct"] == 92.5

    def test_moderate_confidence_stratification(self):
        insight = get_insight("Angry", confidence_pct=72.0, probabilities=[0.72, 0.15, 0.08, 0.05], modalities_agree=True)
        assert insight["level"] == "moderate"
        assert insight["css_class"] == "insight-moderate"
        assert insight["badge_class"] == "badge-info"
        assert insight["conflict_note"] is None

    def test_low_confidence_becomes_conflicted(self):
        insight = get_insight("Sad", confidence_pct=48.0, probabilities=[0.48, 0.25, 0.15, 0.12], modalities_agree=True)
        assert insight["level"] == "conflicted"
        assert insight["css_class"] == "insight-conflicted"
        assert insight["conflict_note"] is not None
        assert "anthropomorphism" in insight["conflict_note"].lower()

    def test_modalities_disagree_triggers_conflicted(self):
        # Even with high confidence, if modalities disagree, it must be flagged conflicted
        insight = get_insight("Fearful", confidence_pct=88.0, probabilities=[0.88, 0.05, 0.04, 0.03], modalities_agree=False)
        assert insight["level"] == "conflicted"
        assert insight["css_class"] == "insight-conflicted"
        assert insight["conflict_note"] is not None

    def test_narrow_probability_margin_triggers_conflicted(self):
        # Top-2 difference < 0.15 (e.g. 0.45 vs 0.38)
        probs = [0.45, 0.38, 0.10, 0.07]
        insight = get_insight("Angry", confidence_pct=82.0, probabilities=probs, modalities_agree=True)
        assert insight["level"] == "conflicted"

    def test_normalizes_0_to_1_float_input(self):
        insight = get_insight("Happy", confidence_pct=0.885)
        assert insight["confidence_pct"] == 88.5
        assert insight["level"] == "high"

    def test_fallback_for_unknown_emotion(self):
        insight = get_insight("Surprised", confidence_pct=75.0)
        assert insight["level"] == "moderate"
        assert "biology" in insight
