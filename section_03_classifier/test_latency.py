import pytest
import time
from section_03_classifier.predict import TicketClassifier

@pytest.fixture(scope="module")
def classifier():
    return TicketClassifier()

TICKETS = [
    ("I cannot log in to my account", "technical_issue"),
    ("Please add a Dark Mode", "feature_request"),
    ("Why was I billed twice?", "billing"),
    ("The app is very slow today", "technical_issue"),
    ("I want to cancel my subscription", "billing"),
    ("Your support is terrible", "complaint"),
    ("How do I change my email?", "other"),
    ("Can we get a Slack integration?", "feature_request"),
    ("The screen goes black on startup", "technical_issue"),
    ("I need a refund for last month", "billing"),
    ("I am very unhappy with the new update", "complaint"),
    ("What is your contact number?", "other"),
    ("Is there a mobile version?", "other"),
    ("Please implement folder organization", "feature_request"),
    ("The reset link is not working", "technical_issue"),
    ("My payment was declined", "billing"),
    ("I've been waiting for an hour", "complaint"),
    ("Great job on the new design!", "other"),
    ("Can I pay with Bitcoin?", "billing"),
    ("The export feature is broken", "technical_issue")
]

@pytest.mark.parametrize("text, expected_label", TICKETS)
def test_prediction_and_latency(classifier, text, expected_label):
    """Test that prediction is valid and latency is under 500ms."""
    result = classifier.predict(text)
    
    # Assertions
    assert result["label"] in ["billing", "technical_issue", "feature_request", "complaint", "other"]
    assert result["latency_ms"] < 500
    
    print(f"\nPASS | Label: {result['label']:<15} | Time: {result['latency_ms']:>6.2f}ms | Text: {text[:50]}...")
