import time
import pytest
import os
from section_03_classifier.predict import TicketClassifier

MODEL_DIR = "section_03_classifier/model"

@pytest.fixture
def classifier():
    if not os.path.exists(MODEL_DIR):
        pytest.skip("Model not found. Run training first.")
    return TicketClassifier()

def test_prediction_latency(classifier):
    """Ensure that a single prediction takes less than 200ms."""
    text = "The server is down and I cannot access my dashboard."
    
    # Warm up
    classifier.predict(text)
    
    start_time = time.time()
    classifier.predict(text)
    end_time = time.time()
    
    latency = (end_time - start_time) * 1000 # in ms
    print(f"\nPrediction latency: {latency:.2f}ms")
    
    # 200ms is a reasonable threshold for DistilBERT on most CPUs for a single short string
    assert latency < 200, f"Latency {latency:.2f}ms exceeds threshold of 200ms"

def test_prediction_output_format(classifier):
    """Ensure the prediction output has correct keys."""
    text = "Help with billing"
    result = classifier.predict(text)
    assert "label" in result
    assert "confidence" in result
    assert isinstance(result["label"], str)
    assert 0 <= result["confidence"] <= 1
