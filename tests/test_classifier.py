import pytest
from app.classifier import intent_classifier, QueryIntent

def test_bi_temporal_change_intent():
    query = "What changed between these two images?"
    result = intent_classifier.classify(query)
    
    assert result.intent == QueryIntent.BI_TEMPORAL_CHANGE
    assert result.confidence == 0.96

def test_vqa_intent():
    query = "What is the building count in this satellite view?"
    result = intent_classifier.classify(query)
    
    assert result.intent == QueryIntent.VQA
    assert result.confidence >= 0.80

def test_grounding_intent():
    query = "Detect and locate all cargo ships in the harbor"
    result = intent_classifier.classify(query)
    
    assert result.intent == QueryIntent.GROUNDING
    assert result.confidence >= 0.85

def test_optical_sar_intent():
    query = "Process synthetic aperture radar SAR image for cloud penetration"
    result = intent_classifier.classify(query)
    
    assert result.intent == QueryIntent.OPTICAL_SAR
    assert result.confidence >= 0.85

def test_captioning_intent():
    query = "Describe the scene in detail"
    result = intent_classifier.classify(query)
    
    assert result.intent == QueryIntent.CAPTIONING
    assert result.confidence >= 0.85
