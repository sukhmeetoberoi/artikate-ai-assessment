# Section 01: ML Model Diagnosis

## Scenario: Low Precision in Production
The customer service ticket classifier was performing well in offline evaluation (92% F1-score) but has dropped to 65% precision in production after 2 weeks.

## 1. Potential Causes
1. **Data Drift**: The distribution of incoming customer tickets has shifted (e.g., new product launch, seasonality).
2. **Concept Drift**: The meaning of labels has changed (e.g., "Technical Support" now includes billing-related tech issues).
3. **Training-Serving Skew**: Features are processed differently in the production pipeline compared to the training pipeline (e.g., different tokenization or text normalization).
4. **Data Leakage**: Training data included features not available at inference time (e.g., "ticket_resolved_by" field).

## 2. Diagnostic Steps
1. **Compare Distributions**: Plot the distribution of predicted labels in production vs. evaluation data.
2. **Error Analysis**: Manually inspect the false positives. Are they all from a specific new category?
3. **Feature Monitoring**: Use tools like Great Expectations or TFDV to check for null values or unexpected ranges in production features.
4. **Backtesting**: Run the production model on a fresh manually-labeled sample from the last 48 hours.

## 3. Recommended Fixes
- **Retraining**: Implement a periodic retraining pipeline with recent data.
- **Human-in-the-loop**: Route low-confidence predictions (e.g., < 0.7) to human agents.
- **Versioning**: Ensure strict parity between the preprocessing code used in training and the inference service.
