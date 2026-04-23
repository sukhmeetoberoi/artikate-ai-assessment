import json
import random
import os

DATA_DIR = "section_03_classifier/data"

def generate_synthetic_data(num_samples=200):
    """Generate synthetic data for support ticket classification."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    categories = ["Technical Support", "Billing", "Feature Request", "Account Access"]
    
    templates = {
        "Technical Support": [
            "My app keeps crashing when I click the submit button.",
            "I'm getting a 500 error on the dashboard.",
            "The integration with Slack is not working.",
            "Slow performance on the mobile app.",
            "Database connection timeout occurring frequently."
        ],
        "Billing": [
            "I was double charged for my subscription this month.",
            "Where can I download my latest invoice?",
            "I want to upgrade my plan to Enterprise.",
            "My payment method was rejected.",
            "How do I cancel my trial?"
        ],
        "Feature Request": [
            "Can you add a dark mode to the interface?",
            "I would love to see a bulk export feature.",
            "It would be great if we could tag other users.",
            "Please support multi-factor authentication.",
            "Add more themes to the profile page."
        ],
        "Account Access": [
            "I forgot my password and can't reset it.",
            "I'm locked out of my account after too many attempts.",
            "How do I change the primary email on my profile?",
            "I can't log in using my Google account.",
            "My account has been suspended for no reason."
        ]
    }

    data = []
    for _ in range(num_samples):
        cat = random.choice(categories)
        text = random.choice(templates[cat])
        # Add some noise/variation
        if random.random() > 0.5:
            text = text.lower()
        
        data.append({"text": text, "label": cat})

    # Split into train and test
    random.shuffle(data)
    split = int(0.8 * num_samples)
    train_data = data[:split]
    test_data = data[split:]

    with open(os.path.join(DATA_DIR, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(DATA_DIR, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples.")

if __name__ == "__main__":
    generate_synthetic_data()
