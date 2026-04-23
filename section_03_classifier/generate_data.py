import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

DATA_DIR = "section_03_classifier/data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
CATEGORIES = ["billing", "technical_issue", "feature_request", "complaint", "other"]

class DataGenerator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def generate_batch(self, category: str, count: int) -> List[Dict]:
        """Generate a batch of synthetic support tickets for a specific category."""
        prompt = (
            f"Generate {count} unique examples of customer support tickets for the category: '{category}'. "
            f"The tickets should be realistic, varying in length and tone. "
            f"Return ONLY a JSON list of strings, nothing else. "
            f"Example format: [\"ticket text 1\", \"ticket text 2\"]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format={"type": "json_object"} if "json_object" in str(self.model) else None
            )
            content = response.choices[0].message.content
            # Handle potential Groq response formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            # Some models return an object with a key, some return the list directly
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        return [{"text": text, "label": category} for text in val[:count]]
            elif isinstance(data, list):
                return [{"text": text, "label": category} for text in data[:count]]
            
            return []
        except Exception as e:
            print(f"Error generating batch for {category}: {e}")
            return []

    def run(self):
        """Generate 1000 examples (200 per category)."""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        all_data = []
        target_per_category = 200
        batch_size = 20

        print(f"--- Starting Synthetic Data Generation ---")
        for category in CATEGORIES:
            print(f"Generating data for category: {category}")
            category_data = []
            while len(category_data) < target_per_category:
                batch = self.generate_batch(category, batch_size)
                category_data.extend(batch)
                
                if len(category_data) % 50 == 0 or len(category_data) == target_per_category:
                    print(f"  Progress for {category}: {len(category_data)}/200")
                
                # Small sleep to avoid rate limits
                time.sleep(1)
            
            all_data.extend(category_data[:target_per_category])

        with open(TRAIN_FILE, "w") as f:
            json.dump(all_data, f, indent=2)
        
        print(f"Successfully saved 1000 examples to {TRAIN_FILE}")

if __name__ == "__main__":
    generator = DataGenerator()
    generator.run()
