import requests
from typing import Dict, Any, List
import os
import time

class HumeClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hume.ai/v0"
        self.headers = {
            "X-Hume-Api-Key": f"{api_key}",
            "Content-Type": "application/json"
        }

    def analyze_text(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze text using Hume.ai Batch API for language emotion detection.
        
        Args:
            texts: list of text strings to analyze
            
        Returns:
            Dict containing emotion scores
        """
        url = f"{self.base_url}/batch/jobs"
        
        # Prepare the request payload
        payload = {
            "text": texts,
            "models": {
                "language": {}
            }
        }

        try:
            # Start the job
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            job_id = response.json()["job_id"]
            time.sleep(1)  # Brief pause to allow processing

            # Get results
            results_url = f"{self.base_url}/batch/jobs/{job_id}/predictions"
            results = requests.get(results_url, headers=self.headers)
            results.raise_for_status()
            
            data = results.json()[0]
            processed_results = []
            
            for prediction in data["results"]["predictions"]:
                emotions = self._aggregate_emotions(prediction)
                processed_results.append(emotions)
                
            return processed_results
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Hume API: {e}")
            return {"emotions": []}

    def _aggregate_emotions(self, prediction: Dict) -> Dict[str, List[Dict[str, float]]]:
        """
        Aggregate word-level predictions into message-level emotions.
        Uses weighted average based on word length.
        """
        # Get all word-level predictions
        words = prediction["models"]["language"]["grouped_predictions"][0]["predictions"]
        
        # Initialize emotion aggregates
        emotion_sums = {}
        total_length = 0
        
        # Sum up emotions weighted by word length
        for word in words:
            word_length = word["position"]["end"] - word["position"]["begin"]
            total_length += word_length
            
            for emotion in word["emotions"]:
                if emotion["name"] not in emotion_sums:
                    emotion_sums[emotion["name"]] = 0
                emotion_sums[emotion["name"]] += emotion["score"] * word_length
        
        # Calculate weighted averages
        aggregated_emotions = [
            {"name": name, "score": score / total_length}
            for name, score in emotion_sums.items()
        ]
        
        # Sort by score descending
        aggregated_emotions.sort(key=lambda x: x["score"], reverse=True)
        
        return {"emotions": aggregated_emotions}

def test_hume_client():
    # Get API key from environment variable
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        raise ValueError("HUME_API_KEY environment variable not set")
    
    client = HumeClient(api_key)
    
    # Test text samples with expected emotional content
    test_cases = [
        "I'm really happy and excited about this!",
        "This makes me very angry and frustrated.",
        "I feel sad and disappointed about the outcome.",
        "I fucking hate you!"
    ]
    
    print(f"\nAnalyzing multiple texts in batch:")
    results = client.analyze_text(test_cases)
    
    for text, result in zip(test_cases, results):
        print(f"\nText: {text}")
        print("Top 3 emotions detected:")
        for emotion in result["emotions"][:3]:
            print(f"{emotion['name']}: {emotion['score']:.2f}")

if __name__ == "__main__":
    test_hume_client() 