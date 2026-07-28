from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "I am very happy today.",
    "The weather is terrible.",
    "ChatGPT helped me finish my project.",
    "I don't like waiting for hours.",
    "This software is excellent."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Result: {result}")
    print("-" * 50)