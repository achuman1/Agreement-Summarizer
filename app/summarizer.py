from transformers import pipeline

# Load model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text: str, max_length=512, min_length=30) -> str:
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']
