from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Summarizer using facebook/bart-large-cnn
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Split large text into manageable chunks
def split_text(text, max_tokens=1000):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/summarize/", response_class=HTMLResponse)
async def summarize(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")
    chunks = split_text(text)
    summaries = [
        summarizer(chunk, max_length=300, min_length=100, do_sample=False)[0]["summary_text"]
        for chunk in chunks
    ]
    full_summary = "\n\n".join(summaries)
    return templates.TemplateResponse("form.html", {"request": request, "summary": full_summary})
