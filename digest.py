import os
import requests
import feedparser
from google import genai

FEEDS = [
    {"source": "Hugging Face", "url": "https://huggingface.co/blog/feed.xml"},
    {"source": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"source": "ArXiv AI", "url": "http://export.arxiv.org/rss/cs.AI"}
]

def fetch_feed_data() -> str:
    feed_items = []
    for f in FEEDS:
        feed = feedparser.parse(f["url"])
        for entry in feed.entries[:3]:
            title = entry.get("title", "No Title")
            link = entry.get("link", "")
            summary = entry.get("summary", "")[:250].replace("\n", " ")
            feed_items.append(f"Source: {f['source']}\nTitle: {title}\nLink: {link}\nSnippet: {summary}")
    return "\n---\n".join(feed_items)

def summarize_with_gemini(raw_text: str) -> str:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = (
        "You are an executive AI curator. Summarize the following news entries into a concise daily briefing.\n\n"
        "Formatting Rules:\n"
        "- Group into 3 to 4 key stories.\n"
        "- For each story: provide a bold title, a clear 2-sentence breakdown of what happened, and a direct link: [Read Source](URL).\n"
        "- Use standard Markdown compatible with Telegram.\n\n"
        f"Raw data:\n{raw_text}"
    )
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    return response.text

def send_telegram_message(message: str):
    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    
    # Telegram message character limit protection
    if len(message) > 4000:
        message = message[:3990] + "..."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    
    res = requests.post(url, json=payload, timeout=20)
    res.raise_for_status()

if __name__ == "__main__":
    articles = fetch_feed_data()
    digest = summarize_with_gemini(articles)
    send_telegram_message(digest)