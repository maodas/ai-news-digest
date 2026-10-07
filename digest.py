import os
import requests
import feedparser

FEEDS = [
    {"source": "HF Daily Papers", "url": "https://huggingface.co/blog/feed.xml"},
    {"source": "ArXiv CS.AI", "url": "http://export.arxiv.org/rss/cs.AI"},
    {"source": "LocalLLaMA", "url": "https://www.reddit.com/r/LocalLLaMA/.rss"},
    {"source": "Machine Learning Reddit", "url": "https://www.reddit.com/r/MachineLearning/.rss"},
    {"source": "MIT Tech Review AI", "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
    {"source": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/"}
]

def fetch_feed_data() -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai-digest/2.0"}
    feed_items = []
    
    for f in FEEDS:
        try:
            resp = requests.get(f["url"], headers=headers, timeout=12)
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:4]:
                title = entry.get("title", "No Title").strip()
                link = entry.get("link", "").strip()
                summary = entry.get("summary", "")[:350].replace("\n", " ").strip()
                feed_items.append(f"Source: {f['source']}\nTitle: {title}\nLink: {link}\nSnippet: {summary}")
        except Exception as e:
            print(f"Skipping {f['source']}: {e}")
            continue

    return "\n---\n".join(feed_items)

def summarize_with_groq(raw_text: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    prompt = (
        "You are an elite frontier AI researcher and technical curator. "
        "Analyze the following incoming raw signals and select the TOP 10 most impactful, early-stage, "
        "or technically significant developments from the last 24 hours.\n\n"
        "Selection Philosophy:\n"
        "- Prioritize emerging architectures, open-weights milestones (e.g., Nous Hermes, unquantized releases), "
        "novel training paradigms, synthetic data methods, or early incubation research over generic corporate PR.\n"
        "- Focus on practical technical shifts.\n\n"
        "Formatting Requirements:\n"
        "- Exactly 10 items, numbered 1 through 10.\n"
        "- For each item: **Bold Title** followed by a thoughtful, analytical 2-sentence breakdown explaining "
        "WHAT it is and WHY it matters technically.\n"
        "- End each entry with a clean markdown link: [Read Source](URL).\n"
        "- Separate items with clean line breaks.\n"
        "- Use standard Telegram-compatible Markdown.\n\n"
        f"Raw data:\n{raw_text}"
    )

    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {"role": "system", "content": "You are a specialized AI systems curator and technical editor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.25
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }

    res = requests.post(url, json=payload, headers=headers, timeout=45)
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"]

def send_telegram_chunks(message: str):
    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Telegram limit is 4096 chars; split by double newlines if needed
    chunk_size = 3800
    chunks = []
    
    if len(message) <= chunk_size:
        chunks = [message]
    else:
        paragraphs = message.split("\n\n")
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
            else:
                current_chunk += p + "\n\n"
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        res = requests.post(url, json=payload, timeout=20)
        res.raise_for_status()

if __name__ == "__main__":
    print("Collecting feeds...")
    articles = fetch_feed_data()
    print("Generating 10-item frontier briefing via Groq...")
    digest = summarize_with_groq(articles)
    print("Sending briefing to Telegram...")
    send_telegram_chunks(digest)
    print("Done!")