import os
import requests
import feedparser

# Balanced feed mix: frontier research, academic preprints, open-source communities, and enterprise tech
FEEDS = [
    {"source": "HF Daily Papers", "url": "https://huggingface.co/blog/feed.xml"},
    {"source": "ArXiv CS.AI", "url": "http://export.arxiv.org/rss/cs.AI"},
    {"source": "LocalLLaMA", "url": "https://www.reddit.com/r/LocalLLaMA/.rss"},
    {"source": "Machine Learning Reddit", "url": "https://www.reddit.com/r/MachineLearning/.rss"},
    {"source": "MIT Tech Review AI", "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
    {"source": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/"}
]

# Supported candidate models in prioritized order
CANDIDATE_MODELS = [
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "meta-llama/llama-guard-3-8b"
]

def fetch_feed_data() -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ai-digest/3.0"}
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
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are an executive AI curator and frontier intelligence analyst.\n\n"
        "Your task is to review the incoming feed text and deliver an executive daily digest "
        "consisting of EXACTLY 10 stories split into two balanced categories:\n\n"
        "🏢 **SECTION 1: ESTABLISHED LABS & INDUSTRY (Items 1 to 5)**\n"
        "- Coverage: OpenAI, Google DeepMind, Anthropic, Meta, Microsoft, Apple, NVIDIA, or major enterprise milestones.\n"
        "- Focus: Official model launches, enterprise features, compute infrastructure, and major product rollouts.\n\n"
        "⚡ **SECTION 2: FRONTIER RESEARCH & OPEN-WEIGHTS (Items 6 to 10)**\n"
        "- Coverage: Community open-weights (e.g., Nous Research/Hermes, community fine-tunes), arXiv preprints, "
        "LocalLLaMA breakthroughs, novel training recipes, synthetic data pipelines, or early incubation tools.\n"
        "- Focus: Architectural breakthroughs, synthetic data recipes, and technical shifts before they hit mainstream press.\n\n"
        "FORMATTING SPECIFICATIONS:\n"
        "- Exactly 10 numbered items total (1-5 in Section 1, 6-10 in Section 2).\n"
        "- Format each entry as:\n"
        "  [Number]. **[Headline]**\n"
        "  [Two-sentence technical analysis: what it is + why it matters].\n"
        "  [Read Source](URL)\n\n"
        "- Output clean, Telegram-compatible markdown with line breaks between items.\n\n"
        f"Raw data to curate from:\n{raw_text}"
    )

    last_error = None
    for model_name in CANDIDATE_MODELS:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a concise, balanced AI technology curator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        print(f"Attempting completion with model: {model_name}...")
        res = requests.post(url, json=payload, headers=headers, timeout=45)
        
        if res.ok:
            data = res.json()
            return data["choices"][0]["message"]["content"]
        
        print(f"Model {model_name} failed ({res.status_code}): {res.text}")
        last_error = res.text

    raise RuntimeError(f"All candidate models failed. Last response: {last_error}")

def send_telegram_chunks(message: str):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variable is missing.")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Telegram limit is 4096 characters per message
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
    print("Generating balanced 10-item briefing...")
    digest = summarize_with_groq(articles)
    print("Sending briefing to Telegram...")
    send_telegram_chunks(digest)
    print("Briefing delivered successfully!")