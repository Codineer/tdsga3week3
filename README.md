# Safe AI Mailroom agent
Run: `pip install -r requirements.txt && LLM_KEY=... uvicorn app:app --host 0.0.0.0 --port 8000`
Works without LLM_KEY (deterministic safe fallback), useful for testing replay/409/400 paths.
Any OpenAI-compatible model: set LLM_URL, LLM_KEY, LLM_MODEL (Groq, Gemini OpenAI endpoint, OpenRouter, Ollama).
Deploy (Render/Fly/Railway) so the URL is public HTTPS, no credentials in the URL. Use ONE worker (SQLite): keep a persistent disk.
Submit: https://<host>/v1/mailroom/actions
