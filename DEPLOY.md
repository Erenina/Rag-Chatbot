# Deploying to Hugging Face Spaces (free)

This app runs on a **Docker Space** (free CPU tier — enough RAM for the local
embedding model). The repo is deploy-ready: it ships a [`Dockerfile`](Dockerfile),
a [`.dockerignore`](.dockerignore) (keeps secrets out of the image), and the HF
config front matter at the top of [`README.md`](README.md) (`sdk: docker`,
`app_port: 7860`).

## Steps

1. **Create a Hugging Face account** — https://huggingface.co/join (free).

2. **Create a new Space** — https://huggingface.co/new-space
   - Owner: your username · Space name: `rag-chatbot`
   - **SDK: Docker** → **Blank**
   - Hardware: **CPU basic** (free)

3. **Add your API key as a secret** — in the Space:
   *Settings → Variables and secrets → New secret*
   - Name: `GROQ_API_KEY`
   - Value: your free key from https://console.groq.com (starts with `gsk_`)

   Generation defaults to Groq, so this single secret is enough. To use Gemini or
   Claude instead, also set `LLM_PROVIDER` and the matching key as secrets.

4. **Push the code to the Space.** From this repo folder:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/rag-chatbot
   git push space main
   ```
   When prompted, use your HF username and a **write** token
   (https://huggingface.co/settings/tokens) as the password.

5. **Wait for the build** (a few minutes), then open your live demo:
   `https://huggingface.co/spaces/<your-username>/rag-chatbot`

The demo comes **pre-loaded** with the example documents (`PRELOAD_EXAMPLES=true`
in the Dockerfile), so visitors can ask questions immediately — or upload their own.

## Notes

- **Secrets never enter the image.** `.env` is in `.dockerignore`; the key is
  injected at runtime from the Space secret.
- **Storage is ephemeral** on the free tier: the example docs are re-loaded on
  every restart; user uploads last until the next restart.
- **Test the image locally** (optional, needs Docker):
  ```bash
  docker build -t rag-chatbot .
  docker run -p 7860:7860 -e GROQ_API_KEY=gsk_your_key rag-chatbot
  # → http://localhost:7860
  ```
