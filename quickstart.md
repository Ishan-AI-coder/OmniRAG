<div align="center">
  <br>
  <kbd>Developer Guide</kbd>
  <h1>🚀 Quickstart & Installation</h1>
  <p><em>Get the Multimodal RAG system running locally and configure your environment limits.</em></p>
  <br>
</div>

---

### 1. Setup the Environment
First, clone the repository and initialize your virtual environment to isolate the project dependencies.

```bash
git clone [https://github.com/yourusername/multimodal-rag.git](https://github.com/yourusername/multimodal-rag.git)
cd multimodal-rag
uv init
uv sync
```

### 2. API Configuration
You will need API keys from Google and Cohere to run the vision and reranking models. 
* **Google Gemini API:** Get your key from [Google AI Studio](https://aistudio.google.com/app/api-keys).
* **Cohere API:** Get your reranker key from the [Cohere Dashboard](https://docs.cohere.com/docs/rate-limits).

Create a `.env` file in the root directory to store your API keys securely:

```env
GEMINI_API_KEY=your_google_ai_studio_key
COHERE_API_KEY=your_cohere_key
FASTEMBED_CACHE_PATH=./model_cache
```
> **Note:** Setting the `FASTEMBED_CACHE_PATH` is a good practice. It ensures your local embedding models download directly into your project folder rather than clogging up your hidden system drives.

### 3. Windows / WSL Memory Fix (CRITICAL)
If you are running this via WSL (Windows Subsystem for Linux), the local Qdrant and FastEmbed models will aggressively consume RAM during embedding generation. This will cause the Linux kernel to deploy an Out-Of-Memory (OOM) sniper, throwing a `Killed` crash. **You must throttle Linux memory.**

1. Open a standard Windows Command Prompt (not your WSL terminal).
2. Run this exact command to forcefully create a configuration file:
   ```cmd
   notepad %USERPROFILE%\.wslconfig
   ```
3. Paste the following limits into Notepad and save the file:
   ```ini
   [wsl2]
   memory=6GB
   ```
4. Shut down the Linux engine to apply the new rules:
   ```cmd
   wsl --shutdown
   ```

### 4. Run the Application
With your environment active, API keys set, and memory capped, launch the Streamlit interface:

```bash
uv run streamlit run app.py
```