# Question_mk

:

##📘 PDF → Questions Generator (MCQ & Essay)

This project converts any uploaded PDF document into automatically generated Multiple-Choice Questions (MCQ) or Essay questions using a Large Language Model (LLM).
It’s built with FastAPI (backend), Gemini API (LLM), and a simple HTML/Tailwind/HTMX frontend for interaction.

🚀 Features

📂 Upload a PDF and instantly generate MCQs or Essay questions.

🎯 Choose difficulty level: easy, medium, hard.

⚙️ Configurable backend/model (default: Google Gemini 2.0 Flash).

🧠 Supports semantic chunking of PDFs & context retrieval.

✨ Reranking support with cross-encoder (optional).

💾 Questions auto-saved as JSON in outputs/.

🖥️ Clean web UI (HTMX + Tailwind).

##🛠️ Tech Stack

Backend: FastAPI

Frontend: TailwindCSS + HTMX

LLM: Google Gemini API

PDF Parsing: PyMuPDF

Embeddings & Reranking: sentence-transformers / cross-encoders

##📂 Project Structure
Question_mk/
│── config/             # Config (backend, model, API key)
│   └── config.py
│── core/               # Core logic
│   ├── pdf_utils.py    # PDF parsing + chunking
│   ├── embed_store.py  # Embedding store
│   ├── llm_client.py   # LLM client (Gemini / HF / Ollama)
│   ├── prompts.py      # Prompt templates
│   └── schemas.py      # Pydantic models
│── extras/             # Optional extras (NER, reranking)
│── front_end/          # UI
│   ├── static/         # JS, CSS
│   │   └── app.js
│   └── templates/      # HTML (index.html)
│── outputs/            # Auto-saved generated questions
│── main.py             # FastAPI entrypoint
│── requirements.txt
└── README.md

##⚙️ Setup
1️⃣ Clone the repo
git clone https://github.com/<your-username>/pdf-question-generator.git
cd pdf-question-generator

2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows (PowerShell)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Configure your API key

Edit config/config.py and add your Gemini API key:

@dataclass
class AppConfig:
    LLM_BACKEND: str = "gemini"
    LLM_MODEL: str   = "gemini-2.0-flash"
    GEMINI_API_KEY: str = "YOUR_GEMINI_KEY"
    USE_RERANK: bool = False
    HOST: str = "127.0.0.2"
    PORT: int = 8000

###▶️ Run the App
python main.py


###App will be served at:

Backend API: http://127.0.0.2:8000

Frontend UI: http://127.0.0.2:8000/

Swagger docs: http://127.0.0.2:8000/docs

##💡 Usage

Open http://127.0.0.2:8000/

Upload a PDF file.

Select:

Question type: MCQ or Essay

Difficulty: easy / medium / hard

Count: number of questions

Topic (optional): narrow focus

Click Generate.

✅ Questions appear on the page:

MCQs → interactive quiz style

Essays → with rubric & target keywords

💾 Download results as JSON (saved automatically in outputs/).



##🔮 Roadmap

 Add CSV export for questions.

 Support more LLMs (OpenAI, Anthropic, Mistral).

 Add option to shuffle MCQ choices.

 Authentication + multi-user support.

##🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to add.

###📜 License

MIT License © 2025
