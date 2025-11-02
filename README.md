🌍 Global Wellness Chatbot

"A Smart Health Companion for Mind and Body"
 

🧠 Overview

Global Wellness Chatbot is an intelligent health assistant that helps users manage stress, anxiety, depression, sleep issues, and burnout through conversational guidance and wellness recommendations.
It supports multiple languages and provides both User and Admin profiles with an interactive, modern UI.


🛡 Security

All user credentials are secured using bcrypt-hashed passwords.

Sensitive data is never stored in plain text.

Admin access includes restricted authentication with session validation.



🗣 Key Features

✅ Multilingual Chatbot (Supports English + Multiple Languages)
✅ Voice Input – Speak directly to the chatbot using your microphone
✅ Voice Output – Click the 🔊 Speaker button to hear the bot’s response aloud
✅ Smart Mental Health Recommendations
✅ Separate Admin Panel for user management and analytics
✅ Dynamic, Interactive, and Responsive UI
✅ Works with Offline or Online Models (Flexible Configuration)



## 💻 Tech Stack  
--------------------------------------------------------
| Category            | Tools / Libraries              |
|---------------------|--------------------------------|
| Backend Framework   | Flask                          |
| Frontend UI         | HTML, CSS, Jinja2 Templates    |
| NLP / AI Models     | Hugging Face Transformers      |
| Sentence Similarity | Sentence Transformers (MiniLM) |
| Translation         | Helsinki-NLP (English ↔ Hindi) |
| Database            | SQLite                         |
| Containerization    | Docker                         |
--------------------------------------------------------

🧩 Models Used

The chatbot uses several pre-trained models from Hugging Face for translation and text generation.
You can either:

Use Online Models: Automatically downloaded during runtime (slower at first run).

Or Use Offline Models: Download them manually from the following links and store them inside your models/ directory.


Example Model Links :


[Helsinki-NLP/opus-mt-en-hi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi
) – English ↔ Hindi

[distilbert-base-uncased](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi
) – Text understanding

[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/distilbert-base-uncased
) – Semantic similarity, emotion analysis, and contextual embedding generation





🧱 How to Run Locally

# ⿡ Clone the repository
git clone https://github.com/K-Pavan-Sai-Varma/wellness-chatbot.git
cd wellness-chatbot

# ⿢ Build Docker Image
docker build -t wellness-bot .

# ⿣ Run the Container
docker run -p 5000:5000 -v "%cd%":/app wellness-bot

Then open your browser and go to →
👉 http://localhost:5000




🚀 Deployment

The project can be easily deployed on:

Hugging Face Spaces (Docker)

Streamit / Render / Railway / AWS EC2 (Optional)

