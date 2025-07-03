

# 🤖 ChatCraft: Personalized WhatsApp-Style AI Chatbot

Fine-Tuning TinyLlama + LoRA + Gradio Web UI

Welcome to **ChatCraft** – an end-to-end pipeline to **fine-tune lightweight open-source language models** on your **WhatsApp-style conversations**, then deploy them as an **AI chatbot with a web UI using Gradio**.

> Think of it as building your own ChatGPT – but trained on your real chat history, running privately, and tuned to sound just like you (or your support team).

---

## 🚀 Live Demo Interface (Gradio Included!)

Use your fine-tuned model in a web browser:

```bash
python gradio_chat.py
```

* Clean UI for testing
* Private, fast, and fully offline
* Ready to be deployed or shared

---

## 🔥 Highlights

✅ Fine-tunes [`TinyLlama-1.1B-Chat`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
✅ Efficient LoRA fine-tuning (requires <8GB VRAM)
✅ Uses real conversational data (WhatsApp, Slack, etc.)
✅ Includes Gradio-based Web UI
✅ Perfect for custom support bots, tutors, companions
✅ Deployable on Hugging Face Spaces, Streamlit, or Colab

---

## 📁 Project Overview

```
ChatCraft/
├── dialogs.txt              # Your chat data (instruction \t response)
├── alpaca_data.jsonl        # Converted format for training
├── train_tinyllama.ipynb    # Notebook to fine-tune TinyLlama
├── gradio_chat.py           # Web UI using Gradio
├── fine_tuned_model/        # Output: trained model and tokenizer
└── README.md
```

---

## 🛠️ Setup

Install dependencies:

```bash
pip install whatstk einops gradio bitsandbytes transformers peft accelerate datasets
```

---

## 📥 Step 1: Prepare Your Data

### Option A: Pre-Formatted Dialogs

Use `dialogs.txt` with tab-separated pairs:

```
How are you?	I’m doing well, thank you!
What’s your name?	I’m ChatCraft, your assistant.
```

### Option B: Use WhatsApp Exports

```python
from whatstk import WhatsAppChat
chat = WhatsAppChat.from_source("whatsapp.txt")
df = chat.df[['date', 'name', 'text']]
```

Then convert into (instruction, response) pairs based on time gaps or sender.

---

## 🔄 Step 2: Format for Fine-Tuning

Convert to [Alpaca-style format](https://github.com/tatsu-lab/stanford_alpaca):

```json
{"instruction": "What's the weather?", "input": "", "output": "It’s sunny and warm today!"}
```

Save as `alpaca_data.jsonl`.

---

## 🧠 Step 3: Fine-Tune the Model

Open `train_tinyllama.ipynb`:

* Loads **TinyLlama** in 4-bit
* Applies **LoRA adapters**
* Tokenizes & formats the data
* Fine-tunes for 1–2 epochs
* Saves your **custom chatbot model**

All this can run smoothly on **Google Colab (T4)**.

---

## 💬 Step 4: Chat via Gradio Web UI

Run the chatbot in-browser:

```bash
python gradio_chat.py
```

You’ll see a clean Gradio interface to interact with your trained model.

---

## 🧠 Why This Works

This project uses:

* **LoRA**: Trains just a few million parameters (not the whole model).
* **4-bit quantization**: Shrinks model size so it runs on laptops or free Colab.
* **Alpaca-style prompts**: Instruction + response formatting.

Together, this gives you:

* ✅ Fast training
* ✅ Low cost
* ✅ Highly personalized AI

---

## 💡 Use Cases

| 🛠️ Scenario            | 👥 Target Users             |
| ----------------------- | --------------------------- |
| Personal Companion Bot  | Individuals, students       |
| Customer Support AI     | Startups, D2C brands        |
| Therapy-style Assistant | Mental health orgs          |
| Educational Tutor       | EdTech, self-learners       |
| Internal Knowledge Bot  | Teams, HR, onboarding       |
| Multilingual Agent      | Rural support, Govt portals |

---

## 🌐 Optional: Upload to Hugging Face

```python
from huggingface_hub import login
model.push_to_hub("your-username/ChatCraft-TinyLlama")
```

---

## 👤 Author

**Varun Haridas**
Email: [varun.haridas321@gmail.com](mailto:varun.haridas321@gmail.com)
Built with ❤️ using TinyLlama, Hugging Face, and Gradio

---

## 📌 Next Add-Ons

* [ ] Contextual memory chat
* [ ] Telegram/WhatsApp bot integration
* [ ] Multi-user dashboard
* [ ] Long-form summarization / retrieval
* [ ] Integration with company databases or Notion

---


