# whatsapp-ai-finetuning
🤖 WhatsApp Chatbot Fine-Tuning with TinyLlama
This project builds a personalized AI chatbot by fine-tuning the TinyLlama model on WhatsApp-style conversations or custom dialog pairs.

It uses LoRA (Low-Rank Adaptation) and 4-bit quantization to enable efficient training on low-resource GPUs like Google Colab (T4).

📦 Features
✅ Fine-tunes TinyLlama on your WhatsApp/exported dialog data

✅ Lightweight: 4-bit quantized + LoRA (RAM/GPU-friendly)

✅ Builds a private, offline chatbot

✅ Includes basic CLI chat interface

✅ Easy to extend to Gradio/Telegram bots

📁 Project Structure
bash
Copy
Edit
.
├── dialogs.txt              # Your instruction-response pairs (TSV)
├── fine_tuned_model/        # Output: Trained model + tokenizer
├── train_tinyllama.ipynb    # Main notebook for fine-tuning
├── chat_interface.py        # CLI-based chat demo (optional)
├── README.md
🛠️ Setup & Installation
Run this in a fresh Colab or your own environment:

bash
Copy
Edit
pip install whatstk einops bitsandbytes transformers peft accelerate datasets
📥 Step 1: Prepare Your Data
Option A: Dialog Pairs File (Recommended)
Format: dialogs.txt — tab-separated text file with:

tsv
Copy
Edit
Hi, how are you?	I’m doing well, thanks!
What’s your name?	My name is VarunBot.
Option B: Raw WhatsApp Export (Optional)
python
Copy
Edit
from whatstk import WhatsAppChat

chat = WhatsAppChat.from_source("chat.txt")
df = chat.df[['date', 'name', 'text']]
You’ll need to clean, group, and structure this into (instruction, response) format.

🔄 Step 2: Convert to Alpaca Format
Transform dialog data into the format:

json
Copy
Edit
{
  "instruction": "User's message",
  "input": "",
  "output": "Bot's reply"
}
Use Python to save this as .jsonl.

🧠 Step 3: Fine-Tune TinyLlama
In train_tinyllama.ipynb, you'll:

Load TinyLlama in 4-bit mode

Apply LoRA adapters

Format and tokenize dataset

Fine-tune with HuggingFace Trainer

Save model and tokenizer

🧪 Step 4: Inference (CLI Chatbot)
Use a simple CLI interface (or build a web UI later):

python
Copy
Edit
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")

def chat(user_input):
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

chat("What’s your name?")
🧠 How It Works
✅ LoRA lets us fine-tune only a small subset of model weights.

✅ Alpaca-style format helps the model understand instruction-following behavior.

✅ The trained model mimics tone and responses from your data.

💡 Real-Life Use Cases
Personalized companion chatbot

Mental health / empathetic bots

Customer support assistant (chat-based)

Internal knowledge bot (train on team chats)

Educational tutor bots

Local language/dialect AI agents

🚀 Next Steps
🔗 Add Gradio UI for demo

🤖 Deploy as a Telegram or WhatsApp bot

☁️ Upload to Hugging Face Hub

🌍 Train on multilingual or domain-specific datasets

👤 Author & Credits
Built by Varun Haridas
Powered by HuggingFace Transformers, PEFT, and TinyLlama

