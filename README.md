 ChatCraft: Fine-Tuned WhatsApp AI Assistant with TinyLlama + Gradio
Welcome to ChatCraft — a powerful pipeline to fine-tune lightweight language models on real conversational data (like WhatsApp chats) and deploy them with a sleek Gradio-powered web UI.

Whether you're building a personal AI companion, a customer support agent, or a domain-specific tutor, ChatCraft helps you train your own custom model using open-source tools and run it efficiently on minimal hardware.

🚀 Demo: Web Chat Interface (Gradio)
With just one command, launch your AI assistant in the browser:

<!-- (replace with actual GIF if you have) -->

bash
Copy
Edit
python gradio_chat.py
🌟 Project Highlights
✅ Fine-tunes TinyLlama-1.1B-Chat
✅ Trains on your WhatsApp chats or dialog datasets
✅ Uses LoRA + 4-bit quantization → train on Google Colab / laptop
✅ Clean Gradio chat UI for instant testing
✅ Easy to extend to Telegram / WhatsApp bot deployment
✅ Fully offline and private

📁 Project Structure
bash
Copy
Edit
.
├── dialogs.txt              # Chat data (instruction \t response)
├── alpaca_data.jsonl        # Converted Alpaca-format dataset
├── train_tinyllama.ipynb    # Fine-tuning notebook (TinyLlama + LoRA)
├── gradio_chat.py           # Gradio-based web chatbot interface
├── fine_tuned_model/        # Output directory after training
├── README.md
🔧 Installation
Requirements
Python 3.8+

CUDA GPU (T4, A100, or Colab) recommended

Install Dependencies
bash
Copy
Edit
pip install whatstk einops gradio bitsandbytes transformers peft accelerate datasets
📥 Step 1: Prepare Your Data
Option A: Predefined Dialogs
Use dialogs.txt with this structure:

ts
Copy
Edit
Hi there!	Hey! How can I help you today?
What's your name?	I'm your AI assistant, trained just for you.
Option B: WhatsApp Export
python
Copy
Edit
from whatstk import WhatsAppChat
chat = WhatsAppChat.from_source("chat.txt")
df = chat.df[['date', 'name', 'text']]
Group and convert messages into dialog pairs using time gaps or names.

🔄 Step 2: Convert to Alpaca Format
Transforms into:

json
Copy
Edit
{"instruction": "Hi!", "input": "", "output": "Hello! How can I assist you?"}
Save as alpaca_data.jsonl.

🧠 Step 3: Fine-Tune the Model
Open train_tinyllama.ipynb to:

Load TinyLlama in 4-bit mode

Apply LoRA adapters

Tokenize and train on your chat data

Save the fine-tuned model

Fine-tunes on Google Colab in under 2 hours!

💬 Step 4: Launch Gradio Chat UI
Run:

bash
Copy
Edit
python gradio_chat.py
This loads your fine-tuned model and serves a browser-based chatbot where anyone can talk to your custom-trained AI.

💼 Real-World Use Cases
🔍 Use Case	💡 Description
Personal AI Companion	Trained on your chat history or tone
Customer Support Chatbot	Trained on company FAQ / WhatsApp tickets
Mental Health Assistant	Mimics therapist-patient conversations
Educational Tutor	Learn from solved doubts and tutor chats
Regional Language Bot	Hinglish / Malayalam-trained assistant
Memory Bot	Remembers past messages / life events

🌐 Next Steps
✅ Add memory to Gradio chat (context-based chat)

🌐 Deploy to Hugging Face Spaces or your server

🤖 Integrate with Telegram/WhatsApp bot

🧠 Train on multilingual or multi-user data

📤 HuggingFace Integration (Optional)
Login and upload your fine-tuned model:

python
Copy
Edit
from huggingface_hub import login
login()
model.push_to_hub("your-model-name")
👨‍💻 Built With
TinyLlama – 1.1B instruction-tuned base model

LoRA – Efficient fine-tuning method

Gradio – Lightweight UI for demo and testing

HuggingFace Transformers – Core ML framework

whatstk – WhatsApp .txt parser

👤 Author
Varun Haridas
Email: varun.haridas321@gmail.com
Made with ❤️ for open-source AI

