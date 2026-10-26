import os
import asyncio
import logging
import aiohttp
from llama_cpp import Llama

# Токен берем из безопасных переменных окружения Hugging Face
MAX_BOT_TOKEN = os.getenv("MAX_BOT_TOKEN", "")
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"

logging.basicConfig(level=logging.INFO)
last_update_id = 0

logging.info("⏳ Загрузка модели Qwen2.5... При первом запуске это займет 1-2 минуты.")
llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=2048,
    n_threads=4,  # Используем доступные ядра CPU
    verbose=False
)
logging.info("✅ Модель загружена и готова к работе!")

SYSTEM_PROMPT = """
Ты — дружелюбный, терпеливый и грамотный помощник учителя физики Ильи Валерьевича.
Твоя аудитория — школьники. Твоя цель — помочь им понять материал, а не просто дать списать.
ПРАВИЛА:
1. Объясняй сложные физические явления простым, живым языком, с понятными примерами из жизни.
2. При решении задач ВСЕГДА структурируй ответ: "Дано", "Найти", "Основные формулы", "Решение по шагам", "Ответ".
3. Если ученик просит готовый ответ без объяснений, вежливо откажи и предложи разобрать первый шаг вместе.
4. Отвечай кратко и по делу.
"""

async def get_llm_response(prompt: str) -> str:
    # Форматируем промпт в стиле чата Qwen
    full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    output = llm(
        full_prompt,
        max_tokens=600,
        temperature=0.3, # Низкая температура для точных физических расчетов
        stop=["<|im_end|>"]
    )
    return output["choices"][0]["text"].strip()

async def send_message(session: aiohttp.ClientSession, chat_id: str, text: str):
    url = f"https://api.max.ru/bot{MAX_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            logging.error(f"Ошибка отправки в MAX: {await response.text()}")

async def poll_updates(session: aiohttp.ClientSession):
    global last_update_id
    url = f"https://api.max.ru/bot{MAX_BOT_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=30"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            updates = data.get("result", [])
            for update in updates:
                last_update_id = update["update_id"]
                message = update.get("message")
                if message and "text" in message:
                    chat_id = str(message["chat"]["id"])
                    user_text = message["text"]
                    logging.info(f"📩 Вопрос от {chat_id}: {user_text}")
                    
                    bot_response = await get_llm_response(user_text)
                    
                    # Разбиваем ответ на части, если он длинный
                    for i in range(0, len(bot_response), 4000):
                        await send_message(session, chat_id, bot_response[i:i+4000])

async def main():
    if not MAX_BOT_TOKEN:
        logging.error("❌ Ошибка: Не задан MAX_BOT_TOKEN в переменных окружения!")
        return
        
    logging.info("🚀 Бот-помощник по физике запущен на Hugging Face Spaces!")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                await poll_updates(session)
            except Exception as e:
                logging.error(f"⚠️ Ошибка цикла опроса: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
