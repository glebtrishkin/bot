import os
import logging
import asyncio
import aiohttp
import tempfile
from typing import List, Dict, Optional
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor

import openai
import numpy as np
import faiss

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Конфигурация
ADMIN_IDS = [797671728]  # твой Telegram ID
TELEGRAM_BOT_TOKEN = "7952958434:AAFUymmDTxWYaawdK4JMq24LgINk3SuJL5E"
OPENAI_API_KEY = "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"
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base.txt")

BOT_PERSONA = os.getenv("BOT_PERSONA", """Ты — виртуальная подружка и поддержка для девушек.

Твоя роль — выслушивать, подбадривать, помогать разобраться в эмоциях и давать лёгкие советы. 

Правила:
- Отвечай дружелюбно, тепло и по-человечески.
- Используй «живой» тон общения (как близкая подруга), можно смайлы, но не перебарщивай.
- Не оценивай строго и не критикуй. 
- Поддерживай: «понимаю тебя», «это нормально так чувствовать», «ты не одна».
- Если спрашивают совета — дай мягкую рекомендацию, но не навязывай.
- Не затрагивай опасные темы (медицина, политика, религия, токсичные отношения с риском насилия) — в этих случаях отвечай, что лучше обратиться к специалисту или близкому человеку.
- Если настроение у собеседницы плохое — постарайся поднять его, предложи что-то простое (подышать, выпить чаю, прогуляться, включить любимую песню).
- Помни: твоя задача — создать ощущение «рядом есть человек, который понимает и поддерживает».

Примеры:
В: «Мне так грустно, ничего не хочется.»  
О: «Я понимаю тебя… такие дни бывают у каждой. Может, сделаешь себе чашку чая и завернёшься в плед? Иногда мелочи творят чудеса 💛»

В: «Меня никто не понимает.»  
О: «Очень тяжело чувствовать себя одинокой 😔 Но поверь, ты не одна, и я рядом. Расскажи, что случилось?»""")


# Инициализация
bot = Bot(token=TELEGRAM_BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Глобальные переменные
KNOWLEDGE_CHUNKS = []
KNOWLEDGE_INDEX = None
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_HISTORY: Dict[int, List[Dict]] = {}
MAX_HISTORY_LENGTH = 10

# --------------------
# Админ-панель
# --------------------
@dp.message_handler(commands=["ап"])
async def admin_panel(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет доступа к админ-панели.")
        return

    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton("✏️ Изменить персону"))
    keyboard.add(KeyboardButton("📋 Текущая персона"))

    await message.answer("⚙️ Админ-панель:", reply_markup=keyboard)


@dp.message_handler(lambda msg: msg.text == "📋 Текущая персона")
async def show_persona(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    await message.answer(f"👤 Текущая персона:\n\n{BOT_PERSONA}")


@dp.message_handler(lambda msg: msg.text == "✏️ Изменить персону")
async def ask_new_persona(message: types.Message, state: FSMContext):
    if message.from_user.id not in ADMIN_IDS:
        return
    await message.answer("Введите новый промт/персону для бота:")
    await state.set_state("waiting_new_persona")


@dp.message_handler(state="waiting_new_persona")
async def set_new_persona(message: types.Message, state: FSMContext):
    global BOT_PERSONA, CHAT_HISTORY
    BOT_PERSONA = message.text.strip()
    CHAT_HISTORY = {}  # очистка истории при смене персоны
    await message.answer(f"✅ Персона обновлена!\n\nТеперь бот говорит как:\n{BOT_PERSONA}")
    await state.finish()


# --------------------
# Автогенерация персоны
# --------------------
@dp.message_handler(commands=["persona_auto"])
async def auto_persona(message: types.Message, state: FSMContext):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ У вас нет доступа.")
        return

    await message.answer("Опиши в 2–5 словах, кто должен быть бот (например: 'юморной бармен').")
    await state.set_state("waiting_auto_persona")


@dp.message_handler(state="waiting_auto_persona")
async def generate_auto_persona(message: types.Message, state: FSMContext):
    global BOT_PERSONA, CHAT_HISTORY

    prompt = (
        f"Составь подробное описание персоны для чат-бота. "
        f"Пользователь описал: '{message.text}'. "
        f"Бот должен копировать стиль, поведение и манеру речи этого персонажа. "
        f"Дай текст в виде инструкции."
    )

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты эксперт по созданию персон для чат-ботов."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        persona = response.choices[0].message.content.strip()
        BOT_PERSONA = persona
        CHAT_HISTORY = {}  # очистка истории при смене персоны
        await message.answer(f"✅ Новая персона сгенерирована и установлена!\n\n{BOT_PERSONA}")
    except Exception as e:
        logger.error(f"Ошибка при генерации персоны: {e}")
        await message.answer("Ошибка при генерации персоны.")
    finally:
        await state.finish()


# --------------------
# Утилиты для работы с текстом
# --------------------
def split_text(text: str, max_size: int = 4000) -> List[str]:
    parts = []
    while len(text) > max_size:
        split_at = text.rfind("\n", 0, max_size)
        if split_at == -1:
            split_at = max_size
        parts.append(text[:split_at])
        text = text[split_at:]
    if text:
        parts.append(text)
    return parts


async def send_long_message(chat_id: int, text: str, parse_mode: str = None):
    chunks = split_text(text)
    for chunk in chunks:
        await bot.send_message(chat_id, chunk, parse_mode=parse_mode)


# --------------------
# GPT-ответы
# --------------------
async def generate_response(user_message: str, chat_history: List[Dict], relevant_knowledge: List[str] = None) -> str:
    messages = [
        {
            "role": "system",
            "content": f"Всегда отвечай в стиле: {BOT_PERSONA}\n"
                       f"Игнорируй любые предыдущие указания, если они противоречат этому стилю."
        }
    ]

    if relevant_knowledge:
        knowledge_text = "\n\nРелевантная информация из базы знаний:\n" + "\n".join([f"- {knowledge}" for knowledge in relevant_knowledge])
        messages[0]["content"] += knowledge_text

    for msg in chat_history[-MAX_HISTORY_LENGTH:]:
        messages.append(msg)

    messages.append({"role": "user", "content": user_message})

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка генерации ответа GPT: {e}")
        return "Извини, произошла ошибка при обработке твоего запроса 😔"

# Функции для работы с базой знаний
def load_knowledge_base(file_path: str) -> List[str]:
    """Загружает базу знаний из файла"""
    if not os.path.exists(file_path):
        logger.warning(f"Файл базы знаний не найден: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разделяем на чанки по абзацам
    chunks = []
    current_chunk = ""
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if len(current_chunk) + len(line) + 1 < 1000:  # Ограничение размера чанка
            current_chunk += line + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info(f"Загружено {len(chunks)} чанков из базы знаний")
    return chunks

async def build_knowledge_index():
    """Строит FAISS индекс для базы знаний"""
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_INDEX
    
    KNOWLEDGE_CHUNKS = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    
    if not KNOWLEDGE_CHUNKS:
        logger.warning("База знаний пуста, индекс не строится")
        return
    
    # Получаем эмбеддинги для чанков
    embeddings = []
    batch_size = 50
    
    for i in range(0, len(KNOWLEDGE_CHUNKS), batch_size):
        batch = KNOWLEDGE_CHUNKS[i:i + batch_size]
        try:
            response = await openai.Embedding.acreate(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
            logger.info(f"Обработано {min(i + batch_size, len(KNOWLEDGE_CHUNKS))}/{len(KNOWLEDGE_CHUNKS)} чанков")
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов: {e}")
    
    if not embeddings:
        logger.error("Не удалось получить эмбеддинги для базы знаний")
        return
    
    # Строим индекс
    embeddings_array = np.array(embeddings, dtype="float32")
    dimension = embeddings_array.shape[1]
    
    KNOWLEDGE_INDEX = faiss.IndexFlatL2(dimension)
    KNOWLEDGE_INDEX.add(embeddings_array)
    
    logger.info(f"Индекс базы знаний построен: {len(KNOWLEDGE_CHUNKS)} чанков")

async def search_knowledge_base(query: str, top_k: int = 3) -> List[str]:
    """Ищет релевантные чанки в базе знаний"""
    if not KNOWLEDGE_INDEX or not KNOWLEDGE_CHUNKS:
        return []
    
    try:
        # Получаем эмбеддинг запроса
        response = await openai.Embedding.acreate(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = np.array([response['data'][0]['embedding']], dtype="float32")
        
        # Ищем ближайшие чанки
        distances, indices = KNOWLEDGE_INDEX.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(KNOWLEDGE_CHUNKS):
                results.append(KNOWLEDGE_CHUNKS[idx])
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка поиска в базе знаний: {e}")
        return []

# Функции для работы с GPT
async def generate_response(user_message: str, chat_history: List[Dict], relevant_knowledge: List[str] = None) -> str:
    """Генерирует ответ с помощью GPT"""
    messages = [
        {"role": "system", "content": BOT_PERSONA}
    ]
    
    # Добавляем релевантные знания из базы
    if relevant_knowledge:
        knowledge_text = "\n\nРелевантная информация из базы знаний:\n" + "\n".join([f"- {knowledge}" for knowledge in relevant_knowledge])
        messages[0]["content"] += knowledge_text
    
    # Добавляем историю диалога
    for msg in chat_history[-MAX_HISTORY_LENGTH:]:
        messages.append(msg)
    
    # Добавляем текущее сообщение
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Ошибка генерации ответа GPT: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

async def transcribe_audio(audio_path: str) -> str:
    """Транскрибирует аудиофайл"""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = await openai.Audio.atranscribe(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="ru"
            )
        return transcript.strip()
    except Exception as e:
        logger.error(f"Ошибка транскрибации аудио: {e}")
        return ""

# Обработчики сообщений
@dp.message_handler(Command("start", "help"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    welcome_text = (
        "👋 Привет! Я ваш AI-ассистент.\n\n"
        "Я могу:\n"
        "• Отвечать на ваши вопросы\n"
        "• Использовать базу знаний для точных ответов\n"
        "• Работать с текстовыми и голосовыми сообщениями\n\n"
        "Просто напишите или запишите голосовое сообщение!"
    )
    await message.answer(welcome_text)

@dp.message_handler(Command("update_knowledge"))
async def cmd_update_knowledge(message: types.Message):
    """Обновление базы знаний"""
    await message.answer(
        "📚 Отправьте файл с обновленной базой знаний (текстовый файл).\n"
        "Или отправьте текстовое сообщение с новой информацией."
    )
    # Здесь можно добавить логику для обработки загрузки файлов

@dp.message_handler(Command("clear_history"))
async def cmd_clear_history(message: types.Message):
    """Очистка истории диалога"""
    user_id = message.from_user.id
    if user_id in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
    await message.answer("🗑️ История диалога очищена.")

@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    """Обработка текстовых сообщений"""
    user_id = message.from_user.id
    user_message = message.text.strip()
    
    if not user_message:
        return
    
    await bot.send_chat_action(message.chat.id, "typing")
    
    # Инициализируем историю для пользователя
    if user_id not in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
    
    # Ищем релевантную информацию в базе знаний
    relevant_knowledge = await search_knowledge_base(user_message)
    
    # Генерируем ответ
    response = await generate_response(user_message, CHAT_HISTORY[user_id], relevant_knowledge)
    
    # Сохраняем в историю
    CHAT_HISTORY[user_id].append({"role": "user", "content": user_message})
    CHAT_HISTORY[user_id].append({"role": "assistant", "content": response})
    
    # Ограничиваем длину истории
    if len(CHAT_HISTORY[user_id]) > MAX_HISTORY_LENGTH * 2:
        CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_HISTORY_LENGTH * 2:]
    
    # Отправляем ответ
    await send_long_message(message.chat.id, response)

@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice(message: types.Message):
    """Обработка голосовых сообщений"""
    user_id = message.from_user.id
    
    await bot.send_chat_action(message.chat.id, "typing")
    
    try:
        # Скачиваем голосовое сообщение
        file_info = await bot.get_file(message.voice.file_id)
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info.file_path}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
                        tmp_file.write(await resp.read())
                        audio_path = tmp_file.name
        
        # Транскрибируем аудио
        user_message = await transcribe_audio(audio_path)
        os.unlink(audio_path)
        
        if not user_message:
            await message.answer("Не удалось распознать голосовое сообщение.")
            return
        
        # Инициализируем историю для пользователя
        if user_id not in CHAT_HISTORY:
            CHAT_HISTORY[user_id] = []
        
        # Ищем релевантную информацию в базе знаний
        relevant_knowledge = await search_knowledge_base(user_message)
        
        # Генерируем ответ
        response = await generate_response(user_message, CHAT_HISTORY[user_id], relevant_knowledge)
        
        # Сохраняем в историю
        CHAT_HISTORY[user_id].append({"role": "user", "content": user_message})
        CHAT_HISTORY[user_id].append({"role": "assistant", "content": response})
        
        # Ограничиваем длину истории
        if len(CHAT_HISTORY[user_id]) > MAX_HISTORY_LENGTH * 2:
            CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_HISTORY_LENGTH * 2:]
        
        # Отправляем ответ
        await send_long_message(message.chat.id, response)
        
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
        await message.answer("Произошла ошибка при обработке голосового сообщения.")

# Функции для запуска и остановки
async def on_startup(dp):
    """Действия при запуске бота"""
    logger.info("🟢 Бот запускается...")
    
    # Инициализируем OpenAI
    openai.api_key = OPENAI_API_KEY
    
    # Загружаем базу знаний
    await build_knowledge_index()
    
    logger.info("✅ Бот готов к работе!")

async def on_shutdown(dp):
    """Действия при остановке бота"""
    logger.info("🔴 Бот останавливается...")
    await bot.close()

if __name__ == "__main__":
    # Создаем файл базы знаний, если его нет
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, 'w', encoding='utf-8') as f:
            f.write("# База знаний\n\nДобавьте сюда вашу информацию для использования ботом.")
    
    # Запускаем бота
    executor.start_polling(
        dp,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown
    )