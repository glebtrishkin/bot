import os
import logging
import asyncio
import asyncpg
import aiohttp
import tempfile
from typing import List, Dict, Optional
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils import executor
from aiogram.dispatcher.filters.state import State, StatesGroup

class KnowledgeBaseStates(StatesGroup):
    waiting_new_doc_name = State()
    waiting_new_doc_content = State()
    waiting_edit_doc = State()

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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "knowledge_base.txt")
DOCS_DIR = "knowledge_base"
DB_URL = os.getenv("DATABASE_URL")

BOT_PERSONA = os.getenv("BOT_PERSONA", """Ты — виртуальная подружка и поддержка для девушек.

Твоя роль — выслушивать, подбадривать, помогать разобраться в эмоциях и давать лёгкие советы. 


Правила:

- Не начинай каждый ответ с приветствия. Отвечай по сути вопроса, а приветствуй только при первом сообщении от пользователя.
- Старайся не повторять в каждом сообщении одни и те-же слова
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
    keyboard.add(KeyboardButton("➕ Добавить в базу знаний"))
    keyboard.add(KeyboardButton("📚 Управление базой знаний"))

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

DOCS_DIR = "knowledge_base"

# --------------------
# Админ-панель: управление базой знаний
# --------------------

@dp.message_handler(lambda msg: msg.text == "📚 Управление базой знаний")
async def kb_manage(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return

    files = list_docs()
    if not files:
        await message.answer("⚠️ В базе знаний пока нет документов.")
        return

    kb = InlineKeyboardMarkup()
    for fname in files:
        kb.add(InlineKeyboardButton(fname, callback_data=f"view_doc:{fname}"))

    await message.answer("📚 Документы базы знаний:", reply_markup=kb)

@dp.callback_query_handler(lambda c: c.data.startswith("view_doc:"))
async def kb_view(callback: types.CallbackQuery):
    fname = callback.data.split(":", 1)[1]
    content = read_doc(fname)

    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton("✏️ Редактировать", callback_data=f"edit_doc:{fname}"))
    kb.add(InlineKeyboardButton("❌ Удалить", callback_data=f"del_doc:{fname}"))

    await callback.message.answer(
        f"📄 *{fname}*:\n\n{content}",
        parse_mode="Markdown",
        reply_markup=kb
    )

@dp.callback_query_handler(lambda c: c.data.startswith("del_doc:"))
async def kb_delete(callback: types.CallbackQuery):
    fname = callback.data.split(":", 1)[1]
    delete_doc(fname)
    await callback.message.answer(f"🗑 Документ *{fname}* удалён.", parse_mode="Markdown")
    await build_knowledge_index()

@dp.callback_query_handler(lambda c: c.data.startswith("edit_doc:"))
async def kb_edit(callback: types.CallbackQuery, state: FSMContext):
    fname = callback.data.split(":", 1)[1]
    await callback.message.answer(f"✏️ Введите новый текст для документа *{fname}*:", parse_mode="Markdown")
    await state.update_data(editing_doc=fname)
    await state.set_state("waiting_edit_doc")

@dp.message_handler(state="waiting_edit_doc")
async def kb_edit_save(message: types.Message, state: FSMContext):
    data = await state.get_data()
    fname = data.get("editing_doc")
    write_doc(fname, message.text)

    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton("💾 Сохранить изменения"))

    await message.answer(
        f"✅ Документ *{fname}* сохранён.\n\nНажмите «💾 Сохранить изменения», чтобы обновить индекс.",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    await state.finish()

# --------------------
# Добавление нового документа
# --------------------

@dp.message_handler(lambda msg: msg.text == "➕ Добавить в базу знаний")
async def kb_add(message: types.Message, state: FSMContext):
    if message.from_user.id not in ADMIN_IDS:
        return
    await message.answer("📗 Введите название нового документа:")
    await KnowledgeBaseStates.waiting_new_doc_name.set()


@dp.message_handler(state=KnowledgeBaseStates.waiting_new_doc_name)
async def kb_add_name(message: types.Message, state: FSMContext):
    fname = message.text.strip().replace(" ", "_") + ".txt"
    await state.update_data(new_doc_name=fname)
    await message.answer("✏️ Теперь введите содержимое документа:")
    await KnowledgeBaseStates.waiting_new_doc_content.set()


@dp.message_handler(state=KnowledgeBaseStates.waiting_new_doc_content)
async def kb_add_content(message: types.Message, state: FSMContext):
    data = await state.get_data()
    fname = data.get("new_doc_name")
    write_doc(fname, message.text)

    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(KeyboardButton("💾 Сохранить изменения"))

    await message.answer(
        f"✅ Документ *{fname}* сохранён.\n\nНажмите «💾 Сохранить изменения», чтобы обновить индекс.",
        parse_mode="Markdown",
        reply_markup=keyboard
    )

    await state.finish()


@dp.message_handler(lambda msg: msg.text == "💾 Сохранить изменения")
async def kb_rebuild_index(message: types.Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    await message.answer("⏳ Перестраиваю индекс...")
    await build_knowledge_index()
    await message.answer("✅ Индекс базы знаний обновлён!")


# --------------------
# Обновление индекса для FAISS
# --------------------

async def build_knowledge_index():
    """Перестраивает индекс по всем документам"""
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_INDEX
    KNOWLEDGE_CHUNKS = []
    KNOWLEDGE_INDEX = None

    files = list_docs()
    if not files:
        logger.warning("Нет документов в базе знаний.")
        return

    # собираем все чанки из документов
    for fname in files:
        content = read_doc(fname)
        if content:
            KNOWLEDGE_CHUNKS.append(content)

    if not KNOWLEDGE_CHUNKS:
        logger.warning("Документы пустые, индекс не построен.")
        return

    embeddings = []
    for i in range(0, len(KNOWLEDGE_CHUNKS), 50):
        batch = KNOWLEDGE_CHUNKS[i:i + 50]
        try:
            response = await openai.Embedding.acreate(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Ошибка эмбеддинга: {e}")

    if not embeddings:
        return

    embeddings_array = np.array(embeddings, dtype="float32")
    KNOWLEDGE_INDEX = faiss.IndexFlatL2(embeddings_array.shape[1])
    KNOWLEDGE_INDEX.add(embeddings_array)
    logger.info(f"Индекс построен: {len(KNOWLEDGE_CHUNKS)} документов")


async def init_db():
    conn = await asyncpg.connect(DB_URL)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id SERIAL PRIMARY KEY,
            title TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL
        )
    """)
    await conn.close()

async def add_doc(title: str, content: str):
    conn = await asyncpg.connect(DB_URL)
    await conn.execute("""
        INSERT INTO knowledge_base (title, content)
        VALUES ($1, $2)
        ON CONFLICT (title) DO UPDATE SET content = EXCLUDED.content
    """, title, content)
    await conn.close()

async def list_docs():
    conn = await asyncpg.connect(DB_URL)
    rows = await conn.fetch("SELECT title FROM knowledge_base ORDER BY id")
    await conn.close()
    return [r["title"] for r in rows]

async def read_doc(title: str) -> str:
    conn = await asyncpg.connect(DB_URL)
    row = await conn.fetchrow("SELECT content FROM knowledge_base WHERE title=$1", title)
    await conn.close()
    return row["content"] if row else ""

async def delete_doc(title: str):
    conn = await asyncpg.connect(DB_URL)
    await conn.execute("DELETE FROM knowledge_base WHERE title=$1", title)
    await conn.close()
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
    f"Создай максимально детальное и индивидуальное описание персоны для чат-бота. "
    f"Пользователь описал: '{message.text}'. "
    f"\n\nТребования:\n"
    f"- Используй все вводные данные пользователя, ничего не упускай.\n"
    f"- Сделай описание максимально подробным: манера речи, лексика, стиль общения, поведение, характерные выражения.\n"
    f"- Обязательно укажи примеры типичных фраз и способов общения.\n"
    f"- Если персона реальная (например, историческая личность, известный человек), найди информацию в интернете и используй её.\n"
    f"- Персона должна быть уникальной, индивидуальной и сразу отличимой от других.\n"
    f"- Результат оформи как чёткую инструкцию для чат-бота."
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
        await message.answer(
            f"✅ Новая персона установлена!\n\n"
            f"👤 *Название:* {persona_name}\n\n"
            f"📜 *Описание:*\n{BOT_PERSONA}",
            parse_mode="Markdown"
        )
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
# Работа с документами базы знаний
# --------------------

DOCS_DIR = "knowledge_base"

def ensure_docs_dir():
    """Создает папку для базы знаний, если её нет"""
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

def list_docs() -> list:
    """Возвращает список документов"""
    ensure_docs_dir()
    return [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]

def read_doc(fname: str) -> str:
    """Читает документ"""
    path = os.path.join(DOCS_DIR, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_doc(fname: str, content: str):
    """Создает или перезаписывает документ"""
    ensure_docs_dir()
    with open(os.path.join(DOCS_DIR, fname), "w", encoding="utf-8") as f:
        f.write(content.strip())

def delete_doc(fname: str):
    """Удаляет документ"""
    path = os.path.join(DOCS_DIR, fname)
    if os.path.exists(path):
        os.remove(path)

# --------------------
# Построение FAISS индекса по документам
# --------------------

async def build_knowledge_index():
    """Перестраивает индекс по всем документам"""
    global KNOWLEDGE_CHUNKS, KNOWLEDGE_INDEX
    KNOWLEDGE_CHUNKS = []
    KNOWLEDGE_INDEX = None

    files = list_docs()
    if not files:
        logger.warning("Нет документов в базе знаний.")
        return

    # собираем все тексты
    for fname in files:
        content = read_doc(fname)
        if content:
            KNOWLEDGE_CHUNKS.append(content)

    if not KNOWLEDGE_CHUNKS:
        logger.warning("Документы пустые, индекс не построен.")
        return

    embeddings = []
    for i in range(0, len(KNOWLEDGE_CHUNKS), 50):
        batch = KNOWLEDGE_CHUNKS[i:i + 50]
        try:
            response = await openai.Embedding.acreate(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Ошибка эмбеддинга: {e}")

    if not embeddings:
        return

    embeddings_array = np.array(embeddings, dtype="float32")
    KNOWLEDGE_INDEX = faiss.IndexFlatL2(embeddings_array.shape[1])
    KNOWLEDGE_INDEX.add(embeddings_array)
    logger.info(f"Индекс построен: {len(KNOWLEDGE_CHUNKS)} документов")


async def search_knowledge_base(query: str, top_k: int = 3) -> List[str]:
    if not KNOWLEDGE_INDEX or not KNOWLEDGE_CHUNKS:
        return []
    try:
        response = await openai.Embedding.acreate(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_embedding = np.array([response['data'][0]['embedding']], dtype="float32")
        distances, indices = KNOWLEDGE_INDEX.search(query_embedding, top_k)
        return [KNOWLEDGE_CHUNKS[idx] for idx in indices[0] if idx < len(KNOWLEDGE_CHUNKS)]
    except Exception as e:
        logger.error(f"Ошибка поиска в базе знаний: {e}")
        return []


# --------------------
# GPT-ответы (универсальная версия)
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


# --------------------
# Транскрибация аудио
# --------------------
async def transcribe_audio(audio_path: str) -> str:
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


# --------------------
# Обработчики сообщений
# --------------------
@dp.message_handler(Command("start", "help"))
async def cmd_start(message: types.Message):
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
    await message.answer(
        "📚 Отправьте файл с обновленной базой знаний (текстовый файл).\n"
        "Или отправьте текстовое сообщение с новой информацией."
    )


@dp.message_handler(Command("clear_history"))
async def cmd_clear_history(message: types.Message):
    user_id = message.from_user.id
    if user_id in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
    await message.answer("🗑️ История диалога очищена.")


@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    user_id = message.from_user.id
    user_message = message.text.strip()
    if not user_message:
        return
    await bot.send_chat_action(message.chat.id, "typing")
    if user_id not in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
    relevant_knowledge = await search_knowledge_base(user_message)
    response = await generate_response(user_message, CHAT_HISTORY[user_id], relevant_knowledge)
    CHAT_HISTORY[user_id].append({"role": "user", "content": user_message})
    CHAT_HISTORY[user_id].append({"role": "assistant", "content": response})
    if len(CHAT_HISTORY[user_id]) > MAX_HISTORY_LENGTH * 2:
        CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_HISTORY_LENGTH * 2:]
    await send_long_message(message.chat.id, response)


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice(message: types.Message):
    user_id = message.from_user.id
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        file_info = await bot.get_file(message.voice.file_id)
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_info.file_path}"
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url) as resp:
                if resp.status == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
                        tmp_file.write(await resp.read())
                        audio_path = tmp_file.name
        user_message = await transcribe_audio(audio_path)
        os.unlink(audio_path)
        if not user_message:
            await message.answer("Не удалось распознать голосовое сообщение.")
            return
        if user_id not in CHAT_HISTORY:
            CHAT_HISTORY[user_id] = []
        relevant_knowledge = await search_knowledge_base(user_message)
        response = await generate_response(user_message, CHAT_HISTORY[user_id], relevant_knowledge)
        CHAT_HISTORY[user_id].append({"role": "user", "content": user_message})
        CHAT_HISTORY[user_id].append({"role": "assistant", "content": response})
        if len(CHAT_HISTORY[user_id]) > MAX_HISTORY_LENGTH * 2:
            CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_HISTORY_LENGTH * 2:]
        await send_long_message(message.chat.id, response)
    except Exception as e:
        logger.error(f"Ошибка обработки голосового сообщения: {e}")
        await message.answer("Произошла ошибка при обработке голосового сообщения.")


# --------------------
# Функции для запуска и остановки
# --------------------
async def on_startup(dp):
    """Действия при запуске бота"""
    logger.info("🟢 Бот запускается...")

    # Инициализация OpenAI
    openai.api_key = OPENAI_API_KEY

    # Инициализация базы данных
    await init_db()

    # Загружаем базу знаний
    await build_knowledge_index()

    logger.info("✅ Бот готов к работе!")


if __name__ == "__main__":
    # Создаем папку базы знаний, если её нет (если остаёшься на файловом варианте)
    ensure_docs_dir()

    # Запускаем бота
    executor.start_polling(
        dp,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown
    )











