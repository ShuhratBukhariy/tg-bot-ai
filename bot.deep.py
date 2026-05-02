import asyncio
import base64
import json
import logging
import os
import time

import aiohttp
import aiosqlite
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_HOST = os.getenv(
    "RAPID_HOST",
    "cheapest-gpt-4-turbo-gpt-4-vision-chatgpt-openai-ai-api.p.rapidapi.com",
)
RAPID_ENDPOINT = os.getenv("RAPID_ENDPOINT", "/v1/chat/completions")
RAPID_MODEL = os.getenv("RAPID_MODEL", "gpt-4o")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
DB_PATH = os.getenv("DB_PATH", "bot.db")

DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "5"))
ANALYSIS_TTL = int(os.getenv("ANALYSIS_TTL", "3600"))
MAX_PHOTO_BYTES = 5 * 1024 * 1024

if not TOKEN or not RAPID_API_KEY:
    raise RuntimeError("BOT_TOKEN va RAPID_API_KEY .env faylda berilgan bo'lishi shart")

dp = Dispatcher()
analysis_cache: dict[int, tuple[float, list]] = {}

PROMPT = """Analyze this person's outfit and propose 4 distinct styles. Respond with ONLY valid JSON in this exact schema:
{
  "styles": [
    {"name": "Casual",     "description": "...", "clothes": ["..."], "stores": [{"name": "Uzum",         "price": "150000 UZS", "link": ""}]},
    {"name": "Formal",     "description": "...", "clothes": ["..."], "stores": [{"name": "Zara",         "price": "...",        "link": ""}]},
    {"name": "Sport",      "description": "...", "clothes": ["..."], "stores": [{"name": "Nike",         "price": "...",        "link": ""}]},
    {"name": "Streetwear", "description": "...", "clothes": ["..."], "stores": [{"name": "Wildberries",  "price": "...",        "link": ""}]}
  ]
}
Use realistic prices in UZS for the Uzbekistan market. Allowed stores: Uzum, Wildberries, Zara, Nike, Adidas. If you don't know a real product URL, use an empty string for "link" — do not invent fake links."""


async def init_db() -> None:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "user_id INTEGER PRIMARY KEY, "
            "full_name TEXT, "
            "created_at INTEGER)"
        )
        await db.execute(
            "CREATE TABLE IF NOT EXISTS requests ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id INTEGER, "
            "created_at INTEGER)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_user_time "
            "ON requests (user_id, created_at)"
        )
        await db.commit()


async def register_user(user_id: int, full_name: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO users (user_id, full_name, created_at) VALUES (?, ?, ?)",
            (user_id, full_name, int(time.time())),
        )
        await db.commit()


async def remaining_today(user_id: int) -> int:
    day_start = int(time.time()) - 86400
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM requests WHERE user_id = ? AND created_at > ?",
            (user_id, day_start),
        ) as cur:
            row = await cur.fetchone()
    used = row[0] if row else 0
    return max(DAILY_LIMIT - used, 0)


async def log_request(user_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO requests (user_id, created_at) VALUES (?, ?)",
            (user_id, int(time.time())),
        )
        await db.commit()


def cache_set(user_id: int, styles: list) -> None:
    analysis_cache[user_id] = (time.time(), styles)


def cache_get(user_id: int):
    entry = analysis_cache.get(user_id)
    if not entry:
        return None
    ts, styles = entry
    if time.time() - ts > ANALYSIS_TTL:
        analysis_cache.pop(user_id, None)
        return None
    return styles


def cache_cleanup() -> None:
    now = time.time()
    expired = [uid for uid, (ts, _) in analysis_cache.items() if now - ts > ANALYSIS_TTL]
    for uid in expired:
        analysis_cache.pop(uid, None)


async def analyze_image(session: aiohttp.ClientSession, image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = f"https://{RAPID_HOST}{RAPID_ENDPOINT}"
    headers = {
        "x-rapidapi-key": RAPID_API_KEY,
        "x-rapidapi-host": RAPID_HOST,
        "Content-Type": "application/json",
    }
    payload = {
        "model": RAPID_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.5,
    }

    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=45),
        ) as res:
            if res.status != 200:
                logging.error("API error %s: %s", res.status, await res.text())
                return None
            data = await res.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        logging.exception("API request failed")
        return None

    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        logging.exception("Unexpected API response shape: %s", data)
        return None

    if "```" in content:
        try:
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        except IndexError:
            pass

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logging.exception("JSON parse failed: %s", content[:300])
        return None


def style_keyboard(styles: list) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"👔 {s.get('name', f'Style {i+1}')}", callback_data=f"style_{i}")]
            for i, s in enumerate(styles)
        ]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Boshqa imijni ko'rish", callback_data="back_to_styles")]
        ]
    )


def render_summary(styles: list) -> str:
    out = "🎨 <b>Sizga mos 4 xil imij:</b>\n\n"
    for i, s in enumerate(styles, 1):
        name = html.quote(str(s.get("name", f"Style {i}")))
        desc = str(s.get("description") or "")
        short = html.quote(desc[:120])
        suffix = "..." if len(desc) > 120 else ""
        out += f"<b>{i}. {name}</b> — {short}{suffix}\n\n"
    out += "👇 Quyidagi tugmadan birini tanlang:"
    return out


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await register_user(message.from_user.id, message.from_user.full_name)
    await message.answer(
        f"Salom, {html.bold(html.quote(message.from_user.full_name))} 👋\n\n"
        f"📸 Iltimos, kiyimingizdagi rasmni yuboring.\n"
        f"Men sizga 4 xil imij va har birining narxini aytaman 🛍️\n\n"
        f"ℹ️ Kuniga {DAILY_LIMIT} ta rasm tahlil qilishingiz mumkin.\n"
        f"/help — yordam"
    )


@dp.message(Command("help"))
async def help_cmd(message: Message) -> None:
    await message.answer(
        "🤖 <b>Foydalanish:</b>\n\n"
        "1. Kiyimingizdagi rasm yuboring\n"
        "2. Bot 4 xil imij taklif qiladi\n"
        "3. Yoqqan imijni tanlang\n"
        "4. Narxlar va do'konlar ro'yxatini oling\n\n"
        "<b>Komandalar:</b>\n"
        "/start — boshlash\n"
        "/help — yordam\n"
        "/limit — qolgan kunlik limit"
    )


@dp.message(Command("limit"))
async def limit_cmd(message: Message) -> None:
    left = await remaining_today(message.from_user.id)
    await message.answer(f"📊 Bugun qolgan limit: <b>{left}</b> / {DAILY_LIMIT}")


@dp.message(Command("stats"))
async def stats_cmd(message: Message) -> None:
    if message.from_user.id != ADMIN_ID:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT COUNT(*) FROM users") as cur:
            users = (await cur.fetchone())[0]
        async with db.execute(
            "SELECT COUNT(*) FROM requests WHERE created_at > ?",
            (int(time.time()) - 86400,),
        ) as cur:
            today = (await cur.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM requests") as cur:
            total = (await cur.fetchone())[0]
    await message.answer(
        "📊 <b>Statistika</b>\n\n"
        f"👤 Foydalanuvchilar: <b>{users}</b>\n"
        f"📸 Bugungi tahlillar: <b>{today}</b>\n"
        f"📈 Jami tahlillar: <b>{total}</b>"
    )


@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    user_id = message.from_user.id
    await register_user(user_id, message.from_user.full_name)

    if await remaining_today(user_id) <= 0:
        await message.answer(
            f"⚠️ Kunlik limit ({DAILY_LIMIT}) tugadi. Ertaga qaytib keling 🙏"
        )
        return

    processing = await message.answer("⏳ Rasm tahlil qilinmoqda... (10–20 soniya)")

    try:
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > MAX_PHOTO_BYTES:
            await processing.edit_text("❌ Rasm hajmi juda katta (max 5MB).")
            return

        try:
            file = await message.bot.get_file(photo.file_id)
        except Exception:
            logging.exception("get_file failed")
            await processing.edit_text("❌ Rasmni Telegram'dan yuklab bo'lmadi.")
            return

        file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file.file_path}"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    file_url, timeout=aiohttp.ClientTimeout(total=20)
                ) as r:
                    if r.status != 200:
                        await processing.edit_text("❌ Rasmni yuklab bo'lmadi.")
                        return
                    image_bytes = await r.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await processing.edit_text("❌ Rasm yuklanmadi. Qayta urinib ko'ring.")
                return

            result = await analyze_image(session, image_bytes)

        if not result or not isinstance(result.get("styles"), list) or not result["styles"]:
            await processing.edit_text(
                "❌ Tahlil qilishda xatolik. Boshqa rasm bilan urinib ko'ring."
            )
            return

        styles = result["styles"][:4]
        cache_set(user_id, styles)
        await log_request(user_id)

        await processing.edit_text(render_summary(styles), reply_markup=style_keyboard(styles))
    except Exception:
        logging.exception("photo_handler crashed")
        try:
            await processing.edit_text("❌ Kutilmagan xatolik. Qayta urinib ko'ring.")
        except Exception:
            pass


@dp.callback_query(F.data.startswith("style_"))
async def style_selection(callback: CallbackQuery) -> None:
    styles = cache_get(callback.from_user.id)
    if not styles:
        await callback.answer("⏳ Sessiya muddati tugadi", show_alert=True)
        await callback.message.edit_text(
            "❌ Ma'lumotlar topilmadi. Iltimos, qaytadan rasm yuboring."
        )
        return

    try:
        idx = int(callback.data.split("_", 1)[1])
        style = styles[idx]
    except (ValueError, IndexError):
        await callback.answer("Xato", show_alert=True)
        return

    name = html.quote(str(style.get("name", "Style")))
    desc = html.quote(str(style.get("description") or "—"))

    parts = [
        f"✨ <b>{name} imiji</b> ✨\n",
        f"📝 <b>Tavsif:</b>\n{desc}\n",
        "👕 <b>Kiyimlar:</b>",
    ]
    for item in style.get("clothes", []) or []:
        parts.append(f"• {html.quote(str(item))}")

    parts.append("\n🛍️ <b>Qayerdan sotib olish mumkin:</b>")
    for store in style.get("stores", []) or []:
        s_name = html.quote(str(store.get("name", "—")))
        s_price = html.quote(str(store.get("price", "—")))
        s_link = str(store.get("link") or "").strip()
        parts.append(f"\n<b>{s_name}</b>")
        parts.append(f"💰 Narxi: {s_price}")
        if s_link.startswith(("http://", "https://")):
            parts.append(f'🔗 <a href="{html.quote(s_link)}">Xarid qilish</a>')

    parts.append("\n💡 <i>Narxlar taxminiy. Aniq narxni do'kondan tekshiring.</i>")

    await callback.message.edit_text(
        "\n".join(parts),
        reply_markup=back_keyboard(),
        disable_web_page_preview=True,
    )
    await callback.answer(f"{style.get('name', '')} tanlandi")


@dp.callback_query(F.data == "back_to_styles")
async def back_to_styles(callback: CallbackQuery) -> None:
    styles = cache_get(callback.from_user.id)
    if not styles:
        await callback.answer("Sessiya tugadi", show_alert=True)
        await callback.message.edit_text("❌ Iltimos, qaytadan rasm yuboring.")
        return

    await callback.message.edit_text(render_summary(styles), reply_markup=style_keyboard(styles))
    await callback.answer()


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    await message.answer(
        "📸 Iltimos, kiyimingizdagi rasm yuboring. /help — yordam."
    )


async def cleanup_loop() -> None:
    while True:
        await asyncio.sleep(600)
        cache_cleanup()


async def main() -> None:
    await init_db()
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    asyncio.create_task(cleanup_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
