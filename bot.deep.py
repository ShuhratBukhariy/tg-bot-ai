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
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image-preview")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
DB_PATH = os.getenv("DB_PATH", "bot.db")

DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "5"))
ANALYSIS_TTL = int(os.getenv("ANALYSIS_TTL", "3600"))
MAX_PHOTO_BYTES = 5 * 1024 * 1024

if not TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("BOT_TOKEN va GEMINI_API_KEY .env faylda berilgan bo'lishi shart")

dp = Dispatcher()
analysis_cache: dict[int, tuple[float, list]] = {}

PROMPT = """Carefully analyze the person in the photo: their body type, build, height impression, skin tone, hair, age range, and gender if visible.

Then choose ONLY 2 to 4 fashion styles that would genuinely suit THIS specific person (do NOT include styles that don't match them). Available styles to pick from: Klassik, Casual, Sport, Streetwear, Biznes, Smart Casual, Bohemian, Formal.

For EACH chosen style, generate 2 or 3 distinct outfit VARIANTS. Each variant is a complete head-to-toe look with specific items.

Respond with ONLY valid JSON in this exact schema (no markdown, no extra text):
{
  "user_analysis": {
    "summary": "1-2 jumla o'zbek tilida: foydalanuvchining tashqi ko'rinishi va nima uchun aynan shu imijlar yarashadi"
  },
  "styles": [
    {
      "name": "Klassik",
      "match_reason": "1 jumla o'zbek tilida: nega aynan bu imij ushbu odamga mos keladi",
      "variants": [
        {
          "name": "Ofis ko'rinishi",
          "description": "O'zbek tilida batafsil tavsif",
          "clothes": ["Qora pidjak", "Oq ko'ylak", "Klassik shim", "Charm tufli"],
          "image_prompt": "navy blue tailored blazer, crisp white cotton dress shirt, charcoal grey wool trousers, black leather oxford shoes, leather belt",
          "stores": [
            {"name": "Zara", "price": "1 200 000 UZS", "link": ""},
            {"name": "Uzum", "price": "800 000 UZS", "link": ""}
          ]
        },
        {
          "name": "Kechki kiyim",
          "description": "...",
          "clothes": ["..."],
          "image_prompt": "English description of the outfit for AI image generation: specific colors, fabrics, item types, footwear, accessories",
          "stores": [{"name": "...", "price": "...", "link": ""}]
        }
      ]
    }
  ]
}

Rules:
- Allowed stores: Uzum, Wildberries, Zara, Nike, Adidas, H&M, Mango, Pull&Bear.
- Prices must be realistic for Uzbekistan market in UZS.
- ALL human-readable text (summary, match_reason, variant name, description, clothes) MUST be in Uzbek.
- Keep "link" as empty string. Do NOT invent URLs.
- The "image_prompt" field MUST be in ENGLISH. Describe the complete outfit visually with specific colors, fabrics and item types so an AI image generator can render it accurately on a mannequin.
- Do not wrap response in code blocks. Output raw JSON only."""


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


def cache_set(user_id: int, data: dict) -> None:
    analysis_cache[user_id] = (time.time(), data)


def cache_get(user_id: int):
    entry = analysis_cache.get(user_id)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > ANALYSIS_TTL:
        analysis_cache.pop(user_id, None)
        return None
    return data


def cache_cleanup() -> None:
    now = time.time()
    expired = [uid for uid, (ts, _) in analysis_cache.items() if now - ts > ANALYSIS_TTL]
    for uid in expired:
        analysis_cache.pop(uid, None)


async def analyze_image(session: aiohttp.ClientSession, image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    url = f"{GEMINI_BASE}/{GEMINI_VISION_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.5,
            "maxOutputTokens": 8192,
        },
    }

    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as res:
            if res.status != 200:
                logging.error("Gemini API error %s: %s", res.status, await res.text())
                return None
            data = await res.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        logging.exception("Gemini API request failed")
        return None

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        logging.error("Unknown Gemini response: %s", str(data)[:500])
        return None

    text = text.strip()
    if "```" in text:
        try:
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        except IndexError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logging.exception("JSON parse failed: %s", text[:300])
        return None


async def generate_outfit_image(session: aiohttp.ClientSession, prompt: str):
    url = f"{GEMINI_BASE}/{GEMINI_IMAGE_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        },
    }

    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as res:
            if res.status != 200:
                logging.error("Gemini image API error %s: %s", res.status, await res.text())
                return None
            data = await res.json()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        logging.exception("Gemini image API request failed")
        return None

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        logging.error("No candidates in Gemini image response: %s", str(data)[:500])
        return None

    for part in parts:
        inline = part.get("inline_data") or part.get("inlineData")
        if isinstance(inline, dict) and inline.get("data"):
            try:
                return base64.b64decode(inline["data"])
            except (ValueError, TypeError):
                logging.exception("Invalid base64 in image response")
                return None

    logging.error("No image data in Gemini parts: %s", str(data)[:500])
    return None


def build_image_prompt(style_name: str, variant: dict) -> str:
    outfit_en = variant.get("image_prompt") or ""
    if not outfit_en:
        clothes = variant.get("clothes") or []
        outfit_en = ", ".join(str(c) for c in clothes[:6])
    return (
        "Professional product photography of a complete fashion outfit displayed on a white mannequin, "
        "clean white background, soft studio lighting, full body shot, e-commerce catalog style, "
        "sharp focus, high quality, photorealistic. "
        f"Style: {style_name}. "
        f"Outfit: {outfit_en}."
    )


def style_keyboard(styles: list) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"👔 {s.get('name', f'Imij {i+1}')}", callback_data=f"style_{i}")]
            for i, s in enumerate(styles)
        ]
    )


def variant_keyboard(style_idx: int, variants: list) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(
            text=f"👕 {v.get('name', f'Variant {i+1}')}",
            callback_data=f"variant_{style_idx}_{i}",
        )]
        for i, v in enumerate(variants)
    ]
    rows.append([InlineKeyboardButton(text="⬅️ Boshqa imijni ko'rish", callback_data="back_to_styles")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def variant_details_keyboard(style_idx: int, variant_idx: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🖼 Rasmini ko'rish", callback_data=f"img_{style_idx}_{variant_idx}")],
            [InlineKeyboardButton(text="⬅️ Boshqa variantni ko'rish", callback_data=f"style_{style_idx}")],
            [InlineKeyboardButton(text="🎨 Boshqa imijni ko'rish", callback_data="back_to_styles")],
        ]
    )


def render_summary(data: dict) -> str:
    user_summary = (data.get("user_analysis") or {}).get("summary", "")
    styles = data.get("styles", [])

    out = ""
    if user_summary:
        out += f"👤 <b>Sizning tashqi ko'rinishingiz:</b>\n<i>{html.quote(str(user_summary))}</i>\n\n"

    out += f"🎨 <b>Sizga mos {len(styles)} ta imij topildi:</b>\n\n"
    for i, s in enumerate(styles, 1):
        name = html.quote(str(s.get("name", f"Imij {i}")))
        reason = str(s.get("match_reason") or "")
        out += f"<b>{i}. {name}</b>\n"
        if reason:
            out += f"   💡 <i>{html.quote(reason)}</i>\n"
        out += "\n"

    out += "👇 Batafsil ko'rish uchun imijni tanlang:"
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

        result["styles"] = result["styles"][:4]
        cache_set(user_id, result)
        await log_request(user_id)

        await processing.edit_text(render_summary(result), reply_markup=style_keyboard(result["styles"]))
    except Exception:
        logging.exception("photo_handler crashed")
        try:
            await processing.edit_text("❌ Kutilmagan xatolik. Qayta urinib ko'ring.")
        except Exception:
            pass


@dp.callback_query(F.data.startswith("style_"))
async def style_selection(callback: CallbackQuery) -> None:
    data = cache_get(callback.from_user.id)
    if not data:
        await callback.answer("⏳ Sessiya muddati tugadi", show_alert=True)
        await callback.message.edit_text(
            "❌ Ma'lumotlar topilmadi. Iltimos, qaytadan rasm yuboring."
        )
        return

    try:
        idx = int(callback.data.split("_", 1)[1])
        style = data["styles"][idx]
    except (ValueError, IndexError, KeyError):
        await callback.answer("Xato", show_alert=True)
        return

    variants = style.get("variants") or []
    name = html.quote(str(style.get("name", "Imij")))
    reason = str(style.get("match_reason") or "")

    text = f"✨ <b>{name} imiji</b>\n\n"
    if reason:
        text += f"💡 <i>{html.quote(reason)}</i>\n\n"

    if not variants:
        text += "❌ Ushbu imij uchun variantlar topilmadi."
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="⬅️ Orqaga", callback_data="back_to_styles")]
            ]),
        )
        await callback.answer()
        return

    text += f"👇 Quyidagi {len(variants)} ta variantdan birini tanlang:"
    await callback.message.edit_text(text, reply_markup=variant_keyboard(idx, variants))
    await callback.answer(f"{style.get('name', '')} tanlandi")


@dp.callback_query(F.data.startswith("variant_"))
async def variant_selection(callback: CallbackQuery) -> None:
    data = cache_get(callback.from_user.id)
    if not data:
        await callback.answer("⏳ Sessiya muddati tugadi", show_alert=True)
        await callback.message.edit_text("❌ Iltimos, qaytadan rasm yuboring.")
        return

    try:
        _, style_idx, variant_idx = callback.data.split("_", 2)
        style_idx_i = int(style_idx)
        variant_idx_i = int(variant_idx)
        style = data["styles"][style_idx_i]
        variant = style["variants"][variant_idx_i]
    except (ValueError, IndexError, KeyError):
        await callback.answer("Xato", show_alert=True)
        return

    s_name = html.quote(str(style.get("name", "Imij")))
    v_name = html.quote(str(variant.get("name", "Variant")))
    desc = html.quote(str(variant.get("description") or "—"))

    parts = [
        f"✨ <b>{s_name} → {v_name}</b>\n",
        f"📝 <b>Tavsif:</b>\n{desc}\n",
        "👕 <b>Kiyimlar:</b>",
    ]
    for item in variant.get("clothes", []) or []:
        parts.append(f"• {html.quote(str(item))}")

    parts.append("\n🛍️ <b>Qayerdan sotib olish mumkin:</b>")
    for store in variant.get("stores", []) or []:
        st_name = html.quote(str(store.get("name", "—")))
        st_price = html.quote(str(store.get("price", "—")))
        st_link = str(store.get("link") or "").strip()
        parts.append(f"\n<b>{st_name}</b>")
        parts.append(f"💰 Narxi: {st_price}")
        if st_link.startswith(("http://", "https://")):
            parts.append(f'🔗 <a href="{html.quote(st_link)}">Xarid qilish</a>')

    parts.append("\n💡 <i>Narxlar taxminiy. Aniq narxni do'kondan tekshiring.</i>")

    await callback.message.edit_text(
        "\n".join(parts),
        reply_markup=variant_details_keyboard(style_idx_i, variant_idx_i),
        disable_web_page_preview=True,
    )
    await callback.answer(f"{variant.get('name', '')} tanlandi")


@dp.callback_query(F.data.startswith("img_"))
async def generate_image_callback(callback: CallbackQuery) -> None:
    data = cache_get(callback.from_user.id)
    if not data:
        await callback.answer("⏳ Sessiya muddati tugadi", show_alert=True)
        return

    try:
        _, style_idx, variant_idx = callback.data.split("_", 2)
        s_idx = int(style_idx)
        v_idx = int(variant_idx)
        style = data["styles"][s_idx]
        variant = style["variants"][v_idx]
    except (ValueError, IndexError, KeyError, TypeError):
        await callback.answer("Xato", show_alert=True)
        return

    caption = (
        f"✨ <b>{html.quote(str(style.get('name', '')))} → "
        f"{html.quote(str(variant.get('name', '')))}</b>\n\n"
        f"🤖 AI tomonidan yaratilgan kontseptual rasm"
    )

    await callback.answer("⏳ Rasm yaratilmoqda...")
    waiting_msg = await callback.message.answer("🎨 Rasm yaratilmoqda... (10-30 soniya)")

    prompt = build_image_prompt(str(style.get("name", "")), variant)

    async with aiohttp.ClientSession() as session:
        image_bytes = await generate_outfit_image(session, prompt)

    if not image_bytes:
        try:
            await waiting_msg.edit_text("❌ Rasm yaratishda xatolik. Qayta urinib ko'ring.")
        except Exception:
            pass
        return

    try:
        await waiting_msg.delete()
    except Exception:
        pass

    try:
        photo = BufferedInputFile(image_bytes, filename="outfit.png")
        await callback.message.answer_photo(photo=photo, caption=caption)
    except Exception:
        logging.exception("Failed to send generated photo")
        await callback.message.answer("⚠️ Rasm yaratildi, lekin yuborib bo'lmadi.")


@dp.callback_query(F.data == "back_to_styles")
async def back_to_styles(callback: CallbackQuery) -> None:
    data = cache_get(callback.from_user.id)
    if not data:
        await callback.answer("Sessiya tugadi", show_alert=True)
        await callback.message.edit_text("❌ Iltimos, qaytadan rasm yuboring.")
        return

    await callback.message.edit_text(render_summary(data), reply_markup=style_keyboard(data.get("styles", [])))
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
