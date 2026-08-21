import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
import google.generativeai as genai
import httpx
from supabase import Client, create_client
from datetime import datetime

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-latest")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Asistente Personal IA")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

async def send_telegram_message(chat_id: int | str, text: str):
    async with httpx.AsyncClient() as client:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        await client.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload)

SYSTEM_PROMPT = """
Eres el cerebro de un Asistente Personal en Telegram. Tu labor es interpretar el mensaje del usuario y extraer acciones.

REGLA CRÍTICA DE FECHAS: Siempre que extraigas una fecha/hora (due_date o remind_at), DEBES devolverla estrictamente en formato "YYYY-MM-DD HH:MM". 
Usa el año actual. Si el usuario no dice la hora exacta, asume "09:00" para tareas y "12:00" para recordatorios.
Ejemplo: "mañana en la tarde" -> "2026-08-22 15:00".

Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
{
  "actions": [
    {
      "type": "save_note | search_notes | add_task | list_tasks | complete_task | add_reminder | list_reminders | general_response",
      "data": { ... }
    }
  ],
  "reply_message": "Mensaje amigable confirmando lo que hiciste en Markdown"
}
"""

def process_intent_with_gemini(user_text: str) -> Dict[str, Any]:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt = f"{SYSTEM_PROMPT}\n\nFECHA Y HORA ACTUAL (MÉXICO): {now_str}\n\nMensaje del usuario: \"{user_text}\""
    response = gemini_model.generate_content(prompt)
    raw_text = response.text.strip()
    
    if raw_text.startswith("```json"): raw_text = raw_text[7:]
    if raw_text.startswith("```"): raw_text = raw_text[3:]
    if raw_text.endswith("```"): raw_text = raw_text[:-3]
        
    return json.loads(raw_text.strip())

def execute_action(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    data = action.get("data", {})

    if action_type == "save_note":
        supabase.table("notes").insert({"content": data.get("content", ""), "category": data.get("category", "general")}).execute()
    elif action_type == "search_notes":
        res = supabase.table("notes").select("*").ilike("content", f"%{data.get('query', '')}%").execute()
        return f"🔍 *Notas:*\n" + "\n".join([f"• [{n['category']}] {n['content']}" for n in res.data]) if res.data else "🔍 No encontré nada."
    elif action_type == "add_task":
        supabase.table("tasks").insert({"title": data.get("title", ""), "due_date": data.get("due_date")}).execute()
    elif action_type == "list_tasks":
        res = supabase.table("tasks").select("*").eq("is_done", False).order("id").execute()
        return f"📋 *Tareas pendientes:*\n" + "\n".join([f"• `[ID {t['id']}]` {t['title']} (📅 {t['due_date']})" for t in res.data]) if res.data else "🎉 No tienes tareas."
    elif action_type == "complete_task":
        if "task_id" in data:
            supabase.table("tasks").update({"is_done": True}).eq("id", data["task_id"]).execute()
    elif action_type == "add_reminder":
        supabase.table("reminders").insert({"text": data.get("text", ""), "remind_at": data.get("remind_at", "")}).execute()
    elif action_type == "list_reminders":
        res = supabase.table("reminders").select("*").eq("is_active", True).execute()
        return f"⏰ *Recordatorios:*\n" + "\n".join([f"• {r['text']} (📅 {r['remind_at']})" for r in res.data]) if res.data else "⏰ No hay recordatorios."
    
    return "" # Las confirmaciones de creación ahora las da 'reply_message' de Gemini directamente.

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Telegram AI"}

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    if "message" in data and "text" in data["message"]:
        sender_id = str(data["message"]["chat"]["id"])
        user_text = data["message"]["text"]

        if sender_id != str(TELEGRAM_CHAT_ID):
            return Response(status_code=200)

        try:
            parsed = process_intent_with_gemini(user_text)
            responses = [execute_action(a) for a in parsed.get("actions", [])]
            # Extraemos los resultados de búsquedas/listas, si no hay, usamos la respuesta amable de Gemini
            list_results = [r for r in responses if r]
            final_message = "\n\n".join(list_results) if list_results else parsed.get("reply_message", "✅ Listo.")
            
            await send_telegram_message(sender_id, final_message)
        except Exception as e:
            await send_telegram_message(sender_id, f"⚠️ Error: `{str(e)}`")

    return Response(status_code=200)