import json
import os
from typing import Any, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
import google.generativeai as genai
import httpx
from supabase import Client, create_client

# Cargar variables de entorno locales
load_dotenv()

# --- Configuración de Variables ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# --- Clientes de Servicios ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Asistente Personal IA")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


async def send_telegram_message(chat_id: int | str, text: str):
    """Envía un mensaje de texto formateado de vuelta a Telegram."""
    async with httpx.AsyncClient() as client:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        await client.post(f"{TELEGRAM_API_URL}/sendMessage", json=payload)


SYSTEM_PROMPT = """
Eres el cerebro de un Asistente Personal en Telegram. Tu labor es interpretar el mensaje del usuario (en lenguaje natural) y extraer las acciones estructuradas que deben ejecutarse.

Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
{
  "actions": [
    {
      "type": "save_note | search_notes | add_task | list_tasks | complete_task | add_reminder | list_reminders | general_response",
      "data": { ... }
    }
  ],
  "reply_message": "Resumen amigable y claro en formato Markdown de lo que hiciste o respuesta general"
}

Estructura de 'data' según el tipo de acción:
- save_note: {"content": "texto de la nota", "category": "general|trabajo|ideas|etc"}
- search_notes: {"query": "palabra clave"}
- add_task: {"title": "nombre de la tarea", "due_date": "YYYY-MM-DD o texto como 'mañana'"}
- list_tasks: {"status": "all | pending | completed"}
- complete_task: {"task_id": 123} (si menciona un ID) o {"task_title": "nombre aproximado"}
- add_reminder: {"text": "motivo", "remind_at": "YYYY-MM-DD HH:MM o texto de fecha/hora", "recurring": "none | daily | weekly | monthly"}
- list_reminders: {}
- general_response: {"message": "Respuesta directa a saludos o preguntas generales"}

Si el usuario pide varias cosas a la vez (ej. "guarda X y recuérdame Y"), genera múltiples acciones dentro de la lista 'actions'.
NO incluyas bloques de código markdown tipo ```json en tu salida, solo el texto JSON puro.
"""


def process_intent_with_gemini(user_text: str) -> Dict[str, Any]:
    """Usa Gemini para transformar lenguaje natural en acciones estructuradas."""
    prompt = f"{SYSTEM_PROMPT}\n\nMensaje del usuario: \"{user_text}\""
    response = gemini_model.generate_content(prompt)
    raw_text = response.text.strip()
    
    # Limpieza en caso de que Gemini añada formato markdown
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
        
    return json.loads(raw_text.strip())


def execute_action(action: Dict[str, Any]) -> str:
    """Ejecuta operaciones contra la base de datos en Supabase."""
    action_type = action.get("type")
    data = action.get("data", {})

    # 1. NOTAS
    if action_type == "save_note":
        supabase.table("notes").insert({
            "content": data.get("content", ""),
            "category": data.get("category", "general")
        }).execute()
        return f"📝 *Nota guardada:* {data.get('content')}"

    elif action_type == "search_notes":
        query = data.get("query", "")
        res = supabase.table("notes").select("*").ilike("content", f"%{query}%").execute()
        if not res.data:
            return f"🔍 No encontré notas con la palabra '{query}'."
        lista = "\n".join([f"• [{n['category']}] {n['content']}" for n in res.data])
        return f"🔍 *Notas encontradas:*\n{lista}"

    # 2. TAREAS
    elif action_type == "add_task":
        supabase.table("tasks").insert({
            "title": data.get("title", ""),
            "due_date": data.get("due_date")
        }).execute()
        due = f" (Para: {data.get('due_date')})" if data.get("due_date") else ""
        return f"✅ *Tarea agregada:* {data.get('title')}{due}"

    elif action_type == "list_tasks":
        res = supabase.table("tasks").select("*").eq("is_done", False).order("id").execute()
        if not res.data:
            return "🎉 ¡No tienes tareas pendientes!"
        lista = "\n".join([f"• `[ID {t['id']}]` {t['title']} {f'(📅 {t["due_date"]})' if t['due_date'] else ''}" for t in res.data])
        return f"📋 *Tareas pendientes:*\n{lista}"

    elif action_type == "complete_task":
        if "task_id" in data:
            supabase.table("tasks").update({"is_done": True}).eq("id", data["task_id"]).execute()
            return f"✔️ Tarea #{data['task_id']} marcada como completada."
        elif "task_title" in data:
            supabase.table("tasks").update({"is_done": True}).ilike("title", f"%{data['task_title']}%").execute()
            return f"✔️ Tarea '{data['task_title']}' marcada como completada."

    # 3. RECORDATORIOS
    elif action_type == "add_reminder":
        supabase.table("reminders").insert({
            "text": data.get("text", ""),
            "remind_at": data.get("remind_at", ""),
            "recurring": data.get("recurring", "none")
        }).execute()
        rec = f" [Recurrencia: {data.get('recurring')}]" if data.get("recurring") != "none" else ""
        return f"⏰ *Recordatorio programado:* {data.get('text')} para el {data.get('remind_at')}{rec}"

    elif action_type == "list_reminders":
        res = supabase.table("reminders").select("*").eq("is_active", True).execute()
        if not res.data:
            return "⏰ No tienes recordatorios activos."
        lista = "\n".join([f"• {r['text']} (📅 {r['remind_at']})" for r in res.data])
        return f"⏰ *Recordatorios activos:*\n{lista}"

    return ""


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Telegram AI Assistant"}


@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Endpoint receptor de eventos de Telegram."""
    data = await request.json()

    if "message" in data and "text" in data["message"]:
        sender_id = str(data["message"]["chat"]["id"])
        user_text = data["message"]["text"]

        # Filtro de Seguridad: Solo responder al dueño autorizado
        if sender_id != str(TELEGRAM_CHAT_ID):
            await send_telegram_message(sender_id, "⛔ Acceso no autorizado.")
            return Response(status_code=200)

        # Mensaje de Bienvenida / Ayuda
        if user_text.strip() in ["/start", "/help"]:
            welcome = (
                "👋 *¡Hola! Soy tu Asistente Personal con IA.*\n\n"
                "Puedes pedirme cosas en lenguaje natural:\n"
                "• *Notas:* \"Anota que el sensor opera a 24V\"\n"
                "• *Búsqueda:* \"Busca notas de sensores\"\n"
                "• *Tareas:* \"Agrega tarea revisar reporte para mañana\"\n"
                "• *Ver tareas:* \"¿Qué tareas tengo pendientes?\"\n"
                "• *Completar:* \"Marcar como lista la tarea 1\"\n"
                "• *Recordatorios:* \"Recuérdame llamar al proveedor el viernes a las 3pm\"\n"
                "• *Combinado:* \"Guarda nota X y crea tarea Y\""
            )
            await send_telegram_message(sender_id, welcome)
            return Response(status_code=200)

        # Procesar con Gemini y ejecutar acciones
        try:
            parsed = process_intent_with_gemini(user_text)
            responses = []

            for action in parsed.get("actions", []):
                result = execute_action(action)
                if result:
                    responses.append(result)

            final_message = "\n".join(responses) if responses else parsed.get("reply_message", "✅ Procesado.")
            await send_telegram_message(sender_id, final_message)

        except Exception as e:
            error_msg = f"⚠️ Ocurrió un error al procesar la solicitud: `{str(e)}`"
            await send_telegram_message(sender_id, error_msg)

    return Response(status_code=200)