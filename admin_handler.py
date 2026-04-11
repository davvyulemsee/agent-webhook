import os
import psycopg2
from psycopg2.extras import RealDictCursor
from twilio.rest import Client

DATABASE_URL = os.getenv("DATABASE_URL")
TWILIO_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")

twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))


def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def send_to_admin(message: str):
    """Send a WhatsApp message to the admin (architect)."""
    if not ADMIN_PHONE:
        return
    try:
        twilio_client.messages.create(
            from_=TWILIO_NUMBER,
            to=f"whatsapp:+{ADMIN_PHONE}",
            body=message
        )
    except Exception as e:
        print(f"Failed to notify admin: {e}")


def handle_admin_command(command_text: str) -> str:
    """Parse and execute an admin slash command."""
    parts = command_text.strip().split(" ", 2)
    cmd = parts[0].lower()

    try:
        conn = get_conn()
        cursor = conn.cursor()

        # /help
        if cmd == "/help":
            return (
                "Amani AI — Admin commands:\n\n"
                "/clients — active clients today\n"
                "/appointments — pending bookings\n"
                "/tickets — open escalations\n"
                "/summary [phone] — AI summary for client\n"
                "/history [phone] — last 5 messages\n"
                "/takeover [phone] — pause AI, handle yourself\n"
                "/release [phone] — resume AI for client\n"
                "/reply [phone] [message] — send message to client\n"
                "/confirm [appointment_id] — confirm an appointment"
            )

        # /clients
        elif cmd == "/clients":
            cursor.execute("""
                SELECT customer_phone, COUNT(*) as msg_count,
                       MAX(timestamp) as last_seen
                FROM conversations
                WHERE DATE(timestamp) = CURRENT_DATE
                  AND customer_phone != %s
                GROUP BY customer_phone
                ORDER BY last_seen DESC
                LIMIT 10
            """, (ADMIN_PHONE,))
            rows = cursor.fetchall()
            if not rows:
                return "No client activity today."
            lines = ["Today's active clients:\n"]
            for i, r in enumerate(rows, 1):
                lines.append(f"{i}. {r['customer_phone']} ({r['msg_count']} messages)")
            return "\n".join(lines)

        # /appointments
        elif cmd == "/appointments":
            cursor.execute("""
                SELECT id, customer_name, appointment_type,
                       preferred_date, preferred_time, status
                FROM appointments
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            rows = cursor.fetchall()
            if not rows:
                return "No pending appointments."
            lines = ["Pending appointments:\n"]
            for i, r in enumerate(rows, 1):
                lines.append(
                    f"{i}. [ID:{r['id']}] {r['customer_name']}\n"
                    f"   {r['appointment_type']}\n"
                    f"   {r['preferred_date']} at {r['preferred_time']}"
                )
            return "\n".join(lines)

        # /tickets
        elif cmd == "/tickets":
            cursor.execute("""
                SELECT id, customer_phone, reason, urgency, created_at
                FROM tickets
                WHERE status = 'open'
                ORDER BY created_at DESC
                LIMIT 10
            """)
            rows = cursor.fetchall()
            if not rows:
                return "No open tickets."
            lines = ["Open tickets:\n"]
            for i, r in enumerate(rows, 1):
                lines.append(
                    f"{i}. {r['customer_phone']} [{r['urgency'].upper()}]\n"
                    f"   {r['reason']}"
                )
            return "\n".join(lines)

        # /summary [phone]
        elif cmd == "/summary":
            if len(parts) < 2:
                return "Usage: /summary [phone]"
            phone = parts[1].replace("+", "")
            cursor.execute(
                "SELECT summary, created_at FROM conversation_summaries WHERE customer_phone = %s",
                (phone,)
            )
            row = cursor.fetchone()
            if not row:
                return f"No summary yet for {phone}. Summaries generate after 30 min of inactivity."
            return f"Summary for {phone}:\n\n{row['summary']}"

        # /history [phone]
        elif cmd == "/history":
            if len(parts) < 2:
                return "Usage: /history [phone]"
            phone = parts[1].replace("+", "")
            cursor.execute("""
                SELECT role, message, timestamp FROM conversations
                WHERE customer_phone = %s
                ORDER BY timestamp DESC
                LIMIT 5
            """, (phone,))
            rows = cursor.fetchall()
            if not rows:
                return f"No conversation history for {phone}."
            lines = [f"Last 5 messages for {phone}:\n"]
            for r in reversed(rows):
                role_label = "Client" if r["role"] == "user" else ("You" if r["role"] == "human" else "AI")
                lines.append(f"{role_label}: {r['message']}")
            return "\n".join(lines)

        # /takeover [phone]
        elif cmd == "/takeover":
            if len(parts) < 2:
                return "Usage: /takeover [phone]"
            phone = parts[1].replace("+", "")
            cursor.execute(
                "UPDATE conversations SET handoff_active = true WHERE customer_phone = %s",
                (phone,)
            )
            conn.commit()
            return (
                f"You are now handling {phone}.\n"
                f"AI is paused for this client.\n\n"
                f"To reply: /reply {phone} [message]\n"
                f"To resume AI: /release {phone}"
            )

        # /release [phone]
        elif cmd == "/release":
            if len(parts) < 2:
                return "Usage: /release [phone]"
            phone = parts[1].replace("+", "")
            cursor.execute(
                "UPDATE conversations SET handoff_active = false WHERE customer_phone = %s",
                (phone,)
            )
            conn.commit()
            return f"AI has resumed for {phone}."

        # /reply [phone] [message]
        elif cmd == "/reply":
            if len(parts) < 3:
                return "Usage: /reply [phone] [message]"
            phone = parts[1].replace("+", "")
            message = parts[2]

            twilio_client.messages.create(
                from_=TWILIO_NUMBER,
                to=f"whatsapp:+{phone}",
                body=message
            )
            cursor.execute(
                "INSERT INTO conversations (customer_phone, message, role) VALUES (%s, %s, %s)",
                (phone, message, "human")
            )
            conn.commit()
            return f"Message sent to {phone}."

        # /confirm [appointment_id]
        elif cmd == "/confirm":
            if len(parts) < 2:
                return "Usage: /confirm [appointment_id]"
            appt_id = parts[1]
            cursor.execute(
                "UPDATE appointments SET status = 'confirmed' WHERE id = %s RETURNING customer_phone, customer_name, preferred_date, preferred_time",
                (appt_id,)
            )
            row = cursor.fetchone()
            conn.commit()
            if not row:
                return f"No appointment found with ID {appt_id}."

            # Notify the client
            twilio_client.messages.create(
                from_=TWILIO_NUMBER,
                to=f"whatsapp:+{row['customer_phone']}",
                body=(
                    f"Hi {row['customer_name']}, your appointment has been confirmed "
                    f"for {row['preferred_date']} at {row['preferred_time']}. "
                    f"We look forward to seeing you!"
                )
            )
            return f"Appointment {appt_id} confirmed. Client {row['customer_name']} has been notified."

        else:
            return f"Unknown command: {cmd}\nSend /help to see all commands."

    except Exception as e:
        return f"Error running {cmd}: {e}"