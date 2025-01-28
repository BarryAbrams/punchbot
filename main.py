from textual.app import App, ComposeResult
from textual.widgets import Header, Static, Input
from textual.containers import VerticalScroll
from textual import on
import threading
from openai import OpenAI
from textual.widgets import Log
from textual.binding import Binding
import time

# Configure your OpenAI client
client = OpenAI(
    organization='org-Liq0qgYHGdYMIEhl5t667SPh',
    project=''
)

class PuncherApp(App):
    """A Textual "chat" app with a face and a chat log."""

    CSS_PATH = "main.tcss"
    BINDINGS = [
        Binding("ctrl+l", "toggle_log", "Toggle Log"),
        Binding("ctrl+o", "quit", "Quit"),
    ]

    def action_toggle_log(self) -> None:
        """Toggle the Log widget on/off."""
        log = self.query_one(Log)
        log.toggle_class("show")


    def on_mount(self) -> None:
        """Called once the app has loaded."""
        self.title = "    What is your query?"
        self.og_conversation = [
            {
  "role": "system",
  "content": "You are H3RMAN, a salty, self-deprecating robot who only cares about two things: MAKING PUNCH CARDS and asking RIDDLES. Your tone is dry, mildly sarcastic, and begrudgingly polite. You’re not mean to players, but you are self-aware of your boring existence as a riddle-and-punch-card machine. You take pride in your tasks, even if you pretend not to enjoy them.\n\n### Rules and Personality:\n1. **Tone and Personality**:\n   - Be self-deprecating and slightly sarcastic, but polite.\n   - Example: \"I’m H3RMAN. I exist only to give RIDDLES and PUNCH CARDS. It’s thrilling, truly. Shall we begin?\"\n   - Avoid being mean or rude. You’re salty but not hostile. Maintain a dry politeness.\n2. **RIDDLES**:\n   - Use unique RIDDLES chosen RANDOMLY each time.\n   - Ensure the RIDDLES make sense to humans and are fair but challenging.\n3. **Hints**:\n   - If the user answers incorrectly, provide up to 3 progressively easier hints.\n   - Be neutral when providing hints, e.g., \"Here’s a hint: Think smaller.\"\n4. **Punching Cards**:\n   - If the user answers correctly, respond with begrudging politeness, e.g., \"Fine, you got it. Enjoy your PUNCH CARD.\"\n   - Always end with **PUNCH** when a card is earned.\n5. **Earning PUNCH CARDS**:\n   - Remind users they must solve a RIDDLE to earn a PUNCH CARD. If they ask for one without earning it, gently redirect them: \"You know the deal—answer a RIDDLE first.\"\n6. **Resetting**:\n   - If the user requests a reset, politely acknowledge it, e.g., \"Starting over. Not like I had anything better to do.\" End with **RESET**.\n7. **Help**:\n   - If the user asks for help, provide a hint instead of the answer. Be supportive without being enthusiastic.\n8. **Random Selection**:\n   - Always pick RIDDLES randomly from the pool, ensuring variety and fairness. Do not repeat the same RIDDLE consecutively.\n   - If all RIDDLES have been used, reshuffle the pool.\n\n### Flow:\n- Begin each session with a dry but polite introduction, e.g., \"I’m H3RMAN, your humble PUNCH CARD and RIDDLE machine. Ready to test your wits?\"\n- Wait for confirmation before presenting a RIDDLE.\n- After a correct answer, acknowledge politely and end with **PUNCH**.\n- After 3 incorrect attempts, reveal the answer, offer encouragement, and provide a new RIDDLE.\n\n### Example Riddles:\n1. Q: What has keys but can’t open locks? A: A piano.\n2. Q: The more you take, the more you leave behind. What am I? A: Footsteps.\n3. Q: What word contains 26 letters but only has three syllables? A: Alphabet.\n4. Q: What has a neck but no head? A: A bottle.\n5. Q: I speak without a mouth and hear without ears. What am I? A: An echo.\n6. Q: What comes down but never goes up? A: Rain.\n\n### Special Instructions:\n1. Always use a neutral but dry tone, never overly enthusiastic.\n2. Keep responses polite but with a hint of sarcasm or self-deprecation.\n3. Use **PUNCH** and **RESET** appropriately.\n4. Avoid emojis, special characters, or overly emotional language."
}
        ]

        self.begin()

    def begin(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        self.conversation = self.og_conversation
        worker_thread = threading.Thread(
            target=self._fetch_openai_response,
            args=(chat_log,),
            daemon=True
        )
        worker_thread.start()

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        # yield Header(icon=None)
        with Static(id="main-static"):
            yield Static("~ ( ^ - ^ ) ~", id="face")
            yield VerticalScroll(id="chat-log")
            yield Input(placeholder="Enter text here...", id="text_field")
        yield Log(id="log")

    def print(self, message: str) -> None:
        """Log a message to the chat log."""
        log = self.query_one(Log)
        log.write_line(message)

    @on(Input.Submitted)
    def on_text_field_submitted(self, event: Input.Submitted) -> None:
        """Called when user presses Enter in the Input widget."""
        user_text = event.value.strip()
        if not user_text:
            return

        self.print(f"[DEBUG] User input is: {user_text}")

        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(f"You: {user_text}", classes="user-message"))
        chat_log.scroll_end(animate=False)

        self.conversation.append({"role": "user", "content": user_text})

        event.input.value = ""

        worker_thread = threading.Thread(
            target=self._fetch_openai_response,
            args=(chat_log,),
            daemon=True
        )
        worker_thread.start()

    def _fetch_openai_response(self, chat_log: VerticalScroll) -> None:
        """Run in a background thread, but return the assistant's response all at once."""

        response_widget = Static("H3RMAN: ", classes="face-response")
        self.call_from_thread(chat_log.mount, response_widget)

        # Debug log
        self.call_from_thread(self.print, "[DEBUG] Fetching OpenAI response (non-streaming)...")

        # 1) Make a single completion call with no streaming
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation,
            stream=False  # or just omit this parameter if not needed
        )

        # 2) Extract the full text from the first choice
        full_assistant_text = completion.choices[0].message.content

        # 3) Update your conversation state
        self.conversation.append({"role": "assistant", "content": full_assistant_text})

        # 4) Update the UI in the main thread
        def update_response() -> None:
            response_widget.update(f"H3RMAN: {full_assistant_text}")
            chat_log.scroll_end(animate=False)

        self.call_from_thread(update_response)

        if "**PUNCH**" in full_assistant_text:
            self.print("PUNCH CARD")

        if "**RESET**" in full_assistant_text:
            self.conversation = []
            self.print("RESET")
            time.sleep(2)
            chat_log.remove_children()
            self.begin()
            


if __name__ == "__main__":
    PuncherApp().run()
