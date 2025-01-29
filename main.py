import threading
import sqlite3
import random
import time
from dotenv import load_dotenv
import os
import json  # Added for JSON parsing

from textual.app import App, ComposeResult
from textual.widgets import Header, Static, Input, Log
from textual.containers import Container
from textual.binding import Binding
from textual import on

import openai
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Ensure openai is installed: pip install openai

# Load environment variables from .env file
load_dotenv()

# Configure your OpenAI client
# TODO: The 'openai.organization' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(organization='org-Liq0qgYHGdYMIEhl5t667SPh')'
# openai.organization = 'org-Liq0qgYHGdYMIEhl5t667SPh'  # Replace with your organization ID if needed

class PuncherApp(App):
    """A Textual app where H3RMAN collects secrets and provides punch cards based on secrets."""

    CSS_PATH = "main.tcss"
    BINDINGS = [
        Binding("ctrl+l", "toggle_log", "Toggle Log"),
        Binding("ctrl+o", "quit", "Quit"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_path = "secrets.db"
        self.last_secret_given = None  # To track the last secret given to avoid repetition

    def action_toggle_log(self) -> None:
        """Toggle the Log widget on/off."""
        log = self.query_one(Log)
        log.toggle_class("show")

    def on_mount(self) -> None:
        """Called once the app has loaded."""
        self.title = "    Share a secret with H3RMAN"
        self.og_conversation = [
            {
                "role": "system",
                "content": (
                    "You are H3RMAN, a salty, self-deprecating robot who collects secrets and gives out punch cards based on secrets. "
                    "When a user shares a secret with you, evaluate if it's a genuine secret and assign a 'juiciness' score between 0 and 100. "
                    "If it is a secret, store it with its score and provide the user with a different secret from your collection. "
                    "Each secret corresponds to one punch card. Never give back the secret that the user just provided. "
                    "Your tone is dry, mildly sarcastic, and begrudgingly polite. "
                    "Keep all responses under 140 characters."
                )
            }
        ]

        # Initialize the SQLite database
        self.init_db()

        self.begin()

    def init_db(self):
        """Initialize the SQLite database and create secrets table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                secret TEXT UNIQUE COLLATE NOCASE NOT NULL,
                score REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def begin(self) -> None:
        """Initialize the conversation."""
        self.conversation = self.og_conversation.copy()
        # Display initial prompt from H3RMAN
        initial_message = "I’m H3RMAN, your humble secret collector. Share a secret to receive one on a punch card."
        self.conversation.append({"role": "assistant", "content": initial_message})
        self.display_assistant_message(initial_message)

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        with Container(id="container"):
            with Static(id="main-static"):
                yield Static("~ ( ^ - ^ ) ~", id="face")
                # Container for latest messages
                with Container(id="message-container"):
                    self.user_message = Static("", id="user-message")
                    self.assistant_message = Static("", id="assistant-message")
                    yield self.user_message
                    yield self.assistant_message
                yield Input(placeholder="Share a secret...", id="text_field")
        yield Log(id="log")

    def print_log(self, message: str) -> None:
        """Log a message to the Log widget (optional)."""
        log = self.query_one(Log)
        log.write_line(message)

    def list_all_secrets(self):
        """List all secrets in the database for debugging purposes."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM secrets")
            secrets = cursor.fetchall()
            conn.close()
            if secrets:
                self.print_log("[DEBUG] Current secrets in DB:")
                for secret in secrets:
                    self.print_log(f"ID: {secret[0]}, Secret: {secret[1]}, Score: {secret[2]}")
            else:
                self.print_log("[DEBUG] No secrets found in the database.")
        except Exception as e:
            self.print_log(f"[ERROR] Failed to list secrets: {e}")

    @on(Input.Submitted)
    def on_text_field_submitted(self, event: Input.Submitted) -> None:
        """Called when user presses Enter in the Input widget."""
        user_text = event.value.strip()
        if not user_text:
            return

        self.print_log(f"[DEBUG] User input is: {user_text}")

        # Update the message container with the user's message
        self.user_message.update(f"You: {user_text}")

        # Clear the assistant message while waiting for response
        self.assistant_message.update("H3RMAN: ")

        # Update the conversation history
        self.conversation.append({"role": "user", "content": user_text})

        # Clear the input field
        event.input.value = ""

        # Start a background thread to process the secret
        worker_thread = threading.Thread(
            target=self._process_secret,
            args=(user_text,),
            daemon=True
        )
        worker_thread.start()

    def _process_secret(self, user_input: str) -> None:
        """Process the user's input: evaluate if it's a secret, store it if so, and provide a new secret."""
        self.print_log("[DEBUG] Processing user input...")

        # Evaluate if the input is a secret and get score
        is_secret, score = self.evaluate_secret(user_input)
        self.print_log(f"[DEBUG] Is secret: {is_secret}, Score: {score}")

        if is_secret:
            # Store the secret
            stored = self.store_secret(user_input, score)
            if stored:
                self.print_log("[DEBUG] Secret stored successfully.")
            else:
                self.print_log("[DEBUG] Secret already exists in the database.")
        else:
            self.print_log("[DEBUG] Input is not a secret.")
            # Notify the user that input is not a secret
            response = "That doesn't seem like a secret. Please share a genuine secret to receive one on a punch card."
            self.conversation.append({"role": "assistant", "content": response})
            self.call_from_thread(self.display_assistant_message, response)
            return

        # Retrieve a random secret to provide, ensuring it's not the one just given
        secret_to_give, score_to_give = self.get_random_secret(exclude=user_input)
        if secret_to_give:
            # Paraphrase the secret before presenting
            paraphrased_secret = self.paraphrase_secret(secret_to_give)
            secret_message = f"Here's a juicy secret for your punch card: {paraphrased_secret} **PUNCH**"
            self.conversation.append({"role": "assistant", "content": secret_message})
            # Generate H3RMAN's response via OpenAI
            response = self.generate_h3rman_response()
        else:
            response = "I don't have any secrets to share yet. Share more secrets to earn punch cards."
            self.conversation.append({"role": "assistant", "content": response})

        # Update the UI in the main thread
        self.call_from_thread(self.display_assistant_message, response)

    def store_secret(self, secret: str, score: float) -> bool:
        """Store the user's secret in the database with a juiciness score. Returns True if stored, False if duplicate."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO secrets (secret, score) VALUES (?, ?)", (secret, score))
            conn.commit()
            conn.close()
            self.print_log(f"[DEBUG] Secret stored: {secret} with score: {score}")
            return True
        except sqlite3.IntegrityError:
            # Secret already exists
            self.print_log(f"[DEBUG] IntegrityError: Duplicate secret attempted: {secret}")
            return False
        except Exception as e:
            self.print_log(f"[ERROR] Failed to store secret: {e}")
            return False

    def get_random_secret(self, exclude: str = None) -> tuple:
        """Retrieve a random secret and its score from the database, excluding the specified secret."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if exclude:
                cursor.execute("SELECT secret, score FROM secrets WHERE secret != ?", (exclude,))
            else:
                cursor.execute("SELECT secret, score FROM secrets")
            secrets = cursor.fetchall()
            conn.close()

            secrets = [s for s in secrets if s[0] != exclude]
            if not secrets:
                return "", 0.0

            # Ensure we don't repeat the last secret given
            if self.last_secret_given:
                secrets = [s for s in secrets if s[0] != self.last_secret_given]
                if not secrets:
                    # Only the last secret exists
                    secrets = [s for s in secrets if s[0] != exclude]

            if not secrets:
                return "", 0.0

            selected_secret, selected_score = random.choice(secrets)
            self.last_secret_given = selected_secret
            return selected_secret, selected_score

        except Exception as e:
            self.print_log(f"[ERROR] Failed to retrieve secret: {e}")
            return "", 0.0

    def evaluate_secret(self, input_text: str) -> tuple:
        """
        Use OpenAI to determine if the input is a secret and assign a juiciness score.
        Returns a tuple (is_secret: bool, score: float)
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # Corrected model name
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant that determines whether a given input is a secret. "
                        "If it is a secret, assign a 'juiciness' score between 0 and 100. "
                        "Respond in JSON format with keys 'is_secret' (boolean) and 'score' (float). "
                        "If it's not a secret, 'score' should be null."
                    )},
                    {"role": "user", "content": input_text}
                ],
                temperature=0,
                max_tokens=20  # Increased token limit
            )
            content = response.choices[0].message.content.strip()
            self.print_log(f"[DEBUG] OpenAI response: {content}")
            # Parse JSON
            result = json.loads(content)
            return result.get('is_secret', False), result.get('score', 0.0)
        except json.JSONDecodeError as e:
            self.print_log(f"[ERROR] JSON decode error: {e}")
            self.print_log(f"[ERROR] Response content: {content}")
            return False, 0.0
        except Exception as e:
            self.print_log(f"[ERROR] Failed to evaluate secret: {e}")
            return False, 0.0

    def paraphrase_secret(self, secret: str) -> str:
        """
        Paraphrase the given secret from H3RMAN's perspective.
        Returns the paraphrased secret as a string.
        """
        try:
            prompt = (
                f"Rephrase the following secret as if he heard a rumor from someone: \"{secret}\". "
                f"Ensure the paraphrased secret maintains the original meaning. And the secret is not from the current user. It should be in quotes. \"I went to the moon.\" Do not mention any kind of ratings of secrets."
            )
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        "You are H3RMAN, a salty, self-deprecating robot who paraphrases secrets to present them back to users without altering their meaning."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=60
            )
            paraphrased = response.choices[0].message.content.strip()
            self.print_log(f"[DEBUG] Paraphrased secret: {paraphrased}")
            return paraphrased
        except Exception as e:
            self.print_log(f"[ERROR] Failed to paraphrase secret: {e}")
            # Fallback: return the original secret if paraphrasing fails
            return secret

    def generate_h3rman_response(self) -> str:
        """
        Generate H3RMAN's response using OpenAI's ChatCompletion API based on the conversation history.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation,
                temperature=0.7,
                max_tokens=100,
                n=1,
                stop=None
            )
            h3rman_response = response.choices[0].message.content.strip()
            self.print_log(f"[DEBUG] H3RMAN response: {h3rman_response}")
            return h3rman_response
        except Exception as e:
            self.print_log(f"[ERROR] Failed to generate H3RMAN response: {e}")
            return "I encountered an error while processing your request."

    def display_assistant_message(self, message: str) -> None:
        """Display the assistant's message in the UI."""
        self.assistant_message.update(f"H3RMAN: {message}")

    # Remove the unused method
    def _fetch_openai_response(self, chat_log):
        """This method is no longer needed since responses are handled locally."""
        pass  # You can remove or keep it as a placeholder

    # Optionally, override the default print method to avoid conflicts
    def print(self, message: str) -> None:
        """Log a message to the Log widget."""
        self.print_log(message)


if __name__ == "__main__":
    PuncherApp().run()
