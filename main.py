#!/usr/bin/env python3
"""
Merged Script: H3RMAN Secret Collector & Card Puncher with Combined Status Indicator, 
Progress Bar, and Real Stepper Motor Control via DRV8825

This app:
  - Uses a Textual UI to let a user share a secret.
  - (In debug mode) Bypasses secret evaluation/storage and directly uses the input.
  - Converts each character in the message into two 4‐bit nibbles.
  - Uses 5 solenoids (from MCP23S17 Port A, pins 0–4) to “punch” a card:
      • Solenoid[1] acts as the high/low nibble indicator:
          – UP (False) means high nibble.
          – DOWN (True) means low nibble.
      • The remaining solenoids represent the 4 bits.
  - Moves paper by either simulating it or by moving a real stepper motor via DRV8825.
      • The move methods now take an argument in revolutions.
      • Internally, the code multiplies the revolutions by a calibrated step count.
  - Combines the status for paper, solenoids, stepper, and progress percentage on a single line.
    For example:
         □       ○ ○ ○ ○ ○ ◑           85%
    (Paper is left‐aligned, solenoids/stepper centered, and percentage right‐aligned.)
"""

from typing import Iterable
import threading
import sqlite3
import random
import time
import json
import os
import heapq
from dotenv import load_dotenv
from digitalio import Direction, Pull

# Textual UI imports
from textual.app import App, ComposeResult, SystemCommand
from textual.widgets import Static, Input, Log, Button, Checkbox, Header, Footer
from textual.containers import Container, Vertical
from textual.binding import Binding
from textual.screen import Screen
from textual import on

# OpenAI
import openai
from openai import OpenAI

# MCP23S17 & hardware-related imports (CircuitPython libraries)
import board
import busio
import digitalio
from adafruit_mcp230xx.mcp23s17 import MCP23S17

# Import the DRV8825 stepper motor driver library
from DRV8825 import DRV8825

# -----------------------------------------------------------------------------
# Timed Event Scheduler
# -----------------------------------------------------------------------------
class TimedEventScheduler:
    def __init__(self):
        self.events = []  # list of tuples: (run_time, callback, args, kwargs)
        self.cv = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def schedule(self, delay: float, callback, *args, **kwargs):
        """Schedule callback(*args, **kwargs) to run after delay seconds."""
        run_time = time.time() + delay
        with self.cv:
            heapq.heappush(self.events, (run_time, callback, args, kwargs))
            self.cv.notify()

    def _run(self):
        while self.running:
            with self.cv:
                while not self.events and self.running:
                    self.cv.wait()
                if not self.running:
                    break
                run_time, callback, args, kwargs = self.events[0]
                now = time.time()
                delay = run_time - now
                if delay > 0:
                    self.cv.wait(timeout=delay)
                    continue
                heapq.heappop(self.events)
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print("Error in scheduled event:", e)

    def stop(self):
        with self.cv:
            self.running = False
            self.cv.notify_all()
        self.thread.join()

# -----------------------------------------------------------------------------
# Helper functions for low-level register access for the MCP23S17.
# -----------------------------------------------------------------------------
WRITE_OPCODE = 0x40  # For address 0 (R/W bit = 0)
READ_OPCODE  = 0x41  # For address 0 (R/W bit = 1)

def mcp_write_register(spi, cs, reg, value):
    """Write a single byte to the specified register on the MCP23S17."""
    cs.value = False  # assert CS
    spi.write(bytes([WRITE_OPCODE, reg, value]))
    cs.value = True   # deassert CS
    time.sleep(0.001)

def mcp_read_register(spi, cs, reg):
    """Read a single byte from the specified register on the MCP23S17."""
    cs.value = False
    result = bytearray(3)
    spi.write_readinto(bytes([READ_OPCODE, reg, 0x00]), result)
    cs.value = True
    time.sleep(0.001)
    return result[2]

# -----------------------------------------------------------------------------
# New Settings Screen
# -----------------------------------------------------------------------------
class SettingsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header("Settings")
        with Vertical():
            yield Button("Stepper Feed", id="stepper-feed")
            yield Button("Solenoid Test", id="solenoid-test")
            # Checkbox to toggle paper sensor
            yield Checkbox("Enable Paper Sensor", id="paper-sensor-toggle", value=self.app.paper_sensor_enabled)
            yield Button("Back", id="back-button")
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "stepper-feed":
            self.app.feed_stepper()
        elif button_id == "solenoid-test":
            self.app.solenoid_test()
        elif button_id == "back-button":
            await self.app.pop_screen()
            # Delay a little before setting focus back to allow Textual to complete its transition.
            await self.sleep(0.1)
            self.app.set_focus(self.app.query_one("#text_field", Input))

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "paper-sensor-toggle":
            self.app.paper_sensor_enabled = event.value
            self.app.print_log(f"[DEBUG] Paper sensor enabled: {self.app.paper_sensor_enabled}")

# -----------------------------------------------------------------------------
# Main Application Class: PuncherApp
# -----------------------------------------------------------------------------
class PuncherApp(App):
    CSS_PATH = "main.tcss"
    BINDINGS = [
        Binding("ctrl+l", "toggle_log", "Toggle Log"),
        Binding("ctrl+o", "quit", "Quit"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_path = "secrets.db"
        self.last_secret_given = None  # To avoid repeating the same punch
        # Hardware attributes for MCP23S17 (solenoids and paper sensor)
        self.mcp = None
        self.spi = None
        self.cs = None
        self.reset = None
        self.output_pins = []      # We'll use 5 solenoids (Port A, pins 0–4)
        self.paper_sensor = None   # Input pin (Port B, pin 14)
        self.paper_available = False
        # For the combined indicator:
        self.stepper_state = "◑"  # Initially idle (used only for UI simulation)
        # For progress tracking (percentage as integer):
        self.progress_percent = 0
        # Debug mode: if True, bypass secret evaluation/storage and punch the input directly.
        self.debug_mode = True
        # Stepper motor instance (to be initialized in init_hardware)
        self.stepper = None
        # Calibrated steps per revolution for the stepper motor
        self.steps_per_rev = 200
        # New attribute: allow toggling paper sensor enabled/disabled.
        self.paper_sensor_enabled = False

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield SystemCommand(
            "Quit the application",
            "Quit the application as soon as possible",
            self.action_quit,
        )
        yield SystemCommand("Stepper Feed", "Feed the stepper", self.feed_stepper)  
        yield SystemCommand("Solenoid Test", "Solenoid Test", self.solenoid_test)
        yield SystemCommand("Settings", "Open settings page", self.open_settings_page)

    def open_settings_page(self):
        self.push_screen(SettingsScreen())

    def feed_stepper(self):
        self.move_paper_async(5, lambda: self.feed_stepper_callback())        

    def feed_stepper_callback(self):
        return False

    def solenoid_test(self):
        def solenoid_testing():
            for pin in self.output_pins:
                pin.value = True
                time.sleep(0.25)
            time.sleep(1)
            for pin in self.output_pins:
                pin.value = False
                time.sleep(0.25)
        threading.Thread(target=solenoid_testing, daemon=True).start()

    def thread_safe_call(self, fn, *args, **kwargs):
        """
        If we're in the main thread, call the function directly;
        otherwise, schedule it via call_from_thread.
        """
        if threading.current_thread() == threading.main_thread():
            fn(*args, **kwargs)
        else:
            self.call_from_thread(fn, *args, **kwargs)

    def action_toggle_log(self) -> None:
        log = self.query_one(Log)
        log.toggle_class("show")

    def on_mount(self) -> None:
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
        self.init_db()
        self.begin()
        self.init_hardware()
        self.start_sensor_polling()
        # Initialize the scheduler for timed events
        self.scheduler = TimedEventScheduler()

    def init_db(self):
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
        self.conversation = self.og_conversation.copy()
        initial_message = "I’m H3RMAN, your humble secret collector. Share a secret to receive one on a punch card."
        self.conversation.append({"role": "assistant", "content": initial_message})
        self.display_assistant_message(initial_message)

    def compose(self) -> ComposeResult:
        with Container(id="container"):
            with Static(id="main-static"):
                with Container(id="message-container"):
                    self.user_message = Static("", id="user-message")
                    self.assistant_message = Static("", id="assistant-message")
                    yield self.user_message
                    yield self.assistant_message
                yield Input(placeholder="Share a secret...", id="text_field")
                with Container(id="indicators"):
                    self.combined_indicator = Static("□           ○ ○ ○ ○ ○ ◑           0%", id="combined_indicator")
                    yield self.combined_indicator
        yield Log(id="log")

    def print_log(self, message: str) -> None:
        """Attempt to write to the Log widget. If not available, print to stdout."""
        try:
            log = self.query_one(Log)
            log.write_line(message)
        except Exception:
            print(message)

    def update_combined_indicator(self):
        """Update the combined status indicator line with paper, solenoids, stepper, and progress."""
        paper_str = "■" if self.paper_available else "□"
        solenoid_str = " ".join("●" if pin.value else "○" for pin in self.output_pins)
        middle_str = f"{solenoid_str} {self.stepper_state}"
        progress_str = f"{self.progress_percent}%"
        combined = f"{paper_str:<8}{middle_str:^20}{progress_str:>8}"
        self.combined_indicator.update(combined)

    @on(Input.Submitted)
    def on_text_field_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return
        self.print_log(f"[DEBUG] User input: {user_text}")
        self.user_message.update(f"You: {user_text}")
        self.assistant_message.update("H3RMAN: ")
        self.conversation.append({"role": "user", "content": user_text})
        event.input.value = ""
        worker_thread = threading.Thread(
            target=self._process_secret,
            args=(user_text,),
            daemon=True
        )
        worker_thread.start()

    def _process_secret(self, user_input: str) -> None:
        if self.debug_mode:
            self.print_log("[DEBUG] Debug mode active. Bypassing secret evaluation.")
            self.thread_safe_call(self.display_assistant_message, "DEBUG: Punching card...")
            self.punch_card(user_input)
            return

        self.print_log("[DEBUG] Processing user input normally...")
        is_secret, score = self.evaluate_secret(user_input)
        self.print_log(f"[DEBUG] Is secret: {is_secret}, Score: {score}")

        if is_secret:
            stored = self.store_secret(user_input, score)
            if stored:
                self.print_log("[DEBUG] Secret stored successfully.")
            else:
                self.print_log("[DEBUG] Secret already exists in the database.")
        else:
            self.print_log("[DEBUG] Input is not a secret.")
            response = "That doesn't seem like a secret. Please share a genuine secret to receive one on a punch card."
            self.conversation.append({"role": "assistant", "content": response})
            self.thread_safe_call(self.display_assistant_message, response)
            return

        secret_to_give, score_to_give = self.get_random_secret(exclude=user_input)
        if secret_to_give:
            paraphrased_secret = self.paraphrase_secret(secret_to_give)
            secret_message = f"Here's a juicy secret for your punch card: {paraphrased_secret} **PUNCH**"
            self.conversation.append({"role": "assistant", "content": secret_message})
            response = self.generate_h3rman_response()
        else:
            response = "I don't have any secrets to share yet. Share more secrets to earn punch cards."
            self.conversation.append({"role": "assistant", "content": response})
        self.thread_safe_call(self.display_assistant_message, response)
        if secret_to_give:
            self.punch_card(secret_to_give)

    def store_secret(self, secret: str, score: float) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO secrets (secret, score) VALUES (?, ?)", (secret, score))
            conn.commit()
            conn.close()
            self.print_log(f"[DEBUG] Stored secret: {secret} (Score: {score})")
            return True
        except sqlite3.IntegrityError:
            self.print_log(f"[DEBUG] Duplicate secret not stored: {secret}")
            return False
        except Exception as e:
            self.print_log(f"[ERROR] Failed to store secret: {e}")
            return False

    def get_random_secret(self, exclude: str = None) -> tuple:
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
            if self.last_secret_given:
                secrets = [s for s in secrets if s[0] != self.last_secret_given]
                if not secrets:
                    secrets = [s for s in secrets if s[0] != exclude]
            if not secrets:
                return "", 0.0
            selected_secret, selected_score = random.choice(secrets)
            self.last_secret_given = selected_secret
            return selected_secret, selected_score
        except Exception as e:
            self.print_log(f"[ERROR] Failed to retrieve a secret: {e}")
            return "", 0.0

    def evaluate_secret(self, input_text: str) -> tuple:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
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
                max_tokens=20
            )
            content = response.choices[0].message.content.strip()
            self.print_log(f"[DEBUG] OpenAI evaluation response: {content}")
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
        try:
            prompt = (
                f"Rephrase the following secret as if you heard it as a rumor: \"{secret}\". "
                "Keep the original meaning intact. Format your response in quotes (e.g., \"I went to the moon.\")."
            )
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        "You are H3RMAN, a sarcastic robot paraphrasing secrets without altering their meaning."
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
            return secret

    def generate_h3rman_response(self) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=self.conversation,
                temperature=0.7,
                max_tokens=100
            )
            h3rman_response = response.choices[0].message.content.strip()
            self.print_log(f"[DEBUG] H3RMAN response: {h3rman_response}")
            return h3rman_response
        except Exception as e:
            self.print_log(f"[ERROR] Failed to generate H3RMAN response: {e}")
            return "I encountered an error while processing your request."

    def display_assistant_message(self, message: str) -> None:
        self.assistant_message.update(f"H3RMAN: {message}")

    # ------------------- Hardware Initialization & Combined Indicator -------------------
    def init_hardware(self):
        try:
            # Initialize SPI and MCP23S17 for solenoids and sensors
            self.spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI, MISO=board.MISO)
            while not self.spi.try_lock():
                pass
            self.spi.configure(baudrate=1000000)
            self.spi.unlock()
            self.cs = digitalio.DigitalInOut(board.D8)
            self.cs.direction = digitalio.Direction.OUTPUT
            self.cs.value = True
            self.reset = digitalio.DigitalInOut(board.D24)
            self.reset.direction = digitalio.Direction.OUTPUT
            self.print_log("Resetting MCP23S17 using D24...")
            self.reset.value = False
            time.sleep(0.1)
            self.reset.value = True
            time.sleep(0.1)
            self.mcp = MCP23S17(self.spi, self.cs, address=0)
            self.print_log("MCP23S17 initialized successfully.")
            self.output_pins = []
            for i in range(5):
                pin = self.mcp.get_pin(i)
                pin.switch_to_output(value=False)
                self.print_log(f"MCP pin {i} set as OUTPUT for solenoid.")
                self.output_pins.append(pin)

            self.paper_sensor = self.mcp.get_pin(14)
            self.paper_sensor.switch_to_input()
            self.paper_sensor.direction = Direction.INPUT
            self.paper_sensor.pull = Pull.UP

            self.print_log("MCP pin 14 set as INPUT (paper sensor)")
            mcp_write_register(self.spi, self.cs, 0x05, 0xC0)
            mcp_write_register(self.spi, self.cs, 0x09, 0x00)
            mcp_write_register(self.spi, self.cs, 0x0A, 0x44)
            _ = mcp_read_register(self.spi, self.cs, 0x0C)
            _ = mcp_read_register(self.spi, self.cs, 0x0D)
            self.print_log("MCP23S17 interrupt registers configured.")
            self.paper_available = self.paper_sensor.value
            self.thread_safe_call(self.update_combined_indicator)
            
            # Initialize the stepper motor via DRV8825
            try:
                self.stepper = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16, 17, 20))
                self.stepper.SetMicroStep('hardward', 'fullstep')
                self.print_log("Stepper motor initialized successfully.")
            except Exception as e:
                self.print_log(f"[ERROR] Stepper motor initialization failed: {e}")
                self.stepper = None

        except Exception as e:
            self.print_log(f"[ERROR] Hardware initialization failed: {e}")
            self.mcp = None

    def poll_paper_sensor(self):
        last_state = self.paper_available
        while True:
            if self.paper_sensor_enabled:
                try:
                    current_state = self.paper_sensor.value if self.paper_sensor is not None else False
                except Exception as e:
                    self.print_log(f"[ERROR] Failed to read paper sensor: {e}")
                    current_state = False
            else:
                current_state = True  # Simulate that paper is always available when sensor is disabled
            
            if current_state != last_state:
                self.print_log(f"Paper sensor changed: {last_state} -> {current_state}")
                self.paper_available = current_state
                self.thread_safe_call(self.update_combined_indicator)
                last_state = current_state
            time.sleep(0.1)

    def start_sensor_polling(self):
        if self.paper_sensor is not None:
            sensor_thread = threading.Thread(target=self.poll_paper_sensor, daemon=True)
            sensor_thread.start()
            self.print_log("Started paper sensor polling thread.")

    # ------------------- Asynchronous Punching Methods -------------------
    def punch_card(self, message: str) -> None:
        self.print_log(f"Attempting to punch card for message: {message}")
        def start_punch():
            self.thread_safe_call(self.display_assistant_message, "Paper detected. Punching...")
            self.move_paper_async(2, lambda: self.process_message_async(message, 0))
        if not self.paper_available:
            self.thread_safe_call(self.display_assistant_message, "Please insert paper for punching.")
            self.wait_for_paper(start_punch)
        else:
            start_punch()

    def wait_for_paper(self, callback):
        if self.paper_available:
            callback()
        else:
            self.scheduler.schedule(0.1, self.wait_for_paper, callback)

    # --- Stepper Motor Based Paper Movement ---
    def move_paper_async(self, revolutions: float, callback):
        steps = int(revolutions * self.steps_per_rev)
        if self.stepper is not None:
            def run_motor():
                self.print_log(f"Moving paper backward {revolutions:.4f} rev ({steps} steps) using stepper motor.")
                self.stepper.TurnStep(Dir='backward', steps=steps, stepdelay=0.001)
                self.stepper.Stop()
                self.thread_safe_call(callback)
            threading.Thread(target=run_motor, daemon=True).start()
        else:
            self.print_log(f"Stepper not available; simulating moving paper {steps} steps.")
            states = ["◐", "◑", "◒", "◓"]
            self._paper_step = 0
            def step():
                if self._paper_step < steps:
                    self.stepper_state = states[self._paper_step % len(states)]
                    self.thread_safe_call(self.update_combined_indicator)
                    self._paper_step += 1
                    self.scheduler.schedule(0.01, step)
                else:
                    self.stepper_state = "◑"
                    self.thread_safe_call(self.update_combined_indicator)
                    callback()
            self.scheduler.schedule(0.0, step)

    def move_paper_reverse_async(self, revolutions: float, callback):
        steps = int(revolutions * self.steps_per_rev)
        if self.stepper is not None:
            def run_motor():
                self.print_log(f"Moving paper backward {revolutions:.4f} rev ({steps} steps) using stepper motor.")
                self.stepper.TurnStep(Dir='forward', steps=steps, stepdelay=0.00001)
                self.stepper.Stop()
                self.thread_safe_call(callback)
            threading.Thread(target=run_motor, daemon=True).start()
        else:
            self.print_log(f"Stepper not available; simulating reverse movement of {steps} steps.")
            states = ["◓", "◒", "◑", "◐"]
            self._paper_step = 0
            def step():
                if self._paper_step < steps:
                    self.stepper_state = states[self._paper_step % len(states)]
                    self.thread_safe_call(self.update_combined_indicator)
                    self._paper_step += 1
                    self.scheduler.schedule(0.01, step)
                else:
                    self.stepper_state = "◑"
                    self.thread_safe_call(self.update_combined_indicator)
                    callback()
            self.scheduler.schedule(0.0, step)

    def process_message_async(self, message: str, index: int):
        if index < len(message):
            self.punch_character_async(message[index], lambda: self.after_char(message, index))
        else:
            self.print_log("Punching complete.")
            self.move_paper_async(3, lambda: self.thread_safe_call(self.update_combined_indicator))

    def after_char(self, message: str, index: int):
        self.progress_percent = int(round((index + 1) / len(message) * 100))
        self.thread_safe_call(self.update_combined_indicator)
        self.scheduler.schedule(0.0, self.process_message_async, message, index + 1)

    def punch_character_async(self, char: str, callback):
        code = ord(char)
        high_nibble = (code >> 4) & 0xF
        low_nibble = code & 0xF
        self.print_log(f"Punching char '{char}' (0x{code:02X}): high nibble 0x{high_nibble:X}, low nibble 0x{low_nibble:X}")
        self.punch_nibble_async(high_nibble, True, lambda: 
            self.move_paper_async(1, lambda: 
                self.punch_nibble_async(low_nibble, False, lambda: 
                    self.scheduler.schedule(0.5, callback)
                )
            )
        )

    def punch_nibble_async(self, nibble: int, is_high: bool, callback):
        indicator_value = False if is_high else True
        self.output_pins[1].value = indicator_value
        bit_mappings = [(0, 3), (2, 2), (3, 1), (4, 0)]
        for sol_index, bit_pos in bit_mappings:
            bit_val = (nibble >> bit_pos) & 1
            self.output_pins[sol_index].value = (bit_val == 1)
        self.thread_safe_call(self.update_combined_indicator)
        self.print_log(f"Punched nibble 0x{nibble:X} as {'HIGH' if is_high else 'LOW'} nibble.")
        self.scheduler.schedule(0.2, self._reset_solenoids)
        def after_nibble():
            self.thread_safe_call(self.update_combined_indicator)
            callback()
        self.scheduler.schedule(0.3, after_nibble)

    def _reset_solenoids(self):
        for pin in self.output_pins:
            pin.value = False
        self.thread_safe_call(self.update_combined_indicator)

    def punched_chars(self) -> int:
        """Helper to return the number of characters punched so far."""
        return int(round(self.total_chars * self.progress_percent / 100))

    def on_shutdown(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.stop()

    def _fetch_openai_response(self, chat_log):
        pass

    def print(self, message: str) -> None:
        self.print_log(message)

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    PuncherApp().run()
