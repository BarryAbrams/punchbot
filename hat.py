#!/usr/bin/env python3
"""
This script sets up the MCP23S17 over SPI, configures:
  - Port A (pins 0–6) as outputs.
  - Port B: pins 15 and 14 as inputs (with pull-ups) and pins 13,12,11,10,9 as outputs.
It then performs an output cycle once, and thereafter polls the two input pins.
When an input changes, the script prints a message showing the change.
"""

import time
import board
import busio
import digitalio
import struct

# Import the CircuitPython MCP23S17 library.
from adafruit_mcp230xx.mcp23s17 import MCP23S17

# (RPi.GPIO imports and interrupt-related code are commented out for now.)
# import RPi.GPIO as GPIO

# ---------------------------------------------------------------------------
# Helper functions for low-level register access to the MCP23S17.
#
# The MCP23S17 expects an opcode of the form:
#   0b0100[A2][A1][A0][R/W]
#
# Here we assume the DIP switches give address 0 (i.e. A2,A1,A0=0):
#   WRITE_OPCODE = 0x40, READ_OPCODE = 0x41.
# ---------------------------------------------------------------------------
WRITE_OPCODE = 0x40  # for address 0, write (R/W bit = 0)
READ_OPCODE  = 0x41  # for address 0, read  (R/W bit = 1)

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

# ---------------------------------------------------------------------------
# Setup SPI bus and chip-select pin.
# ---------------------------------------------------------------------------
spi = busio.SPI(clock=board.SCK, MOSI=board.MOSI, MISO=board.MISO)
while not spi.try_lock():
    pass
spi.configure(baudrate=1000000)  # 1 MHz
spi.unlock()

cs = digitalio.DigitalInOut(board.D8)
cs.direction = digitalio.Direction.OUTPUT
cs.value = True  # Deassert CS initially

# ---------------------------------------------------------------------------
# Setup the reset pin on D24.
# ---------------------------------------------------------------------------
reset = digitalio.DigitalInOut(board.D24)
reset.direction = digitalio.Direction.OUTPUT


# ---------------------------------------------------------------------------
# Reset the MCP23S17.
# ---------------------------------------------------------------------------
print("Resetting MCP23S17 using D24...")
reset.value = False
time.sleep(0.1)
reset.value = True
time.sleep(0.1)

# ---------------------------------------------------------------------------
# Initialize the MCP23S17 using the Adafruit library.
# ---------------------------------------------------------------------------
mcp = MCP23S17(spi, cs, address=0)

# ---------------------------------------------------------------------------
# Configure Port A:
# Use only pins 0–6 (7 outputs). (Pin 7 is left unconfigured.)
# ---------------------------------------------------------------------------
print("Configuring Port A (pins 0-6) as outputs:")
for i in range(7):
    pin = mcp.get_pin(i)
    pin.switch_to_output(value=False)
    print(f"  MCP pin {i} set as OUTPUT")

# ---------------------------------------------------------------------------
# Configure Port B:
#
# We'll use:
#  - Pins 15 and 14 (physical PB7 and PB6) as inputs with pull-ups.
#  - Pins 13,12,11,10,9 as outputs (set low to keep LEDs off).
# ---------------------------------------------------------------------------
print("Configuring Port B:")
# Configure inputs:
for pin_num in [14]:
    pin = mcp.get_pin(pin_num)
    pin.switch_to_input()
    print(f"  MCP pin {pin_num} set as INPUT with pull-up")

# Configure remaining Port B pins as outputs:
for pin_num in [13, 13, 12, 11, 10, 9]:
    pin = mcp.get_pin(pin_num)
    pin.switch_to_output(value=False)
    print(f"  MCP pin {pin_num} set as OUTPUT (LED off)")

# ---------------------------------------------------------------------------
# Configure the MCP23S17 interrupts for Port B.
#
# (Interrupt configuration remains as before even if we won't use RPi.GPIO interrupts.)
# ---------------------------------------------------------------------------
print("Configuring MCP23S17 interrupt registers...")
mcp_write_register(spi, cs, 0x05, 0xC0)  # GPINTENB = 0xC0: enable interrupts on pins 15 and 14.
mcp_write_register(spi, cs, 0x09, 0x00)  # INTCONB = 0x00: interrupt on any change.
mcp_write_register(spi, cs, 0x0A, 0x44)  # IOCON = 0x44: open-drain and mirrored interrupts.
# Clear any existing interrupts.
_ = mcp_read_register(spi, cs, 0x0C)  # INTCAPA
_ = mcp_read_register(spi, cs, 0x0D)  # INTCAPB

print("Interrupt registers configured.\n")

# ---------------------------------------------------------------------------
# Perform one output cycle on Port A:
# Turn on each output (pins 0-6) one at a time (over 1 second total), then turn them off.
# ---------------------------------------------------------------------------
print("Performing one output cycle on Port A:")
outputs = [mcp.get_pin(i) for i in range(7)]
delay = 1.0 / len(outputs)
for idx, pin in enumerate(outputs):
    pin.value = True
    print(f"  Turning ON MCP output pin {idx}")
    time.sleep(delay)
for idx, pin in enumerate(outputs):
    pin.value = False
    print(f"  Turning OFF MCP output pin {idx}")

print("Output cycle complete.\n")

# ---------------------------------------------------------------------------
# Polling loop: check for changes on the input pins (MCP pins 15 and 14).
# ---------------------------------------------------------------------------
print("Polling for input changes on MCP pins 15 and 14...")
last_values = {}
# Initialize with the current state.
for pin_num in [14]:
    last_values[pin_num] = mcp.get_pin(pin_num).value

try:
    while True:
        for pin_num in [14]:
            current_value = mcp.get_pin(pin_num).value
            print(current_value)
            if current_value != last_values[pin_num]:
                print(f"Input change detected on MCP pin {pin_num}: {last_values[pin_num]} -> {current_value}")
                last_values[pin_num] = current_value
                outputs[0].value = current_value
        time.sleep(0.1)  # Poll every 100ms.
except KeyboardInterrupt:
    print("Exiting program...")

# (RPi.GPIO cleanup is not necessary now since we're not using it.)
