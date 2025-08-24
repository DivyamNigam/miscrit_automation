import pyautogui
import time

print("Move your mouse to the button and press Ctrl+C to get coordinates")
print("Waiting 3 seconds to start...")
time.sleep(3)

try:
    while True:
        x, y = pyautogui.position()
        print(f"Current position: ({x}, {y})", end='\r')
        time.sleep(0.1)
except KeyboardInterrupt:
    x, y = pyautogui.position()
    print(f"\nFinal coordinates: ({x}, {y})")