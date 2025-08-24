import pytesseract
import pyautogui
from PIL import Image, ImageEnhance, ImageFilter
import time
import re
import sys
import os
import random
from datetime import datetime
import cv2
import numpy as np
import pytesseract
import pyautogui
import os
import time
import sys

# Set Tesseract path - now correctly installed
# NOTE: You may need to change this path depending on where Tesseract is installed on your VM.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Verify Tesseract is working
try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
except Exception as e:
    print(f"ERROR: Tesseract not working properly: {e}")
    print("Please check installation at: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    # sys.exit() is commented out so the program can still run for demonstration purposes
    # even if Tesseract isn't configured correctly.
    # For your project, you might want to re-enable this.

class GameAutomation:
    """
    Automates gameplay for a Miscrits-style game for a cybersecurity project.
    
    This script performs actions like attacking, healing, and capturing miscrits based on
    screen analysis. It is designed to be the 'attacker' that a separate 'defender' program
    will try to stop or detect.
    """
    def __init__(self):
        # Disable pyautogui failsafe for automation
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        # === Time Delay Settings (Easily Editable) ===
        # These delays are crucial for simulating human-like behavior and for
        # giving the 'defender' program a chance to react.
        self.turn_wait_time = 7
        self.right_button_delay = 1
        self.capture_screenshot_delay = 10 
        self.capture_button_delay = 9
        
        # Random delay range after a normal attack
        self.min_battle_delay = 10
        self.max_battle_delay = 15
        
        # Random delay range after clicking 'Continue' and before 'Search'
        self.min_search_delay = 20
        self.max_search_delay = 45
        
        # Fixed delay after clicking 'Search' or 'Keep'
        self.fixed_post_click_delay = 7

        # UPDATED: Delay before clicking the search button (increased to 3 seconds)
        self.search_button_pre_click_delay = 3
        # NEW: Hold time for the search button in milliseconds (100ms = 0.1s)
        self.search_button_hold_time = 0.1
        # NEW: Delay between 'keep' and 'continue' buttons
        self.keep_to_continue_delay = 4
        # NEW: Delay for miscrit training check
        self.training_check_delay = 4
        
        # Fixed button coordinates
        # These coordinates must be calibrated to your screen resolution and game window.
        # You can use a tool like "pyautogui.displayMousePosition()" to find them.
        self.normal_attack_button = (491, 922)       # Mighty Bash
        self.right_button = (1644, 929)             # Right button
        self.left_button = (277, 933)               # Left button
        self.capture_attack_button = (1450, 925)     # Small attack for capture
        self.capture_button = (958, 198)             # Capture button
        self.keep_button = (863, 728)               # Keep button for captured miscrits
        self.continue_button = (953, 883)            # Continue button on post-battle screen
        self.search_button = (1004, 422)             # Search button for new miscrits
        self.healing_potion_coord = (1251, 97)       # Healing potion
        self.heal_now_continue_button = (887, 607) # NEW: Continue button after healing
        
        # UPDATED: Training sequence coordinates
        self.sub_miscrit_coord = (520, 56)           # Coordinate for the sub miscrit
        self.train_button = (984, 120)               # Coordinate for the train button
        self.confirm_train_button = (1130, 739)      # Coordinate for the confirm train button
        self.new_button = (1141, 741)                # Coordinate for the 'new' button on a miscrit
        self.train_miscrit_continue_button = (1159, 951) # NEW: Continue button on miscrit training screen
        self.close_button = (1496, 76)               # Coordinate for the close button
        
        # === NEW ===
        self.evolve_okay_button = (963, 892)
        self.next_click_after_evolve = (962, 763)
        # === END NEW ===

        # OCR regions (left, top, right, bottom)
        # These are crucial for the script's decision-making process.
        self.miscrit_name_region = (1340, 65, 1508, 92)       # Miscrit name area
        self.capture_percent_region = (914, 224, 1005, 255)    # Capture % area
        self.continue_text_region = (871, 859, 1046, 904)
        self.afterburn_region = (443, 65, 548, 92)
        self.ready_to_train_region = (556, 154, 707, 180)
        
        # UPDATED: New text region based on user feedback
        self.new_text_region = (690, 292, 784, 330)
        
        # Region for own miscrit health
        self.own_miscrit_health_region = (558, 118, 597, 152) 
        # UPDATED: Region for target miscrit health using coordinates from user
        # (left, top, width, height) = (min_x, min_y, max_x-min_x, max_y-min_y)
        self.target_miscrit_health_region = (1470, 97, 1510, 127)
        # === NEW ===
        self.miscrit_evolve_region = (909, 874, 1013, 911)
        # === END NEW ===
        
        # Screenshot saving
        self.save_screenshots = True
        self.screenshot_folder = r"E:\data\miscrits"
        self.create_screenshot_folder()
        
    def test_ocr_regions(self):
        """Test OCR regions and save images for debugging"""
        print("Testing OCR regions...")
        
        # Test name region
        print("Testing miscrit name region...")
        name_screenshot = self.take_partial_screenshot(self.miscrit_name_region)
        if name_screenshot:
            name_screenshot.save("debug_name_region_original.png")
            print("Saved: debug_name_region_original.png")
        
        # Test capture region  
        print("Testing capture percentage region...")
        capture_screenshot = self.take_partial_screenshot(self.capture_percent_region)
        if capture_screenshot:
            capture_screenshot.save("debug_capture_region_original.png")
            processed_capture = self.preprocess_capture_image(capture_screenshot)
            processed_capture.save("debug_capture_region_processed.png")
            print("Saved: debug_capture_region_original.png and debug_capture_region_processed.png")
        
        # Test continue button text region
        print("Testing continue button text region...")
        continue_text_ss = self.take_partial_screenshot(self.continue_text_region)
        if continue_text_ss:
            continue_text_ss.save("debug_continue_text_region.png")
            print("Saved: debug_continue_text_region.png")
        
        # Test miscrit health regions
        print("Testing own miscrit health region...")
        own_health_ss = self.take_partial_screenshot(self.own_miscrit_health_region)
        if own_health_ss:
            own_health_ss.save("debug_own_health_region.png")
            print("Saved: debug_own_health_region.png")
        
        print("Testing target miscrit health region...")
        target_health_ss = self.take_partial_screenshot(self.target_miscrit_health_region)
        if target_health_ss:
            target_health_ss.save("debug_target_health_region.png")
            print("Saved: debug_target_health_region.png")

        # Test full screenshot
        full_screenshot = pyautogui.screenshot()
        full_screenshot.save("debug_full_screenshot.png")
        print("Saved: debug_full_screenshot.png")
        
        print("Check these images to verify the regions are correct!")
    
    def create_screenshot_folder(self):
        """Create screenshot folder if it doesn't exist"""
        try:
            if not os.path.exists(self.screenshot_folder):
                os.makedirs(self.screenshot_folder)
                print(f"Created screenshot folder: {self.screenshot_folder}")
        except Exception as e:
            print(f"Error creating screenshot folder: {e}")
            self.save_screenshots = False
    
    def save_screenshot(self, image, prefix="screenshot"):
        """Save screenshot with timestamp"""
        if not self.save_screenshots:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_folder, filename)
            
            if isinstance(image, Image.Image):
                 image.save(filepath)
            elif isinstance(image, np.ndarray):
                cv2.imwrite(filepath, image)
            
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    def get_user_input(self):
        """Get only target miscrit name"""
        print("=== GAME AUTOMATION SETUP ===")
        print("\nFixed Attack Logic:")
        print("- Normal miscrits: Mighty Bash -> Check for 'Continue' -> 'Search'")
        print("- Target miscrit: Mighty Bash -> Right Button -> Capture Attack (until 100%) -> Capture -> 'Keep' -> 'Continue'")
        
        self.target_miscrit = input("\nEnter target miscrit name to capture: ").strip().lower()
        
        print(f"\n=== CONFIGURATION ===")
        print(f"Target Miscrit: {self.target_miscrit}")
        print(f"OCR Regions:")
        print(f"  Name: {self.miscrit_name_region}")
        print(f"  Capture %: {self.capture_percent_region}")
        print(f"  Continue Text: {self.continue_text_region}")
        print(f"  Afterburn Text: {self.afterburn_region}")
        
        confirm = input("\nStart automation? (y/n) or 't' to test regions: ").strip().lower()
        if confirm == 't':
            self.test_ocr_regions()
            sys.exit()
        elif confirm != 'y':
            print("Automation cancelled.")
            sys.exit()
        
        # Added a 5-second timer to allow user to switch to the game window
        print("Starting in 5 seconds... Switch to your game window now!")
        time.sleep(5)

    def take_partial_screenshot(self, region):
        """Take screenshot of specific region"""
        try:
            # region format: (left, top, right, bottom)
            left, top, right, bottom = region
            width = right - left
            height = bottom - top
            
            # PyAutoGUI uses (left, top, width, height) format
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            return screenshot
        except Exception as e:
            print(f"Error taking partial screenshot: {e}")
            return None
    
    def extract_number_from_image(self, image_path):   
        try:
        # Load the image using OpenCV
            image = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
            if image is None:
                print(f"Error: Unable to load image at {image_path}")
                return None

        # Convert the image to grayscale, which is a common pre-processing step for OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to create a black and white image,
        # which helps to isolate the text from the background.
        # THRESH_OTSU automatically determines the optimal threshold value.
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Use pytesseract to perform OCR on the pre-processed image.
        # We use a specific configuration to improve accuracy for numbers:
        # --psm 6: Assumes the image contains a single uniform block of text.
        # --oem 3: Uses the default Tesseract OCR engine mode.
        # -c tessedit_char_whitelist=0123456789: Restricts Tesseract to only look for digits.
            text = pytesseract.image_to_string(thresh, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        
        # Use a regular expression to find all sequences of digits in the output.
        # This helps to clean up any non-numeric characters that might have been
        # picked up by the OCR process.
            numbers = re.findall(r'\d+', text)
        
        # Join the found digits to form a single number string.
        # This is useful in case the OCR returns the number with spaces.
            if numbers:
                return ''.join(numbers)
            else:
                return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_text_from_region(self, region):
        """Read text from a general region with enhanced OCR"""
        try:
            screenshot = self.take_partial_screenshot(region)
            if not screenshot:
                return ""
                
            self.save_screenshot(screenshot, f"ocr_region_original_{region[0]}")
            
            # Preprocess the image for better OCR
            processed_image = screenshot.convert('L')
            processed_image = ImageEnhance.Contrast(processed_image).enhance(2.0)
            processed_image = ImageEnhance.Brightness(processed_image).enhance(1.5)
            processed_image = processed_image.filter(ImageFilter.SHARPEN)
            processed_image = processed_image.resize(
                (processed_image.width * 4, processed_image.height * 4), Image.Resampling.LANCZOS
            )
            self.save_screenshot(processed_image, f"ocr_region_processed_{region[0]}")
            image_to_ocr = processed_image
            
            # Use specific OCR configs
            configs = [
                '--psm 6', # General text, single block
                '--psm 7', # Single line
                '--psm 8', # Single word
                '--psm 13', # Raw line
            ]
            
            best_result = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image_to_ocr, config=config)
                    cleaned = text.strip().lower()
                    if len(cleaned) > len(best_result):
                        best_result = cleaned
                except:
                    continue
                    
            return best_result

        except Exception as e:
            print(f"Error reading text from region: {e}")
            return ""

    def get_miscrit_name(self):
        """Get miscrit name from specific screen region without preprocessing"""
        try:
            screenshot = self.take_partial_screenshot(self.miscrit_name_region)
            if not screenshot:
                return ""
            
            self.save_screenshot(screenshot, "name_region_original")
            
            # Use original screenshot directly with Tesseract
            configs = [
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                '--psm 8',
                '--psm 7'
            ]
            
            best_result = ""
            for config in configs:
                try:
                    text = pytesseract.image_to_string(screenshot, config=config)
                    cleaned = text.strip().lower()
                    cleaned = re.sub(r'[^a-zA-Z0-9]', '', cleaned)
                    
                    if len(cleaned) > len(best_result) and len(cleaned) > 2:
                        best_result = cleaned
                except:
                    continue
            
            print(f"OCR attempts: {[pytesseract.image_to_string(screenshot, config=config).strip() for config in configs[:3]]}")
            print(f"Best result: '{best_result}'")
            
            return best_result
        except Exception as e:
            print(f"Error reading miscrit name: {e}")
            return ""
    
    def get_capture_percentage(self):
        """
        Get capture percentage from specific screen region with dedicated preprocessing.
        Now includes a more robust preprocessing step using OpenCV for better OCR.
        """
        try:
            screenshot = self.take_partial_screenshot(self.capture_percent_region)
            if not screenshot:
                return 0
            
            # Save the original screenshot
            self.save_screenshot(screenshot, "capture_region_original")
            
            # Convert the PIL Image to a NumPy array for OpenCV processing
            img_np = np.array(screenshot.convert('RGB'))
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # === NEW Preprocessing steps using OpenCV ===
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Thresholding to create a binary (black and white) image
            # The background is dark, so we want to invert it to make the text white
            _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Dilate to make the white text thicker and connect any broken characters
            kernel = np.ones((1,1), np.uint8)
            processed_image_cv = cv2.dilate(binary, kernel, iterations=1)
            
            # Save the processed image for debugging
            self.save_screenshot(processed_image_cv, "capture_region_processed")
            
            # Now, use Tesseract on the processed image
            configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789%',
                '--psm 7 -c tessedit_char_whitelist=0123456789%',
                '--psm 6 -c tessedit_char_whitelist=0123456789%',
                '--psm 13 -c tessedit_char_whitelist=0123456789%',
                '-c tessedit_char_whitelist=0123456789%'
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_image_cv, config=config)
                    print(f"Capture OCR: '{text.strip()}' (config: {config[:8]})")
                    
                    percentage_pattern = r'(\d+)%'
                    matches = re.findall(percentage_pattern, text)
                    if matches:
                        percentage = int(matches[0])
                        if 0 <= percentage <= 100:  # Check if the number is in a valid range
                            print(f"✅ Detected capture percentage: {percentage}%")
                            return percentage
                        else:
                            print(f"⚠️ Invalid percentage detected ({percentage}%). Ignoring and trying again.")
                            
                    number_pattern = r'(\d+)'
                    numbers = re.findall(number_pattern, text)
                    if numbers and len(numbers[0]) <= 3:
                        percentage = int(numbers[0])
                        if 0 <= percentage <= 100:  # Check if the number is in a valid range
                            print(f"✅ Detected number (assuming %): {percentage}")
                            return percentage
                        else:
                            print(f"⚠️ Invalid number detected ({percentage}). Ignoring and trying again.")
                except Exception as e:
                    print(f"OCR attempt failed: {e}")
                    continue
                    
            print("❌ Could not detect capture percentage")
            return 0
        except Exception as e:
            print(f"Error reading capture percentage: {e}")
            return 0
            
    def check_afterburn_text(self):
        """Check for 'Afterburn' text in a specific region"""
        try:
            # Use the dedicated OCR method for general text
            text = self.get_text_from_region(self.afterburn_region)
            
            if "afterburn" in text.lower():
                print("🔥 'Afterburn' text detected. Starting battle.")
                return True
            else:
                print("❌ 'Afterburn' text not detected.")
                return False
        except Exception as e:
            print(f"Error checking for 'Afterburn' text: {e}")
            return False

    def click_button(self, coordinates, button_name, pre_click_delay=0, hold_time=0):
        """
        Moves cursor to a location, waits, and then clicks the button at specified coordinates.
        Includes an optional hold time for extended clicks.
        """
        try:
            x, y = coordinates
            if pre_click_delay > 0:
                print(f"Moving cursor to {button_name} at ({x}, {y})...")
                pyautogui.moveTo(x, y)
                print(f"Waiting {pre_click_delay} seconds before clicking...")
                time.sleep(pre_click_delay)

            if hold_time > 0:
                # Perform an extended click (mouse down, wait, mouse up)
                print(f"Performing extended click (hold for {hold_time*1000}ms) on {button_name}.")
                pyautogui.mouseDown(x, y)
                time.sleep(hold_time)
                pyautogui.mouseUp(x, y)
            else:
                # Perform a standard click
                pyautogui.click(x, y)
            
            print(f"Clicked {button_name} at ({x}, {y})")
            return True
        except Exception as e:
            print(f"Error clicking {button_name}: {e}")
            return False

    def get_number_from_region_and_file(self, region, debug_name):
        """Helper function to take a screenshot, save it, and extract a number."""
        try:
            screenshot = self.take_partial_screenshot(region)
            if not screenshot:
                print(f"Error: Could not take screenshot for {debug_name}")
                return 0
            
            # Save the screenshot with a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_name}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_folder, filename)
            screenshot.save(filepath)
            
            print(f"Screenshot saved: {filename}")
            
            # Use the existing function to extract the number from the saved file
            number_str = self.extract_number_from_image(filepath)
            
            if number_str and number_str.isdigit():
                return int(number_str)
            else:
                return 0
        except Exception as e:
            print(f"Error in get_number_from_region_and_file: {e}")
            return 0
    
    # NEW FUNCTION
    def check_own_miscrit_health(self):
        """
        Checks the health of the player's miscrit after a fight.
        If below 50, it clicks the healing potion.
        """
        print("\nChecking own miscrit's health for healing...")
        pyautogui.moveTo(453, 69)  # Hover over the specified location
        print("Hovering over own miscrit health area.")
        time.sleep(2)  # Wait for 2 seconds as requested
        
        health = self.get_number_from_region_and_file(self.own_miscrit_health_region, "own_miscrit_health")
        
        if health > 0 and health < 50:
            print(f"⚠️ Own miscrit's health is {health}. Healing now.")
            self.click_button(self.healing_potion_coord, "healing potion")
            time.sleep(2)  # Wait for heal animation
            
            # --- NEW: Clicking the 'Heal Now Continue' button ---
            print("Waiting 2 seconds for 'Heal Now' button to appear...")
            time.sleep(2)
            self.click_button(self.heal_now_continue_button, "Heal Now Continue button")
            # --- End of new logic ---
            
            return True
        elif health >= 50:
            print(f"✅ Own miscrit's health is {health}. No healing needed.")
        else:
            print("❌ Could not detect own miscrit's health.")
            
        return False
    
    # NEW FUNCTION
    def check_target_miscrit_health(self):
        """
        Checks the health of the target miscrit.
        Returns the detected health or 0 if not found.
        """
        print("\nChecking target miscrit's health...")
        # Now using the updated region from the user's feedback
        health = self.get_number_from_region_and_file(self.target_miscrit_health_region, "target_miscrit_health")
        return health
        
    def run_automation(self):
        """Main automation loop"""
        print("\n=== STARTING AUTOMATION ===")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n{'='*50}")
                print("Taking screenshot and analyzing...")
                
                full_screenshot = pyautogui.screenshot()
                self.save_screenshot(full_screenshot, "full_game_screenshot")
                
                current_miscrit = self.get_miscrit_name()
                
                is_target = self.target_miscrit in current_miscrit if current_miscrit else False
                
                print(f"Current miscrit: '{current_miscrit}'")
                print(f"Is target: {is_target}")
                
                if is_target:
                    print("\n[TARGET] TARGET MISCRIT DETECTED!")
                    print("Starting capture sequence...")
                    
                    # Attack with Mighty Bash once
                    print("Using Mighty Bash once...")
                    self.click_button(self.normal_attack_button, "mighty bash")
                    
                    # Wait for attack animation and turn to finish, plus the new screenshot delay
                    print(f"Waiting {self.capture_screenshot_delay} seconds to take capture screenshot...")
                    time.sleep(self.capture_screenshot_delay)
                    
                    # Click right button twice
                    print("Clicking right button twice...")
                    self.click_button(self.right_button, "right button")
                    time.sleep(1) # Delay between clicks
                    self.click_button(self.right_button, "right button")
                    time.sleep(self.right_button_delay)

                    # Now loop to use capture attack until conditions are met
                    while True:
                        # NEW LOGIC: Check both conditions before acting
                        target_health = self.check_target_miscrit_health()
                        capture_percent = self.get_capture_percentage()
                        
                        print(f"Current Target Health: {target_health}")
                        print(f"Current Capture Percentage: {capture_percent}")

                        if target_health > 0 and target_health < 20:
                            print("[ALERT] Target miscrit's health is below 20! Automatically attempting capture.")
                            self.click_button(self.capture_button, "capture")
                            
                            # Post-capture logic
                            print("Capture clicked. Starting post-capture sequence...")
                            print(f"Waiting {self.fixed_post_click_delay} seconds...")
                            time.sleep(self.fixed_post_click_delay)
                            self.click_button(self.keep_button, "keep")
                            print(f"Waiting {self.keep_to_continue_delay} seconds before clicking continue...")
                            time.sleep(self.keep_to_continue_delay) 
                            self.click_button(self.continue_button, "continue")
                            print("\n[SUCCESS] Capture sequence complete. Exiting script.")
                            sys.exit() # Stop the script
                        elif capture_percent >= 94:
                            print("[SUCCESS] CAPTURE PERCENTAGE AT 94% OR HIGHER! Attempting capture...")
                            
                            # Capture sequence
                            print(f"Waiting {self.capture_button_delay} seconds before clicking capture...")
                            time.sleep(self.capture_button_delay)
                            self.click_button(self.capture_button, "capture")
                            
                            # Post-capture logic
                            print("Capture clicked. Starting post-capture sequence...")
                            print(f"Waiting {self.fixed_post_click_delay} seconds...")
                            time.sleep(self.fixed_post_click_delay)
                            self.click_button(self.keep_button, "keep")
                            print(f"Waiting {self.keep_to_continue_delay} seconds before clicking continue...")
                            time.sleep(self.keep_to_continue_delay) 
                            self.click_button(self.continue_button, "continue")
                            print("\n[SUCCESS] Capture sequence complete. Exiting script.")
                            sys.exit() # Stop the script
                        else:
                            print(f"Capture percentage: {capture_percent}%. Target health: {target_health}%. Using capture attack again.")
                            self.click_button(self.capture_attack_button, "capture attack")
                            
                            # Wait for attack animation and new screenshot delay
                            print(f"Waiting {self.capture_screenshot_delay} seconds to take capture screenshot...")
                            time.sleep(self.capture_screenshot_delay)

                else:
                    print("\n[BATTLE] Normal battle - using Mighty Bash")
                    
                    # Loop for attacking until 'Continue' is found
                    while True:
                        self.click_button(self.normal_attack_button, "mighty bash")
                        random_wait = random.randint(self.min_battle_delay, self.max_battle_delay)
                        print(f"Waiting {random_wait} seconds for post-battle screen...")
                        time.sleep(random_wait)

                        # Check for target miscrit again after each attack
                        current_miscrit = self.get_miscrit_name()
                        is_target_now = self.target_miscrit in current_miscrit if current_miscrit else False
                        if is_target_now:
                            print(f"[TARGET] Switched targets! The miscrit is now '{current_miscrit}'.")
                            break
                            
                        continue_text = self.get_text_from_region(self.continue_text_region)
                        print(f"Text in continue region: '{continue_text}'")
                        
                        if "continue" in continue_text:
                            print("[SUCCESS] 'Continue' text detected. Clicking continue button.")
                            self.click_button(self.continue_button, "continue")
                            
                            # --- New healing check logic ---
                            time.sleep(4) # Wait 4 seconds as requested
                            self.check_own_miscrit_health()
                            # --- End of new logic ---
                            
                            break # Exit this inner loop
                        else:
                            print("[INFO] 'Continue' text not detected. Attacking again.")

                    # === UPDATED TRAINING SEQUENCE ===
                    print("\nChecking for miscrit training...")
                    print(f"Waiting {self.training_check_delay} seconds...")
                    time.sleep(self.training_check_delay)
                    
                    # Move cursor and wait
                    pyautogui.moveTo(self.sub_miscrit_coord)
                    print(f"Cursor moved to {self.sub_miscrit_coord}. Waiting 2 seconds before checking...")
                    time.sleep(2)
                    
                    # Check for "ready to train" text
                    train_text = self.get_text_from_region(self.ready_to_train_region)
                    
                    # Print the OCR result for debugging
                    print(f"OCR result from 'ready to train' region: '{train_text}'")
                    
                    # More robust check for "ready to train"
                    cleaned_train_text = re.sub(r'[^a-zA-Z]', '', train_text).lower()
                    if "readytotrain" in cleaned_train_text:
                        print("[TRAINING] 'Ready to train' detected! Starting training sequence.")
                        
                        # Click the sub miscrit button
                        self.click_button(self.sub_miscrit_coord, "sub miscrit")
                        time.sleep(3)
                        
                        # Click the "train" button
                        self.click_button(self.train_button, "train button")
                        time.sleep(3)
                        
                        # Click the "confirm train" button
                        self.click_button(self.confirm_train_button, "confirm train button")
                        time.sleep(2)
                        
                        # Click continue button
                        print("Clicking continue button.")
                        self.click_button(self.train_miscrit_continue_button, "train continue button")
                        time.sleep(2)
                        
                        # Check for "new" text in the specified region with improved OCR
                        new_region = (696, 292, 784, 330)  # tr, tl, bl, br coordinates converted to region
                        new_screenshot = self.take_partial_screenshot(new_region)
                        self.save_screenshot(new_screenshot, "new_text_check")
                        
                        # Use improved OCR method for "new" text detection
                        new_text = ""
                        if new_screenshot:
                            # Try multiple OCR configurations specifically for text detection
                            configs = [
                                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                '--psm 8',
                                '--psm 7',
                                '--psm 6'
                            ]
                            
                            best_result = ""
                            for config in configs:
                                try:
                                    text = pytesseract.image_to_string(new_screenshot, config=config)
                                    cleaned = text.strip().lower()
                                    cleaned = re.sub(r'[^a-zA-Z]', '', cleaned)
                                    
                                    print(f"OCR attempt with {config[:8]}: '{text.strip()}' -> cleaned: '{cleaned}'")
                                    
                                    if len(cleaned) > len(best_result) and len(cleaned) >= 2:
                                        best_result = cleaned
                                        
                                    # Check specifically for "new"
                                    if "new" in cleaned:
                                        new_text = cleaned
                                        break
                                        
                                except Exception as e:
                                    print(f"OCR attempt failed: {e}")
                                    continue
                            
                            if not new_text:
                                new_text = best_result
                        
                        print(f"Final OCR result from 'new' region: '{new_text}'")
                        
                        if "new" in new_text.lower():
                            print("[SUCCESS] 'New' text detected. Clicking new ability continue.")
                            self.click_button((1130, 739), "new ability continue")
                            time.sleep(2)
                            
                            # Check for first "okay" text in evolution region with improved OCR
                            evolution_region = (909, 877, 1015, 917)  # tr, br, bl, tl coordinates converted to region
                            evolution_screenshot = self.take_partial_screenshot(evolution_region)
                            self.save_screenshot(evolution_screenshot, "evolution_okay_check")
                            
                            # Use improved OCR method for "okay" text detection
                            evolution_text = ""
                            if evolution_screenshot:
                                configs = [
                                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                    '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                    '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                    '--psm 8',
                                    '--psm 7',
                                    '--psm 6'
                                ]
                                
                                best_result = ""
                                for config in configs:
                                    try:
                                        text = pytesseract.image_to_string(evolution_screenshot, config=config)
                                        cleaned = text.strip().lower()
                                        cleaned = re.sub(r'[^a-zA-Z]', '', cleaned)
                                        
                                        print(f"Evolution OCR attempt with {config[:8]}: '{text.strip()}' -> cleaned: '{cleaned}'")
                                        
                                        if len(cleaned) > len(best_result) and len(cleaned) >= 2:
                                            best_result = cleaned
                                            
                                        # Check specifically for "okay"
                                        if "okay" in cleaned or "ok" in cleaned:
                                            evolution_text = cleaned
                                            break
                                            
                                    except Exception as e:
                                        print(f"Evolution OCR attempt failed: {e}")
                                        continue
                                
                                if not evolution_text:
                                    evolution_text = best_result
                            
                            print(f"Final evolution OCR result: '{evolution_text}'")
                            
                            if "okay" in evolution_text.lower() or "ok" in evolution_text.lower():
                                print("[SUCCESS] First 'Okay' text detected. Clicking okay button.")
                                self.click_button((963, 887), "evolution okay button")
                                time.sleep(2)
                                self.click_button((1496, 76), "close button")
                            else:
                                print("[INFO] First 'Okay' text not found. Clicking close button.")
                                self.click_button((1496, 76), "close button")
                        else:
                            print("[INFO] 'New' text not found. Clicking close button.")
                            self.click_button((1496, 76), "close button")
                        
                        # Wait 2 seconds and check for second okay region with improved OCR
                        time.sleep(2)
                        second_okay_region = (909, 745, 1012, 787)  # tr, br, tl, bl coordinates converted to region
                        second_okay_screenshot = self.take_partial_screenshot(second_okay_region)
                        self.save_screenshot(second_okay_screenshot, "second_okay_check")
                        
                        # Use improved OCR method for second "okay" text detection
                        second_okay_text = ""
                        if second_okay_screenshot:
                            configs = [
                                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                                '--psm 8',
                                '--psm 7',
                                '--psm 6'
                            ]
                            
                            best_result = ""
                            for config in configs:
                                try:
                                    text = pytesseract.image_to_string(second_okay_screenshot, config=config)
                                    cleaned = text.strip().lower()
                                    cleaned = re.sub(r'[^a-zA-Z]', '', cleaned)
                                    
                                    print(f"Second okay OCR attempt with {config[:8]}: '{text.strip()}' -> cleaned: '{cleaned}'")
                                    
                                    if len(cleaned) > len(best_result) and len(cleaned) >= 2:
                                        best_result = cleaned
                                        
                                    # Check specifically for "okay"
                                    if "okay" in cleaned or "ok" in cleaned:
                                        second_okay_text = cleaned
                                        break
                                        
                                except Exception as e:
                                    print(f"Second okay OCR attempt failed: {e}")
                                    continue
                            
                            if not second_okay_text:
                                second_okay_text = best_result
                        
                        print(f"Final second okay OCR result: '{second_okay_text}'")
                        
                        # --- MODIFIED CODE BELOW ---
                        if "okay" in second_okay_text.lower() or "ok" in second_okay_text.lower():
                            print("[SUCCESS] Second 'Okay' text detected. Clicking okay button.")
                            self.click_button((962, 764), "second okay button")
                        else:
                            print("[INFO] Second 'Okay' text not found. Proceeding to search.")
                        # --- MODIFIED CODE ABOVE ---
                        
                        print("Training sequence complete.")
                    else:
                        print("[INFO] 'Ready to train' not detected. Skipping training sequence.")
                    # === END OF UPDATED TRAINING SEQUENCE ===
                    
                    # Loop for searching until a new miscrit is found
                    while True:
                        random_search_wait = random.randint(self.min_search_delay, self.max_search_delay)
                        print(f"Waiting {random_search_wait} seconds for search screen...")
                        time.sleep(random_search_wait)
                        
                        print("Clicking search button.")
                        # This is the new extended click
                        self.click_button(
                            self.search_button, 
                            "search", 
                            pre_click_delay=self.search_button_pre_click_delay,
                            hold_time=self.search_button_hold_time
                        )
                        
                        # New step: Wait 7 seconds and check for "Afterburn" text
                        print(f"Waiting 7 seconds to check for 'Afterburn'...")
                        time.sleep(7)
                        
                        if self.check_afterburn_text():
                            # If "Afterburn" is found, go back to the top of the main loop
                            break
                        else:
                            # If "Afterburn" is not found, retry searching
                            print("[INFO] 'Afterburn' was not detected. Retrying search.")
                            continue

        except KeyboardInterrupt:
            print("\n\nAutomation stopped by user.")
        except Exception as e:
            print(f"\nUnexpected error: {e}")

def main():
    # Initialize automation
    automation = GameAutomation()
    
    # Get user input
    automation.get_user_input()
    
    # Start automation
    automation.run_automation()

if __name__ == "__main__":
    main()