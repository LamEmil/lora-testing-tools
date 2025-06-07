import sys
import json
import uuid
import random
import urllib.request
import urllib.parse
import websocket
import threading
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QGridLayout, QScrollArea, QMessageBox,
    QFormLayout
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QPixmap, QImage
import copy
import os

# --- Configuration ---
SERVER_ADDRESS = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "flux_dev_example_with_upscale.json")

# --- Load ComfyUI Workflow from JSON file ---
def load_workflow_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- ComfyUI API Communication ---

def queue_prompt(prompt_workflow_str):
    """Queues a prompt on the ComfyUI server."""
    try:
        # The workflow is already a string, so we load it into a dictionary here
        # just for the purpose of wrapping it in the required API structure.
        prompt = json.loads(prompt_workflow_str)
        p = {"prompt": prompt, "client_id": CLIENT_ID}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def get_image(filename, subfolder, folder_type):
    """Fetches an image from the ComfyUI server."""
    try:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/view?{url_values}") as response:
            return response.read()
    except Exception as e:
        print(f"Error getting image: {e}")
        return None

def get_history(prompt_id):
    """Gets the history for a given prompt ID."""
    try:
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error getting history: {e}")
        return None


class ComfyUIWorker(QObject):
    """
    Worker thread to handle communication with the ComfyUI API,
    preventing the UI from freezing.
    """
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, lora_models, prompt, seed, workflow_dict):
        super().__init__()
        self.lora_models = lora_models
        self.prompt_text = prompt
        self.seed = seed
        self.workflow_dict = workflow_dict

    def run(self):
        """
        Main worker function. Generates one image for each LoRA model by
        updating the workflow dict loaded from file.
        """
        if not self.workflow_dict:
            self.error.emit("The workflow file is missing or invalid.")
            return

        images = []
        ws = None
        try:
            ws = websocket.WebSocket()
            ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")

            for lora_model in self.lora_models:
                print(f"Generating image for LoRA: {lora_model} with prompt: '{self.prompt_text}'")

                # Deep copy the workflow for each LoRA
                wf = copy.deepcopy(self.workflow_dict)

                # Update the workflow fields
                wf["66"]["inputs"]["lora_name"] = lora_model
                wf["6"]["inputs"]["text"] = self.prompt_text
                wf["25"]["inputs"]["noise_seed"] = self.seed

                # Dump to JSON string
                prompt_workflow_str = json.dumps(wf)

                # Queue the prompt
                queued_data = queue_prompt(prompt_workflow_str)
                if not queued_data or 'prompt_id' not in queued_data:
                    error_details = queued_data.get('error', {}).get('details', '') if queued_data else "No response from server."
                    self.error.emit(f"Failed to queue prompt for LoRA: {lora_model}\n\nDetails: {error_details}")
                    continue

                prompt_id = queued_data['prompt_id']

                # Wait for execution to finish by listening to the websocket
                while True:
                    out = ws.recv()
                    if isinstance(out, str):
                        message = json.loads(out)
                        if message['type'] == 'executing':
                            data = message['data']
                            if data['node'] is None and data['prompt_id'] == prompt_id:
                                break  # Execution is done
                    else:
                        continue # Ignore binary preview data

                # Get history and retrieve the generated image
                history = get_history(prompt_id).get(prompt_id)
                if not history:
                    self.error.emit(f"Could not retrieve history for prompt_id: {prompt_id}")
                    continue

                # Find the output image from any node
                for node_id in history['outputs']:
                    node_output = history['outputs'][node_id]
                    if 'images' in node_output:
                        for image_info in node_output['images']:
                            image_data = get_image(image_info['filename'], image_info['subfolder'], image_info['type'])
                            if image_data:
                                images.append({'lora': lora_model, 'image_data': image_data})
                                break
                        break

            self.finished.emit(images)

        except Exception as e:
            self.error.emit(f"An error occurred during generation: {e}")
        finally:
            if ws:
                ws.close()


class ImageLabel(QLabel):
    """
    A custom QLabel that can be clicked and stores associated data.
    """
    clicked = Signal(str)

    def __init__(self, image_data, lora_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_model = lora_model
        
        q_image = QImage()
        q_image.loadFromData(image_data)
        pixmap = QPixmap(q_image)
        
        self.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setCursor(Qt.PointingHandCursor)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #333; border-radius: 8px;")

    def mousePressEvent(self, event):
        self.clicked.emit(self.lora_model)
        self.setStyleSheet("border: 4px solid #00aaff; border-radius: 8px;")


class LoraVoteApp(QMainWindow):
    """
    Main application window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoRA Model Voter")
        self.setGeometry(100, 100, 1200, 800)

        # --- Prompts ---
        self.prompts = [
            "A majestic lion in a lush jungle, cinematic lighting",
            "A futuristic city skyline at dusk, with flying vehicles",
            "An enchanted forest with glowing mushrooms and mystical creatures",
            "A portrait of a wise old wizard with a long white beard",
            "A tranquil beach scene with crystal clear water and a beautiful sunset",
            "A cyberpunk street market at night, neon signs reflecting in puddles",
            "A detailed steampunk airship flying through the clouds",
            "A dramatic fantasy battle between knights and a dragon",
            "A cozy, cluttered artist's studio filled with paintings and sculptures",
            "An astronaut planting a flag on a vibrant, alien planet"
        ]
        self.current_prompt_index = 0

        # --- State ---
        self.lora_inputs = []
        self.lora_scores = {}
        self.worker_thread = None

        # --- Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- Left Panel (Controls) ---
        self.left_panel = QWidget()
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        self.left_panel.setFixedWidth(350)

        # LoRA inputs
        self.lora_form_layout = QFormLayout()
        self.add_lora_input() # Start with one input

        # Add LoRA button
        self.add_lora_button = QPushButton("+ Add Another LoRA Model")
        self.add_lora_button.clicked.connect(self.add_lora_input)

        # Generate button
        self.generate_button = QPushButton("Start Voting Session")
        self.generate_button.setStyleSheet("font-size: 16px; padding: 10px;")
        self.generate_button.clicked.connect(self.start_generation_round)

        # Status Label
        self.status_label = QLabel("Add at least two LoRA models to begin.")
        self.status_label.setWordWrap(True)
        
        self.left_panel_layout.addLayout(self.lora_form_layout)
        self.left_panel_layout.addWidget(self.add_lora_button)
        self.left_panel_layout.addWidget(self.generate_button)
        self.left_panel_layout.addWidget(self.status_label)
        self.left_panel_layout.addStretch()

        # --- Right Panel (Image Grid) ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_grid_container = QWidget()
        self.image_grid_layout = QGridLayout(self.image_grid_container)
        self.scroll_area.setWidget(self.image_grid_container)
        
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.scroll_area)

        # Load the workflow once at startup
        try:
            self.workflow_dict = load_workflow_from_file(WORKFLOW_PATH)
        except Exception as e:
            self.workflow_dict = None
            QMessageBox.critical(self, "Workflow Load Error", f"Could not load workflow file:\n{e}")

    def add_lora_input(self):
        """Adds a new QLineEdit for a LoRA model path."""
        new_lora_input = QLineEdit()
        new_lora_input.setPlaceholderText(r"e.g., loras\my_lora.safetensors")
        self.lora_inputs.append(new_lora_input)
        self.lora_form_layout.addRow(f"LoRA Model {len(self.lora_inputs)}:", new_lora_input)

    def start_generation_round(self):
        """Kicks off a round of image generation."""
        lora_models = [inp.text().strip() for inp in self.lora_inputs if inp.text().strip()]
        if len(lora_models) < 2:
            self.show_message("Error", "Please provide at least two valid LoRA model paths.")
            return
            
        if self.current_prompt_index == 0:
            # First round, initialize scores
            self.lora_scores = {model: 0 for model in lora_models}

        self.set_ui_for_loading(True)
        prompt = self.prompts[self.current_prompt_index]
        seed = random.randint(0, 2**32 - 1)
        
        self.status_label.setText(f"Round {self.current_prompt_index + 1}/10\n\nPrompt: '{prompt}'\n\nGenerating images with seed: {seed}...")

        # Run API calls in a separate thread
        self.worker = ComfyUIWorker(lora_models, prompt, seed, self.workflow_dict)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_generation_finished(self, images):
        """Handles the results from the worker thread."""
        self.set_ui_for_loading(False)
        if not images:
            self.on_generation_error("No images were generated. Check your server and paths.")
            return

        self.clear_image_grid()
        random.shuffle(images)
        
        self.status_label.setText(f"Round {self.current_prompt_index + 1}/10\n\nWhich image best fits the prompt?\n\n'{self.prompts[self.current_prompt_index]}'")

        # Display images in the grid
        row, col = 0, 0
        for img_info in images:
            image_label = ImageLabel(img_info['image_data'], img_info['lora'])
            image_label.clicked.connect(self.image_voted)
            self.image_grid_layout.addWidget(image_label, row, col)
            col += 1
            if col > 2: # 3 images per row
                col = 0
                row += 1

    def on_generation_error(self, error_message):
        """Shows an error message from the worker thread."""
        self.set_ui_for_loading(False)
        self.show_message("Error", error_message)

    def image_voted(self, lora_model):
        """Handles the user clicking on an image to vote."""
        self.lora_scores[lora_model] += 1
        print(f"Voted for: {lora_model}. Current scores: {self.lora_scores}")

        self.current_prompt_index += 1
        if self.current_prompt_index >= len(self.prompts):
            self.show_final_scores()
        else:
            self.start_generation_round()

    def show_final_scores(self):
        """Displays the final voting results."""
        self.clear_image_grid()
        
        # Sort scores from best to worst
        sorted_scores = sorted(self.lora_scores.items(), key=lambda item: item[1], reverse=True)

        results_text = "Voting Complete! Here are the final scores:\n\n"
        for i, (model, score) in enumerate(sorted_scores):
            results_text += f"{i+1}. {model}  -  Votes: {score}\n"
            
        self.status_label.setText("Session finished. Start a new one with the button above.")
        self.show_message("Final Scores", results_text)

        # Reset for a new session
        self.current_prompt_index = 0
        self.lora_scores = {}
        self.generate_button.setText("Start New Voting Session")


    def set_ui_for_loading(self, is_loading):
        """Enables/disables UI elements during processing."""
        self.generate_button.setEnabled(not is_loading)
        self.add_lora_button.setEnabled(not is_loading)
        for inp in self.lora_inputs:
            inp.setEnabled(not is_loading)
        
        if is_loading:
            self.clear_image_grid()
            loading_label = QLabel("Loading, please wait...")
            loading_label.setAlignment(Qt.AlignCenter)
            loading_label.setStyleSheet("font-size: 24px;")
            self.image_grid_layout.addWidget(loading_label, 0, 0)

    def clear_image_grid(self):
        """Removes all widgets from the image grid layout."""
        while self.image_grid_layout.count():
            child = self.image_grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def show_message(self, title, message):
        """Displays a simple message box."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LoraVoteApp()
    window.show()
    sys.exit(app.exec())
