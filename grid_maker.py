import sys
import os
import json
import random
import copy
import pathlib
import urllib.request
import urllib.parse
import websocket
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor
from PySide6.QtCore import QRect, Qt, Signal, QObject, QThread, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QProgressBar, QTextEdit,
    QHBoxLayout  # <-- Add this import
)

# --- Configuration ---
# Ensure your ComfyUI server is running at this address
SERVER_ADDRESS = "127.0.0.1:8188"
# The workflow file must exist. This example uses a specific workflow.
# Make sure your workflow has the correctly numbered nodes for prompts, LoRAs, and seeds.
# Node "66": LoRA Loader (for lora_name)
# Node "6": Prompt Text (for text)
# Node "25": KSampler (for noise_seed)
WORKFLOW_PATH = os.path.join(os.path.dirname(__file__), "flux_dev_example_with_upscale.json")
OUTPUT_GRID_PATH = os.path.join(os.path.dirname(__file__), "comparison_grid.png")
CLIENT_ID = "grid_maker_client"

DEFAULT_PROMPTS = [
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

# --- IMPORTANT ---
# Add the full paths to your LoRA model files here.
# Use raw strings (r"...") on Windows to handle backslashes correctly.
# Example: r"C:\ComfyUI\models\loras\my_lora.safetensors"
LORA_MODELS = [
    # Add your LoRA model paths here
]

# --- Utility Functions ---
def load_workflow_from_file(path):
    """Loads a JSON workflow file."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def queue_prompt(prompt_workflow_str):
    """Sends a generation request to the ComfyUI server."""
    try:
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
    """Fetches a generated image from the ComfyUI server."""
    try:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/view?{url_values}") as response:
            return response.read()
    except Exception as e:
        print(f"Error getting image: {e}")
        return None

def get_history(prompt_id):
    """Retrieves the history for a specific prompt ID."""
    try:
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error getting history: {e}")
        return None

def draw_text_multiline_boxed(painter, rect, text, font, color=QColor("black")):
    """Draws text inside a QRect, wrapping it to fit."""
    painter.save()
    painter.setFont(font)
    painter.setPen(color)
    metrics = painter.fontMetrics()
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if metrics.horizontalAdvance(test_line) > rect.width() - 8 and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    # Forcibly break words that are too long
    final_lines = []
    for line in lines:
        if metrics.horizontalAdvance(line) > rect.width() - 8:
            temp_line = ""
            for char in line:
                if metrics.horizontalAdvance(temp_line + char) > rect.width() - 8:
                    final_lines.append(temp_line)
                    temp_line = char
                else:
                    temp_line += char
            if temp_line:
                final_lines.append(temp_line)
        else:
            final_lines.append(line)

    total_height = len(final_lines) * metrics.height()
    y = rect.y() + (rect.height() - total_height) // 2
    for line in final_lines:
        painter.drawText(rect.x() + 4, y + metrics.ascent(), line)
        y += metrics.height()
    painter.restore()


class GenerationWorker(QObject):
    """Worker thread for generating images via ComfyUI API."""
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, prompts, lora_models, workflow_dict):
        super().__init__()
        self.prompts = prompts
        self.lora_models = lora_models
        self.workflow_dict = workflow_dict
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        images_grid = []
        total = len(self.prompts) * len(self.lora_models)
        count = 0
        for prompt_idx, prompt in enumerate(self.prompts):
            if not self.is_running: break
            row_images = []
            seed = random.randint(0, 2**32 - 1) # Use the same seed for each prompt row
            for model_idx, lora_model in enumerate(self.lora_models):
                if not self.is_running: break
                count += 1
                self.progress.emit(count, total, f"Prompt {prompt_idx+1}/{len(self.prompts)}: {os.path.basename(lora_model)}")

                wf = copy.deepcopy(self.workflow_dict)
                try:
                    # Update workflow with current prompt, model, and seed
                    wf["66"]["inputs"]["lora_name"] = lora_model
                    wf["6"]["inputs"]["text"] = prompt
                    wf["25"]["inputs"]["noise_seed"] = seed
                except KeyError as e:
                    self.error.emit(f"Workflow is missing expected node ID: {e}. Check your workflow file.")
                    return

                prompt_workflow_str = json.dumps(wf)
                queued_data = queue_prompt(prompt_workflow_str)
                if not queued_data or 'prompt_id' not in queued_data:
                    self.error.emit(f"Failed to queue prompt for model {lora_model}")
                    row_images.append({'lora': lora_model, 'image_data': b''})
                    continue

                prompt_id = queued_data['prompt_id']
                ws = websocket.WebSocket()
                try:
                    ws.connect(f"ws://{SERVER_ADDRESS}/ws?clientId={CLIENT_ID}")
                    while self.is_running:
                        out = ws.recv()
                        if isinstance(out, str):
                            message = json.loads(out)
                            if message['type'] == 'executing':
                                data = message['data']
                                if data['node'] is None and data['prompt_id'] == prompt_id:
                                    break # Execution is complete
                except Exception as e:
                    self.error.emit(f"WebSocket error: {e}")
                    break
                finally:
                    ws.close()

                if not self.is_running: break

                history = get_history(prompt_id).get(prompt_id)
                image_data = None
                if history and history.get('outputs'):
                    for node_id in history['outputs']:
                        node_output = history['outputs'][node_id]
                        if 'images' in node_output:
                            image_info = node_output['images'][0]
                            image_data = get_image(image_info['filename'], image_info['subfolder'], image_info['type'])
                            if image_data: break
                
                row_images.append({'lora': lora_model, 'image_data': image_data or b''})
            
            if row_images:
                images_grid.append(row_images)

        if self.is_running:
            self.finished.emit(images_grid)


class SaveGridWorker(QObject):
    """Worker thread for saving the final image grid to prevent UI freezing."""
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, images_grid, model_names, filename):
        super().__init__()
        self.images_grid = images_grid
        self.model_names = model_names
        self.filename = filename
        self.cell_size = (256, 256)
        self.font_size = 14
        self.margin = 32
        self.header_extra = 24

    def run(self):
        try:
            if not self.images_grid or not self.images_grid[0]:
                self.error.emit("No images were generated to save.")
                return

            rows = len(self.images_grid)
            cols = len(self.model_names)
            cell_w, cell_h = self.cell_size
            font = QFont("Arial", self.font_size, QFont.Bold)

            # Use a dummy image to get font metrics for calculating text wrapping
            dummy_img = QImage(1, 1, QImage.Format_RGB32)
            painter = QPainter(dummy_img)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            painter.end()

            # Calculate header height dynamically based on model name wrapping
            max_lines = 0
            for model_name in self.model_names:
                words = model_name.split()
                lines = 1
                current_line = ""
                for word in words:
                    test_line = current_line + (" " if current_line else "") + word
                    if metrics.horizontalAdvance(test_line) > cell_w - 8:
                        lines += 1
                        current_line = word
                    else:
                        current_line = test_line
                max_lines = max(max_lines, lines)

            header_h = max_lines * metrics.height() + self.header_extra + self.margin

            width = cols * (cell_w + self.margin) + self.margin
            height = header_h + rows * (cell_h + self.margin)

            grid_image = QImage(width, height, QImage.Format_RGB32)
            grid_image.fill(QColor("white"))
            
            painter = QPainter(grid_image)
            painter.setFont(font)

            # Draw model names as column headers
            for col, model_name in enumerate(self.model_names):
                x = self.margin + col * (cell_w + self.margin)
                text_rect = QRect(x, self.margin, cell_w, header_h - self.margin)
                draw_text_multiline_boxed(painter, text_rect, model_name, font)

            # Draw the generated images into the grid
            for row, image_row in enumerate(self.images_grid):
                for col, img_info in enumerate(image_row):
                    x = self.margin + col * (cell_w + self.margin)
                    y = header_h + row * (cell_h + self.margin)
                    
                    if img_info['image_data']:
                        qimg = QImage()
                        qimg.loadFromData(img_info['image_data'])
                        pixmap = QPixmap.fromImage(qimg).scaled(cell_w, cell_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        img_x = x + (cell_w - pixmap.width()) // 2
                        img_y = y + (cell_h - pixmap.height()) // 2
                        painter.drawPixmap(img_x, img_y, pixmap)
                    else:
                        # Draw a placeholder for missing images
                        painter.setPen(Qt.gray)
                        painter.drawRect(x, y, cell_w, cell_h)
                        painter.drawText(QRect(x, y, cell_w, cell_h), Qt.AlignCenter, "Image\nFailed")

            painter.end()

            if grid_image.save(self.filename):
                self.finished.emit(os.path.abspath(self.filename))
            else:
                self.error.emit("Failed to save the final grid image file.")
        except Exception as e:
            self.error.emit(f"An error occurred while creating the grid image: {e}")


class GridMakerUI(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoRA Model Comparison Grid Maker for ComfyUI")
        self.setGeometry(100, 100, 900, 700)
        
        self.worker_thread = None
        self.generation_worker = None
        self.save_thread = None
        self.save_worker = None
        
        self.prompts = DEFAULT_PROMPTS.copy()
        self.lora_models = LORA_MODELS.copy()

        # --- Main Widget and Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- UI Elements ---
        main_layout.addWidget(QLabel("Prompts (one per line):"))
        self.prompts_edit = QTextEdit()
        self.prompts_edit.setPlainText("\n".join(self.prompts))
        main_layout.addWidget(self.prompts_edit)

        main_layout.addWidget(QLabel("LoRA Models (full path, one per line):"))
        self.models_edit = QTextEdit()
        self.models_edit.setPlainText("\n".join(self.lora_models))
        main_layout.addWidget(self.models_edit)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready. Load your workflow and models, then start.")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        self.start_btn = QPushButton("Start Generation")
        self.start_btn.clicked.connect(self.start_generation)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-weight: bold; border-radius: 5px; } QPushButton:disabled { background-color: #aaa; }")
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-weight: bold; border-radius: 5px; } QPushButton:disabled { background-color: #aaa; }")
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        main_layout.addLayout(button_layout)
        
        self.check_workflow()

    def check_workflow(self):
        self.workflow_dict = load_workflow_from_file(WORKFLOW_PATH)
        if not self.workflow_dict:
            self.status_label.setText(f"ERROR: Workflow file not found at '{WORKFLOW_PATH}'. Please ensure it exists.")
            self.start_btn.setEnabled(False)
        if not self.lora_models:
             self.status_label.setText("Warning: No LoRA models listed. Please add paths to your models.")

    def start_generation(self):
        self.workflow_dict = load_workflow_from_file(WORKFLOW_PATH)
        if not self.workflow_dict:
            self.status_label.setText(f"ERROR: Cannot start. Workflow file not found at '{WORKFLOW_PATH}'.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Starting generation...")

        self.prompts = [p.strip() for p in self.prompts_edit.toPlainText().splitlines() if p.strip()]
        self.lora_models = [m.strip() for m in self.models_edit.toPlainText().splitlines() if m.strip()]

        if not self.prompts or not self.lora_models:
            self.status_label.setText("Error: Both prompts and LoRA models must be provided.")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return

        self.progress_bar.setValue(0)

        self.worker_thread = QThread()
        self.generation_worker = GenerationWorker(self.prompts, self.lora_models, self.workflow_dict)
        self.generation_worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.generation_worker.run)
        self.generation_worker.progress.connect(self.on_progress)
        self.generation_worker.finished.connect(self.on_generation_finished)
        self.generation_worker.error.connect(self.on_generation_error)
        
        # Cleanup connections
        self.generation_worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.generation_worker.finished.connect(self.generation_worker.deleteLater)
        
        self.worker_thread.start()

    def stop_generation(self):
        self.status_label.setText("Stopping...")
        if self.generation_worker:
            self.generation_worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generation stopped by user.")

    def on_progress(self, count, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(count)
        self.status_label.setText(f"In Progress ({count}/{total}):\n{msg}")

    def on_generation_finished(self, images_grid):
        self.status_label.setText("Generation complete. Saving the final grid image...")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        self.save_thread = QThread()
        self.save_worker = SaveGridWorker(
            images_grid,
            [os.path.basename(m) for m in self.lora_models],
            OUTPUT_GRID_PATH
        )
        self.save_worker.moveToThread(self.save_thread)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.finished.connect(self.on_grid_saved)
        self.save_worker.error.connect(self.on_generation_error) # Reuse error handler
        
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_worker.finished.connect(self.save_worker.deleteLater)
        
        self.save_thread.start()

    def on_grid_saved(self, path):
        self.status_label.setText(f"Successfully saved grid to:\n{path}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_generation_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
        if self.save_thread and self.save_thread.isRunning():
            self.save_thread.quit()

    def closeEvent(self, event):
        """Ensure threads are stopped when closing the window."""
        self.stop_generation()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = GridMakerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
