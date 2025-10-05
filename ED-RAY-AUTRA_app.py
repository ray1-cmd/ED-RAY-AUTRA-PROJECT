# ED-RAY AUTRA PROJECT
# Exoplanet Detection and Analysis System
#RAY AUTRA TEAM - NASA SPACE APP CHALLENGE 2025
import sys
import json
import os
import warnings
import requests
import math
import random
import webbrowser
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

# Suppress all warnings for clean interface
import sys
import logging
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
logging.getLogger().setLevel(logging.ERROR)
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.svg=false'

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QFileDialog,
    QMessageBox, QInputDialog, QProgressBar, QProgressDialog, QTabWidget, QScrollArea, QGroupBox,
    QCheckBox, QSlider, QSplitter, QStackedWidget, QListWidget,
    QListWidgetItem, QFrame, QGridLayout, QMenuBar, QMenu, QAction,
    QStatusBar, QToolBar, QSizePolicy, QSplashScreen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QEventLoop, QPoint
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPalette, QColor, QPainter, QBrush, QPen, QPolygon

# Disable qt_material to avoid SVG errors
QT_MATERIAL_AVAILABLE = False

# Imports ML
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Warning: Machine Learning dependencies not available: {e}")
    print("Install with: pip install torch scikit-learn joblib")

# Visualisation
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import plotly.graph_objects as go
    PLOT_AVAILABLE = True
except ImportError as e:
    PLOT_AVAILABLE = False
    print(f"Warning: Visualization dependencies not available: {e}")
    print("Install with: pip install matplotlib plotly")

# Text-to-Speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"Warning: Text-to-Speech not available: {e}")
    print("Install with: pip install pyttsx3")

# ==================== CONFIGURATION ====================
HF_REPO = "ED-RAY-AUTRA-PROJECT/EXOfind-1"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Use absolute paths for better reliability
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "model"
LANGUAGES_DIR = BASE_DIR / "languages"
COURSES_DIR = BASE_DIR / "courses"
PREFERENCES_FILE = BASE_DIR / "user_preferences.json"

# Create directories if they don't exist
for dir_path in [MODEL_DIR, LANGUAGES_DIR, COURSES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Default preferences
DEFAULT_PREFERENCES = {
    'language': 'English',
    'accessibility': {
        'high_contrast': False,
        'text_to_speech': False,
        'font_size': 12,
        'screen_reader': False,
        'keyboard_nav': True
    },
    'theme': 'dark',
    'course_font_size': 12,
    'tts_voice_language': 'en'
}

# ==================== PREFERENCES MANAGEMENT ====================

def load_preferences():
    """Load user preferences from file"""
    try:
        if PREFERENCES_FILE.exists():
            with open(PREFERENCES_FILE, 'r', encoding='utf-8') as f:
                prefs = json.load(f)
                # Merge with defaults to ensure all keys exist
                for key in DEFAULT_PREFERENCES:
                    if key not in prefs:
                        prefs[key] = DEFAULT_PREFERENCES[key]
                return prefs
        return DEFAULT_PREFERENCES.copy()
    except (IOError, json.JSONDecodeError, PermissionError) as e:
        print(f"Warning: Error loading preferences: {e}")
        return DEFAULT_PREFERENCES.copy()
    except Exception as e:
        print(f"Unexpected error loading preferences: {e}")
        return DEFAULT_PREFERENCES.copy()

def save_preferences(preferences):
    """Save user preferences to file"""
    try:
        # Ensure parent directory exists
        PREFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2)
        return True
    except (IOError, PermissionError, OSError) as e:
        print(f"Warning: Cannot save preferences: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving preferences: {e}")
        return False

# ==================== ML MODEL ====================

class MLP(nn.Module):
    """PyTorch MLP Model"""
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# ==================== THREADS ====================

class ModelLoaderThread(QThread):
    """Thread to load model in background"""
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)
    
    def run(self):
        try:
            config_path = MODEL_DIR / "config.json"
            model_path = MODEL_DIR / "model.pth"
            preprocessor_path = MODEL_DIR / "preprocessor.joblib"
            
            if not all([config_path.exists(), model_path.exists(), preprocessor_path.exists()]):
                self.error.emit("Model files missing")
                return
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            preprocessor = joblib.load(preprocessor_path)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MLP(
                input_dim=config['input_dim'],
                hidden_layers=config['hidden_layers'],
                output_dim=config['output_dim']
            ).to(device)
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            self.finished.emit(model, preprocessor, config)
        except Exception as e:
            self.error.emit(str(e))

class PredictionThread(QThread):
    """Thread for predictions"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, model, preprocessor, config, input_data):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.input_data = input_data
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'model') and self.model:
            # Move model to CPU to free GPU memory
            self.model.cpu()
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Process data efficiently
            X_processed = self.preprocessor.transform(self.input_data)
            if hasattr(X_processed, "toarray"):
                X_processed = X_processed.toarray()
            
            # Convert to tensor efficiently
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = self.model(X_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
            
            # Convert to Python types for signal emission
            result = {
                'class': int(predicted_class.cpu().numpy()[0]),
                'probabilities': probabilities.cpu().numpy()[0].tolist(),
                'class_names': self.config.get('target_mapping', {})
            }
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ==================== CUSTOM WIDGETS ====================

class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib Canvas for Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        if not PLOT_AVAILABLE:
            raise ImportError("Matplotlib is not available. Install with: pip install matplotlib")
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

class StatsWidget(QWidget):
    """Widget to display statistics"""
    def __init__(self):
        if not PLOT_AVAILABLE:
            raise ImportError("Matplotlib is not available. Install with: pip install matplotlib")
        super().__init__()
        self.init_ui()
        self.stats = {
            'total_predictions': 0,
            'predictions_history': [],
            'class_counts': {'CANDIDATE': 0, 'CONFIRMED': 0, 'FALSE POSITIVE': 0},
            'confidence_history': []
        }
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(" Real-Time Statistics")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Metrics
        metrics_layout = QGridLayout()
        
        self.total_label = QLabel("Predictions: 0")
        self.accuracy_label = QLabel("Accuracy: N/A")
        self.confidence_label = QLabel("Avg Confidence: 0%")
        
        metrics_layout.addWidget(self.total_label, 0, 0)
        metrics_layout.addWidget(self.accuracy_label, 0, 1)
        metrics_layout.addWidget(self.confidence_label, 0, 2)
        
        layout.addLayout(metrics_layout)
        
        # Chart (optional)
        if PLOT_AVAILABLE:
            self.canvas = MatplotlibCanvas(self, width=8, height=3)
            layout.addWidget(self.canvas)
        else:
            layout.addWidget(QLabel("Charts unavailable (matplotlib not installed)"))
        
        self.setLayout(layout)
    
    def update_stats(self, prediction_result):
        """Update statistics"""
        self.stats['total_predictions'] += 1
        
        class_names_inv = {v: k for k, v in prediction_result['class_names'].items()}
        predicted_class = class_names_inv.get(prediction_result['class'], 'Unknown')
        
        if predicted_class in self.stats['class_counts']:
            self.stats['class_counts'][predicted_class] += 1
        
        confidence = float(prediction_result['probabilities'][prediction_result['class']])
        self.stats['confidence_history'].append(confidence)
        
        # Update labels
        self.total_label.setText(f"Predictions: {self.stats['total_predictions']}")
        avg_conf = np.mean(self.stats['confidence_history']) * 100
        self.confidence_label.setText(f"Avg Confidence: {avg_conf:.1f}%")
        
        # Update chart
        self.update_chart()
    
    def update_chart(self):
        """Update chart"""
        if not hasattr(self, 'canvas'):
            return
        self.canvas.axes.clear()
        
        if self.stats['confidence_history']:
            self.canvas.axes.plot(self.stats['confidence_history'], 'o-', color='#00CC96')
            self.canvas.axes.set_xlabel('Prediction #')
            self.canvas.axes.set_ylabel('Confidence')
            self.canvas.axes.set_ylim([0, 1])
            self.canvas.axes.grid(True, alpha=0.3)
        
        self.canvas.draw()

# ==================== PLANET TEXTURE UTILS ====================

class PlanetTextureUtils:
    @staticmethod
    def generate_2d_texture(features):
        size = 256
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Planet characteristics
        radius = features.get('koi_prad', 1.0)
        temp = features.get('koi_teq', 300)
        
        # Base color
        if temp < 200:
            base_color = (50, 100, 255)  # Cold blue
        elif temp < 400:
            base_color = (0, 200, 100)   # Temperate green
        elif temp < 600:
            base_color = (230, 180, 50)  # Hot orange
        else:
            base_color = (200, 50, 50)   # Very hot red
        
        # Draw the planet
        center = (size // 2, size // 2)
        planet_radius = int(size * 0.4 * min(radius / 2, 1.5))
        
        # Base circle
        draw.ellipse([(center[0]-planet_radius, center[1]-planet_radius),
                     (center[0]+planet_radius, center[1]+planet_radius)],
                    fill=base_color, outline=None)
        
        # Add details (clouds, continents, etc.)
        PlanetTextureUtils.add_planet_features(draw, center, planet_radius, features)
        
        # Blur slightly for a more natural effect
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        return img
    
    @staticmethod
    def add_planet_features(draw, center, radius, features):
        temp = features.get('koi_teq', 300)
        
        # Clouds for cold planets
        if temp < 400:
            for _ in range(10):
                x = center[0] + random.randint(-radius//2, radius//2)
                y = center[1] + random.randint(-radius//2, radius//2)
                if (x-center[0])**2 + (y-center[1])**2 < (radius*0.8)**2:
                    cloud_r = random.randint(5, radius//3)
                    draw.ellipse([(x-cloud_r, y-cloud_r), (x+cloud_r, y+cloud_r)],
                               fill=(255, 255, 255, 128), outline=None)
        
        # Surface details for hot planets
        else:
            for _ in range(15):
                x = center[0] + random.randint(-radius, radius)
                y = center[1] + random.randint(-radius, radius)
                if (x-center[0])**2 + (y-center[1])**2 < radius**2:
                    r = random.randint(2, radius//4)
                    color = (random.randint(50, 200), random.randint(50, 100), 0, 200)
                    draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=color, outline=None)

# ==================== HOME PAGE ====================

class HomePage(QWidget):
    """Home page"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("ED-RAY AUTRA")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Exoplanet Detection and Analysis System")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        # Model Information - Simplified
        info_group = QGroupBox("Model Status")
        info_layout = QGridLayout()
        
        config_path = MODEL_DIR / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Row 1: Author and Model Type
            author_label = QLabel(f"Author: {config.get('author', 'N/A')}")
            model_type_label = QLabel(f"Model Type: {config.get('model_type', 'N/A')}")
            
            info_layout.addWidget(author_label, 0, 0)
            info_layout.addWidget(model_type_label, 0, 1)
            
            # Row 2: Accuracy and Batch Size
            metrics = config.get('final_metrics', {})
            accuracy_label = QLabel(f"Accuracy: {metrics.get('test_accuracy', 0)*100:.2f}%")
            hyperparams = config.get('hyperparameters', {})
            batch_label = QLabel(f"Batch Size: {hyperparams.get('batch_size', 'N/A')}")
            
            info_layout.addWidget(accuracy_label, 1, 0)
            info_layout.addWidget(batch_label, 1, 1)
            
            # Row 3: Classes
            target_mapping = config.get('target_mapping', {})
            classes_list = ", ".join(target_mapping.keys())
            classes_label = QLabel(f"Classes: {classes_list}")
            
            info_layout.addWidget(classes_label, 2, 0, 1, 2)
        else:
            no_model_label = QLabel("No model found")
            info_layout.addWidget(no_model_label, 0, 0)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Statistics
        self.stats_widget = StatsWidget()
        layout.addWidget(self.stats_widget)
        
        # Quick guide
        guide_group = QGroupBox(" Quick Start")
        guide_layout = QVBoxLayout()
        guide_layout.addWidget(QLabel("1. Load your observation data"))
        guide_layout.addWidget(QLabel("2. Run a prediction"))
        guide_layout.addWidget(QLabel("3. Visualize results in 3D"))
        guide_group.setLayout(guide_layout)
        layout.addWidget(guide_group)
        
        layout.addStretch()
        self.setLayout(layout)

# ==================== PREDICTION PAGE ====================

class PredictionPage(QWidget):
    """Prediction page"""
    prediction_made = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.preprocessor = None
        self.config = None
        self.configdata = None
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(" Exoplanet Prediction")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Input method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Manual Input", "Import CSV"])
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        layout.addLayout(method_layout)
        
        # Stack for different methods
        self.input_stack = QStackedWidget()
        
        # Manual input page
        manual_page = QWidget()
        manual_layout = QVBoxLayout()
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.manual_form_layout = QGridLayout()
        scroll_content.setLayout(self.manual_form_layout)
        scroll.setWidget(scroll_content)
        manual_layout.addWidget(scroll)
        
        manual_page.setLayout(manual_layout)
        self.input_stack.addWidget(manual_page)
        
        # CSV page
        csv_page = QWidget()
        csv_layout = QVBoxLayout()
        
        csv_btn_layout = QHBoxLayout()
        self.load_csv_btn = QPushButton("Load CSV")
        self.load_csv_btn.clicked.connect(self.load_csv)
        csv_btn_layout.addWidget(self.load_csv_btn)
        csv_btn_layout.addStretch()
        csv_layout.addLayout(csv_btn_layout)
        
        self.csv_table = QTableWidget()
        csv_layout.addWidget(self.csv_table)
        
        csv_page.setLayout(csv_layout)
        self.input_stack.addWidget(csv_page)
        
        layout.addWidget(self.input_stack)
        
        # Prediction button
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.clicked.connect(self.make_prediction)
        self.predict_btn.setMinimumHeight(40)
        layout.addWidget(self.predict_btn)
        
        # Results
        self.result_group = QGroupBox("Results")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("No prediction made")
        self.result_label.setFont(QFont("Arial", 12))
        result_layout.addWidget(self.result_label)
        
        if PLOT_AVAILABLE:
            self.prob_canvas = MatplotlibCanvas(self, width=6, height=3)
            result_layout.addWidget(self.prob_canvas)
        else:
            self.prob_canvas = None
            result_layout.addWidget(QLabel("Charts unavailable (matplotlib not installed)"))
        
        self.result_group.setLayout(result_layout)
        self.result_group.setVisible(False)
        layout.addWidget(self.result_group)
        
        self.setLayout(layout)
    
    def load_model(self):
        """Load model"""
        # Guard when ML stack is unavailable
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "Error", 
                "PyTorch and ML dependencies are not available. Install requirements to use prediction features.\n\n"
                "Install with: pip install torch scikit-learn joblib")
            return

        self.loader_thread = ModelLoaderThread()
        self.loader_thread.finished.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_error)
        self.loader_thread.start()
        
        # Load configdata
        configdata_path = MODEL_DIR / "configdata.json"
        if configdata_path.exists():
            with open(configdata_path, 'r') as f:
                self.configdata = json.load(f)
                self.create_manual_form()
    
    def on_model_loaded(self, model, preprocessor, config):
        """Callback when model is loaded"""
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        QMessageBox.information(self, "Success", "Model loaded successfully!")
    
    def on_model_error(self, error):
        """Error callback"""
        QMessageBox.warning(self, "Error", f"Model loading error: {error}")
    
    def create_manual_form(self):
        """Create manual input form"""
        if not self.configdata:
            return
        
        # Clean existing layout
        while self.manual_form_layout.count():
            item = self.manual_form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.input_fields = {}
        expected_cols = self.configdata['expected_columns']
        numeric_cols = self.configdata['numeric_columns']
        categorical_cols = self.configdata['categorical_columns']
        schema = self.configdata.get('schema_summary', {})
        
        row = 0
        col = 0
        for col_name in expected_cols:
            label = QLabel(col_name + ":")
            self.manual_form_layout.addWidget(label, row, col * 2)
            
            if col_name in numeric_cols:
                field = QDoubleSpinBox()
                field.setRange(-1e9, 1e9)
                field.setDecimals(6)
                field.setValue(0.0)
            elif col_name in categorical_cols:
                field = QComboBox()
                sample_values = schema.get(col_name, {}).get('sample_values', [])
                if sample_values:
                    field.addItems([str(v) for v in sample_values])
                else:
                    field.setEditable(True)
            else:
                field = QLineEdit()
            
            self.manual_form_layout.addWidget(field, row, col * 2 + 1)
            self.input_fields[col_name] = field
            
            col += 1
            if col >= 2:
                col = 0
                row += 1
    
    def on_method_changed(self, index):
        """Change input method"""
        self.input_stack.setCurrentIndex(index)
    
    def load_csv(self):
        """Load CSV file"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                df = pd.read_csv(filename)
                self.csv_table.setRowCount(len(df))
                self.csv_table.setColumnCount(len(df.columns))
                self.csv_table.setHorizontalHeaderLabels(df.columns.tolist())
                
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        item = QTableWidgetItem(str(df.iloc[i, j]))
                        self.csv_table.setItem(i, j, item)
                
                self.csv_data = df
                if hasattr(self, 'configdata') and self.configdata:
                    expected = set(self.configdata['expected_columns'])
                    actual = set(df.columns)
                    if not expected.issubset(actual):
                        QMessageBox.warning(self, "Column Mismatch", f"Missing columns: {expected - actual}")
                QMessageBox.information(self, "Success", f"CSV loaded: {len(df)} rows")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Loading error: {e}")
    
    def make_prediction(self):
        """Launch prediction"""
        if not self.model:
            QMessageBox.warning(self, "Error", "Model not loaded")
            return
        
        # Get data
        if self.method_combo.currentIndex() == 0:  # Manuel
            input_dict = {}
            for col_name, field in self.input_fields.items():
                if isinstance(field, QDoubleSpinBox):
                    input_dict[col_name] = field.value()
                elif isinstance(field, QComboBox):
                    input_dict[col_name] = field.currentText()
                else:
                    input_dict[col_name] = field.text()
            input_data = pd.DataFrame([input_dict])
        else:  # CSV
            if not hasattr(self, 'csv_data'):
                QMessageBox.warning(self, "Error", "No CSV loaded")
                return
            input_data = self.csv_data
        
        # Launch prediction in thread
        self.prediction_thread = PredictionThread(
            self.model, self.preprocessor, self.config, input_data
        )
        self.prediction_thread.finished.connect(self.on_prediction_finished)
        self.prediction_thread.error.connect(self.on_prediction_error)
        self.prediction_thread.start()
        
        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("⏳ Prediction in progress...")
    
    def on_prediction_finished(self, result):
        """Callback when prediction is finished"""
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Run Prediction")
        
        # Display results
        class_names_inv = {v: k for k, v in result['class_names'].items()}
        predicted_class = class_names_inv.get(result['class'], 'Unknown')
        confidence = result['probabilities'][result['class']] * 100
        
        self.result_label.setText(
            f"<b>Predicted Class:</b> {predicted_class}<br>"
            f"<b>Confidence:</b> {confidence:.2f}%"
        )
        
        # Probabilities chart
        if self.prob_canvas:
            self.prob_canvas.axes.clear()
            classes = [class_names_inv.get(i, f'Class {i}') for i in range(len(result['probabilities']))]
            self.prob_canvas.axes.bar(classes, result['probabilities'], color='#667eea')
            self.prob_canvas.axes.set_ylabel('Probability')
            self.prob_canvas.axes.set_title('Class Probabilities')
            self.prob_canvas.draw()
        
        self.result_group.setVisible(True)
        
        # Emit signal
        self.prediction_made.emit(result)
        
        QMessageBox.information(self, "Success", f"Prediction: {predicted_class} ({confidence:.1f}%)")
    
    def on_prediction_error(self, error):
        """Error callback"""
        self.predict_btn.setEnabled(True)
        self.predict_btn.setText("Run Prediction")
        QMessageBox.warning(self, "Error", f"Prediction error: {error}")

# ==================== DATA PAGE ====================

class DatasetPage(QWidget):
    """Dataset management page"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.configdata = None
        self.dataset_rows = []
        self.init_ui()
        self.load_configdata()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(" Dataset Management")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Tabs
        tabs = QTabWidget()
        
        # Create tab
        create_tab = QWidget()
        create_layout = QVBoxLayout()
        
        # Column info
        info_btn = QPushButton("ℹ View Expected Columns")
        info_btn.clicked.connect(self.show_column_info)
        create_layout.addWidget(info_btn)
        
        # Form
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_widget = QWidget()
        self.form_layout = QGridLayout()
        form_widget.setLayout(self.form_layout)
        form_scroll.setWidget(form_widget)
        create_layout.addWidget(form_scroll)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("Remove Last Row")
        remove_btn.clicked.connect(self.remove_last_row)
        btn_layout.addWidget(remove_btn)
        
        create_layout.addLayout(btn_layout)
        
        # Table preview
        self.preview_table = QTableWidget()
        create_layout.addWidget(QLabel("Dataset Preview:"))
        create_layout.addWidget(self.preview_table)
        
        # Save buttons
        save_layout = QHBoxLayout()
        save_csv_btn = QPushButton("Save CSV")
        save_csv_btn.clicked.connect(lambda: self.save_dataset('csv'))
        save_layout.addWidget(save_csv_btn)
        
        save_json_btn = QPushButton("Save JSON")
        save_json_btn.clicked.connect(lambda: self.save_dataset('json'))
        save_layout.addWidget(save_json_btn)
        
        create_layout.addLayout(save_layout)
        create_tab.setLayout(create_layout)
        tabs.addTab(create_tab, "Create Dataset")
        
        # Load tab
        load_tab = QWidget()
        load_layout = QVBoxLayout()
        
        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_dataset)
        load_layout.addWidget(load_btn)
        
        self.loaded_table = QTableWidget()
        load_layout.addWidget(self.loaded_table)
        
        load_tab.setLayout(load_layout)
        tabs.addTab(load_tab, "Load Dataset")
        
        layout.addWidget(tabs)
        self.setLayout(layout)
    
    def load_configdata(self):
        """Load configdata.json"""
        configdata_path = MODEL_DIR / "configdata.json"
        if configdata_path.exists():
            with open(configdata_path, 'r') as f:
                self.configdata = json.load(f)
                self.create_form()
    
    def create_form(self):
        """Create input form"""
        if not self.configdata:
            return
        
        self.input_fields = {}
        expected_cols = self.configdata['expected_columns'] + [self.configdata['target_column']]
        numeric_cols = self.configdata['numeric_columns']
        categorical_cols = self.configdata['categorical_columns']
        schema = self.configdata.get('schema_summary', {})
        
        row = 0
        col = 0
        for col_name in expected_cols:
            label = QLabel(col_name + ":")
            self.form_layout.addWidget(label, row, col * 2)
            
            if col_name in numeric_cols:
                field = QDoubleSpinBox()
                field.setRange(-1e9, 1e9)
                field.setDecimals(6)
                field.setValue(0.0)
            elif col_name == self.configdata['target_column']:
                field = QComboBox()
                # Add all possible labels including CONFIRMED
                field.addItems(["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"])
            elif col_name in categorical_cols:
                field = QComboBox()
                sample_values = schema.get(col_name, {}).get('sample_values', [])
                if sample_values:
                    field.addItems([str(v) for v in sample_values])
                field.setEditable(True)
            else:
                field = QLineEdit()
            
            self.form_layout.addWidget(field, row, col * 2 + 1)
            self.input_fields[col_name] = field
            
            col += 1
            if col >= 2:
                col = 0
                row += 1
    
    def show_column_info(self):
        """Display column information"""
        if not self.configdata:
            QMessageBox.warning(self, "Error", "Configdata not loaded")
            return

        info = f"Numeric Columns ({len(self.configdata['numeric_columns'])}):\n"
        info += ", ".join(self.configdata['numeric_columns'][:10]) + "...\n\n"
        info += f"Categorical Columns ({len(self.configdata['categorical_columns'])}):\n"
        info += ", ".join(self.configdata['categorical_columns']) + "\n\n"
        info += f"Target Column: {self.configdata['target_column']}\n"
        info += f"Classes: {', '.join(self.configdata['label_mapping'].keys())}"

        QMessageBox.information(self, "Expected Columns", info)


    
    def add_row(self):
        """Add row to dataset"""
        if not self.input_fields:
            return
        
        row_data = {}
        for col_name, field in self.input_fields.items():
            if isinstance(field, QDoubleSpinBox):
                row_data[col_name] = field.value()
            elif isinstance(field, QComboBox):
                row_data[col_name] = field.currentText()
            else:
                row_data[col_name] = field.text()
        
        self.dataset_rows.append(row_data)
        self.update_preview()
        QMessageBox.information(self, "Success", f"Row added! Total: {len(self.dataset_rows)}")
    
    def remove_last_row(self):
        """Remove last row"""
        if self.dataset_rows:
            self.dataset_rows.pop()
            self.update_preview()
    
    def update_preview(self):
        """Update preview"""
        if not self.dataset_rows:
            self.preview_table.setRowCount(0)
            return
        
        df = pd.DataFrame(self.dataset_rows)
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                self.preview_table.setItem(i, j, item)
    
    def save_dataset(self, format_type):
        """Save dataset"""
        if not self.dataset_rows:
            QMessageBox.warning(self, "Error", "No data to save")
            return
        
        df = pd.DataFrame(self.dataset_rows)
        
        if format_type == 'csv':
            filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if filename:
                df.to_csv(filename, index=False)
                QMessageBox.information(self, "Success", f"Dataset saved: {filename}")
        else:
            filename, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
            if filename:
                df.to_json(filename, orient='records', indent=2)
                QMessageBox.information(self, "Success", f"Dataset saved: {filename}")
    
    def load_dataset(self):
        """Load dataset"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "Data Files (*.csv *.json)")
        if filename:
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filename)
                else:
                    df = pd.read_json(filename)
                
                self.loaded_table.setRowCount(len(df))
                self.loaded_table.setColumnCount(len(df.columns))
                self.loaded_table.setHorizontalHeaderLabels(df.columns.tolist())
                
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        item = QTableWidgetItem(str(df.iloc[i, j]))
                        self.loaded_table.setItem(i, j, item)
                
                QMessageBox.information(self, "Success", f"Dataset loaded: {len(df)} rows")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Loading error: {e}")

# ==================== TRAINING THREAD ====================

class TrainingThread(QThread):
    """Thread for model training"""
    progress = pyqtSignal(int, float, float)  # epoch, loss, accuracy
    finished_signal = pyqtSignal(object, object, object)  # model, preprocessor, config
    error = pyqtSignal(str)
    
    def __init__(self, data, configdata, epochs, batch_size, lr, patience, test_size):
        super().__init__()
        self.data = data
        self.configdata = configdata
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.test_size = test_size
        self.paused = False
    
    def run(self):
        try:
            # Prepare data
            target_col = self.configdata['target_column']
            X = self.data.drop(columns=[target_col])
            y = self.data[target_col]
            
            # Encode labels
            label_mapping = self.configdata['label_mapping']
            y = y.map(label_mapping)
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42, stratify=y
            )
            
            # Create preprocessor
            numeric_cols = self.configdata['numeric_columns']
            categorical_cols = self.configdata['categorical_columns']
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            # Fit and transform
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            if hasattr(X_train_processed, "toarray"):
                X_train_processed = X_train_processed.toarray()
                X_test_processed = X_test_processed.toarray()
            
            # Create model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_dim = X_train_processed.shape[1]
            output_dim = len(label_mapping)
            
            model = MLP(
                input_dim=input_dim,
                hidden_layers=[256, 128],
                output_dim=output_dim,
                dropout=0.2
            ).to(device)
            
            # Prepare tensors
            X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Optimizer and loss
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            # Training
            best_loss = float('inf')
            best_epoch = 0
            patience_counter = 0
            
            for epoch in range(self.epochs):
                # Check if paused
                while self.paused:
                    QThread.msleep(100)  # Sleep for 100ms while paused
                
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                avg_loss = total_loss / len(train_loader)
                accuracy = correct / total
                
                # Emit progress signal
                self.progress.emit(epoch + 1, avg_loss, accuracy)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break
            
            # Final test
            model.eval()
            with torch.no_grad():
                X_test_device = X_test_tensor.to(device)
                y_test_device = y_test_tensor.to(device)
                outputs = model(X_test_device)
                _, predicted = torch.max(outputs.data, 1)
                test_accuracy = (predicted == y_test_device).sum().item() / y_test_device.size(0)
            
            # Create config in the correct format
            config = {
                'author': 'RAY AUTRA TEAM',
                'created_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                'model_type': 'PyTorch_MLP_tabular',
                'model_file': 'model.pth',
                'checkpoint_file': 'checkpoint.pt',
                'preprocessor_file': 'preprocessor.joblib',
                'input_dim': input_dim,
                'hidden_layers': [256, 128],
                'output_dim': output_dim,
                'num_classes': output_dim,
                'hyperparameters': {
                    'batch_size': self.batch_size,
                    'epochs_requested': self.epochs,
                    'learning_rate': self.lr,
                    'patience': self.patience
                },
                'data_split': {
                    'train_size': len(train_loader.dataset),
                    'val_size': 0,
                    'test_size': len(X_test)
                },
                'target_mapping': {v: k for k, v in label_mapping.items()},
                'final_metrics': {
                    'best_epoch': best_epoch,
                    'test_loss': round(best_loss, 4),
                    'test_accuracy': round(test_accuracy, 4)
                },
                'notes': 'Preprocessor is a ColumnTransformer (num imputer+scaler, cat imputer+onehot). High-cardinality cat columns were dropped automatically.'
            }
            
            self.finished_signal.emit(model, preprocessor, config)
            
        except Exception as e:
            self.error.emit(str(e))

# ==================== TRAINING PAGE ====================

class TrainingPage(QWidget):
    """Model training page"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_data = None
        self.configdata = None
        self.training_thread = None
        self.loss_history = []
        self.acc_history = []
        self.init_ui()
        self.load_configdata()
    
    def load_configdata(self):
        """Load configdata on initialization"""
        try:
            configdata_path = MODEL_DIR / "configdata.json"
            if configdata_path.exists():
                with open(configdata_path, 'r', encoding='utf-8') as f:
                    self.configdata = json.load(f)
        except Exception as e:
            print(f"Error loading configdata: {e}")
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🧠 Model Training")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Dataset
        dataset_group = QGroupBox("Training Dataset")
        dataset_layout = QVBoxLayout()
        
        dataset_btn_layout = QHBoxLayout()
        self.load_dataset_btn = QPushButton("Load Dataset")
        self.load_dataset_btn.clicked.connect(self.load_training_dataset)
        dataset_btn_layout.addWidget(self.load_dataset_btn)
        dataset_btn_layout.addStretch()
        dataset_layout.addLayout(dataset_btn_layout)
        
        self.dataset_info_label = QLabel("No dataset loaded")
        dataset_layout.addWidget(self.dataset_info_label)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Hyperparameters
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout()
        
        hyper_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(40)
        hyper_layout.addWidget(self.epochs_spin, 0, 1)
        
        hyper_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(8, 512)
        self.batch_spin.setValue(128)
        hyper_layout.addWidget(self.batch_spin, 0, 3)
        
        hyper_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setValue(0.001)
        hyper_layout.addWidget(self.lr_spin, 1, 1)
        
        hyper_layout.addWidget(QLabel("Patience:"), 1, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(8)
        hyper_layout.addWidget(self.patience_spin, 1, 3)
        
        hyper_layout.addWidget(QLabel("Test Size (%):"), 2, 0)
        self.test_slider = QSlider(Qt.Horizontal)
        self.test_slider.setRange(5, 30)
        self.test_slider.setValue(10)
        self.test_label = QLabel("10%")
        self.test_slider.valueChanged.connect(lambda v: self.test_label.setText(f"{v}%"))
        hyper_layout.addWidget(self.test_slider, 2, 1)
        hyper_layout.addWidget(self.test_label, 2, 2)
        
        hyper_group.setLayout(hyper_layout)
        layout.addWidget(hyper_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton(" Start Training")
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton(" Stop")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.clicked.connect(self.pause_training)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.export_chart_btn = QPushButton("Export Chart")
        self.export_chart_btn.clicked.connect(self.export_training_chart)
        self.export_chart_btn.setEnabled(False)
        btn_layout.addWidget(self.export_chart_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Charts
        if PLOT_AVAILABLE:
            try:
                self.training_canvas = MatplotlibCanvas(self, width=8, height=4)
                layout.addWidget(self.training_canvas)
            except Exception as e:
                print(f"Error creating training canvas: {e}")
                self.training_canvas = None
                layout.addWidget(QLabel("Charts unavailable (matplotlib not installed)"))
        else:
            self.training_canvas = None
            layout.addWidget(QLabel("Charts unavailable (matplotlib not installed)"))
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_training_dataset(self):
        """Load dataset for training"""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "CSV Files (*.csv)")
        if filename:
            try:
                df = pd.read_csv(filename)
                
                # Load configdata if not already loaded
                if not hasattr(self, 'configdata'):
                    configdata_path = MODEL_DIR / "configdata.json"
                    if not configdata_path.exists():
                        QMessageBox.warning(self, "Error", "Configdata not found. Please ensure model files are present.")
                        return
                    with open(configdata_path, 'r', encoding='utf-8') as f:
                        self.configdata = json.load(f)
                
                # Validate required columns
                target = self.configdata['target_column']
                expected_cols = set(self.configdata['expected_columns'])
                actual_cols = set(df.columns)
                
                # Check for target column
                if target not in df.columns:
                    QMessageBox.warning(self, "Error", 
                        f"Target column '{target}' not found in dataset.\n\n"
                        f"Required columns: {', '.join(sorted(expected_cols))}\n"
                        f"Plus target column: {target}")
                    return
                
                # Check for missing expected columns
                missing_cols = expected_cols - actual_cols
                if missing_cols:
                    reply = QMessageBox.question(
                        self,
                        "Missing Columns",
                        f"The following expected columns are missing:\n{', '.join(sorted(missing_cols))}\n\n"
                        f"Do you want to continue anyway? (Missing columns will be filled with default values)",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                    
                    # Add missing columns with default values
                    for col in missing_cols:
                        if col in self.configdata['numeric_columns']:
                            df[col] = 0.0
                        else:
                            df[col] = "UNKNOWN"
                
                # Validate target column values
                valid_labels = set(self.configdata['label_mapping'].keys())
                invalid_labels = set(df[target].unique()) - valid_labels
                if invalid_labels:
                    QMessageBox.warning(self, "Invalid Labels",
                        f"Invalid labels found in target column:\n{', '.join(invalid_labels)}\n\n"
                        f"Valid labels are: {', '.join(valid_labels)}")
                    return
                
                self.training_data = df
                self.dataset_info_label.setText(
                    f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns\n"
                    f"Target distribution: {df[target].value_counts().to_dict()}"
                )
                QMessageBox.information(self, "Success", 
                    f"Dataset loaded successfully!\n\n"
                    f"File: {filename}\n"
                    f"Rows: {len(df)}\n"
                    f"Columns: {len(df.columns)}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Loading error: {e}")
    
    def start_training(self):
        """Start training"""
        if not hasattr(self, 'training_data') or self.training_data is None:
            QMessageBox.warning(
                self, 
                "No Dataset Loaded", 
                "Please load a training dataset before starting the training process.\n\n"
                "You can load a dataset by clicking the 'Load Dataset' button and selecting a CSV file "
                "that contains the required features and target column."
            )
            return
        
        if not TORCH_AVAILABLE:
            QMessageBox.warning(self, "Error", 
                "PyTorch is not available. Install with: pip install torch scikit-learn joblib")
            return
        
        # Check that dataset has target column
        configdata_path = MODEL_DIR / "configdata.json"
        if not configdata_path.exists():
            QMessageBox.warning(self, "Error", "configdata.json not found")
            return
        
        with open(configdata_path, 'r') as f:
            configdata = json.load(f)
        
        target_col = configdata['target_column']
        if target_col not in self.training_data.columns:
            QMessageBox.warning(self, "Error", f"Target column '{target_col}' missing in dataset")
            return
        
        # Launch training in thread
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.status_label.setText("Training in progress...")
        
        # Create training thread
        self.training_thread = TrainingThread(
            self.training_data,
            configdata,
            self.epochs_spin.value(),
            self.batch_spin.value(),
            self.lr_spin.value(),
            self.patience_spin.value(),
            self.test_slider.value() / 100.0
        )
        self.training_thread.progress.connect(self.on_training_progress)
        self.training_thread.finished_signal.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        self.training_thread.start()
    
    def on_training_progress(self, epoch, loss, accuracy):
        """Progress callback"""
        self.progress_bar.setValue(epoch)
        self.status_label.setText(f"Epoch {epoch}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        # Update chart
        self.loss_history.append(loss)
        self.acc_history.append(accuracy)
        
        if self.training_canvas and PLOT_AVAILABLE:
            try:
                self.training_canvas.axes.clear()
                self.training_canvas.axes.plot(self.loss_history, label='Loss', color='red')
                self.training_canvas.axes.plot(self.acc_history, label='Accuracy', color='green')
                self.training_canvas.axes.legend()
                self.training_canvas.axes.set_xlabel('Epoch')
                self.training_canvas.axes.grid(True, alpha=0.3)
                self.training_canvas.draw()
            except Exception as e:
                print(f"Error updating training chart: {e}")
    
    def on_training_finished(self, model, preprocessor, config):
        """Training finished callback"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.export_chart_btn.setEnabled(True)
        self.status_label.setText("Training completed!")
        self.progress_bar.setValue(100)

        # Save model
        reply = QMessageBox.question(
            self,
            "Save",
            "Training completed! Do you want to save the model?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Ask for author name
            from PyQt5.QtWidgets import QInputDialog
            author_name, ok = QInputDialog.getText(
                self,
                "Author Name",
                "Enter your name (author of this retraining):"
            )

            if not ok or not author_name:
                author_name = "Anonymous"

            try:
                # Save model and preprocessor
                torch.save(model.state_dict(), MODEL_DIR / "model.pth")
                joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

                # Update config with new information
                config['last_retrained_at_utc'] = datetime.utcnow().isoformat()
                config['last_retrained_by'] = author_name

                # Add retraining history
                if 'retraining_history' not in config:
                    config['retraining_history'] = []

                config['retraining_history'].append({
                    'date': datetime.utcnow().isoformat(),
                    'author': author_name,
                    'test_accuracy': config['final_metrics']['test_accuracy'],
                    'epochs': self.epochs_spin.value(),
                    'learning_rate': self.lr_spin.value()
                })

                # Save updated config
                with open(MODEL_DIR / "config.json", 'w') as f:
                    json.dump(config, f, indent=2)

                QMessageBox.information(
                    self,
                    "Success",
                    f"Model saved successfully!\n\n"
                    f"Author: {author_name}\n"
                    f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                )

                # Ask if user wants to share the model on Hugging Face
                share_model_reply = QMessageBox.question(
                    self,
                    "Share Model",
                    "Would you like to share your trained model with the community on Hugging Face?\n\n"
                    "This helps improve the ED-RAY AUTRA project!",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if share_model_reply == QMessageBox.Yes:
                    try:
                        webbrowser.open("https://huggingface.co/ED-RAY-AUTRA-PROJECT")
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Error opening Hugging Face page: {e}")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Save error: {e}")
    
    def on_training_error(self, error):
        """Error callback"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.status_label.setText("Training error")
        QMessageBox.warning(self, "Error", f"Error: {error}")
    def stop_training(self):
        """Stop training"""
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.status_label.setText("Training stopped by user")
            QMessageBox.information(self, "Stopped", "Training has been stopped")
    
    def pause_training(self):
        """Pause/Resume training"""
        if hasattr(self, 'training_thread') and self.training_thread.isRunning():
            if hasattr(self.training_thread, 'paused'):
                if self.training_thread.paused:
                    self.training_thread.paused = False
                    self.pause_btn.setText("⏸ Pause")
                    self.status_label.setText("Training resumed...")
                else:
                    self.training_thread.paused = True
                    self.pause_btn.setText("▶ Resume")
                    self.status_label.setText("Training paused")
    
    def export_training_chart(self):
        """Export training chart as image"""
        if not self.training_canvas or not PLOT_AVAILABLE:
            QMessageBox.warning(self, "Error", "No chart available to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Training Chart", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filename:
            try:
                self.training_canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Chart saved to:\n{filename}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save chart: {e}")

# ==================== COURSES PAGE ====================

class CoursesPage(QWidget):
    """Courses page with TTS and font size control"""
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.courses = []
        self.tts_engine = None
        self.is_reading = False
        self.course_font_size = 12
        self.init_ui()
        self.load_courses()
        self.init_tts()
    
    def get_text(self, key, default=""):
        """Get translated text"""
        if self.main_window and hasattr(self.main_window, 'get_text'):
            return self.main_window.get_text(key, default)
        return default or key
    
    def init_tts(self):
        """Initialize TTS engine"""
        if not TTS_AVAILABLE:
            self.tts_engine = None
            return
        
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            self.voices_dict = {}
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                if 'english' in voice_name or 'en' in voice_id or 'us' in voice_id or 'uk' in voice_id:
                    self.voices_dict['en'] = voice.id
                elif 'french' in voice_name or 'fr' in voice_id:
                    self.voices_dict['fr'] = voice.id
            
            # Set default properties for better quality
            if self.tts_engine:
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 0.9)  # Volume
        except Exception as e:
            print(f"TTS initialization error: {e}")
            self.tts_engine = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Courses & Learning")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Font size control
        controls_layout.addWidget(QLabel("Font Size:"))
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(8, 32)
        self.font_size_slider.setValue(12)
        self.font_size_slider.valueChanged.connect(self.change_font_size)
        controls_layout.addWidget(self.font_size_slider)
        
        self.font_size_label = QLabel("12pt")
        controls_layout.addWidget(self.font_size_label)
        
        # TTS controls
        self.tts_lang_combo = QComboBox()
        self.tts_lang_combo.addItems(["English Voice", "French Voice"])
        controls_layout.addWidget(self.tts_lang_combo)
        
        self.tts_btn = QPushButton("Start Reading")
        self.tts_btn.clicked.connect(self.toggle_tts)
        controls_layout.addWidget(self.tts_btn)
        
        self.stop_tts_btn = QPushButton("Stop Reading")
        self.stop_tts_btn.clicked.connect(self.stop_tts)
        self.stop_tts_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_tts_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Main content: Horizontal layout with course list on left and viewer on right
        main_content = QHBoxLayout()
        
        # Left side: Course list with buttons
        left_panel = QVBoxLayout()
        
        # Course list
        self.course_list = QListWidget()
        self.course_list.itemDoubleClicked.connect(self.open_course)
        self.course_list.setMaximumWidth(300)
        left_panel.addWidget(QLabel("Available Courses:"))
        left_panel.addWidget(self.course_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_courses)
        btn_layout.addWidget(refresh_btn)
        
        community_btn = QPushButton(self.get_text("courses_community", "Hugging Face Community"))
        community_btn.clicked.connect(self.open_community)
        btn_layout.addWidget(community_btn)
        
        left_panel.addLayout(btn_layout)
        
        # Right side: Course viewer
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Course Content:"))
        self.course_viewer = QTextEdit()
        self.course_viewer.setReadOnly(True)
        right_panel.addWidget(self.course_viewer)
        
        # Add panels to main content
        main_content.addLayout(left_panel, 1)
        main_content.addLayout(right_panel, 3)
        
        layout.addLayout(main_content)
        
        self.setLayout(layout)
    
    def change_font_size(self, size):
        """Change course font size"""
        self.course_font_size = size
        self.font_size_label.setText(f"{size}pt")
        font = QFont("Arial", size)
        self.course_viewer.setFont(font)
    
    def toggle_tts(self):
        """Toggle text-to-speech"""
        if not TTS_AVAILABLE:
            QMessageBox.warning(
                self,
                "TTS Not Available",
                "Text-to-Speech is not available.\n\nInstall pyttsx3:\npip install pyttsx3\n\nThen restart the application."
            )
            return
        
        if not self.tts_engine:
            QMessageBox.warning(
                self,
                "TTS Error",
                "Text-to-Speech engine failed to initialize.\nPlease check your system's TTS support."
            )
            return
        
        if self.is_reading:
            self.stop_tts()
        else:
            self.start_tts()
    
    def start_tts(self):
        """Start reading course content"""
        content = self.course_viewer.toPlainText()
        if not content:
            QMessageBox.warning(self, "No Content", "Please select a course first")
            return
        
        # Reinitialize TTS engine if needed
        if not self.tts_engine:
            if not TTS_AVAILABLE:
                QMessageBox.warning(self, "TTS Not Available", "Text-to-Speech is not available. Install pyttsx3.")
                return
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
            except Exception as e:
                QMessageBox.warning(self, "TTS Error", f"Cannot initialize TTS: {e}")
                return
        
        # Set voice based on selection
        voice_lang = 'en' if self.tts_lang_combo.currentIndex() == 0 else 'fr'
        if hasattr(self, 'voices_dict') and voice_lang in self.voices_dict:
            try:
                self.tts_engine.setProperty('voice', self.voices_dict[voice_lang])
            except:
                pass
        
        self.is_reading = True
        self.tts_btn.setEnabled(False)
        self.stop_tts_btn.setEnabled(True)
        self.tts_lang_combo.setEnabled(False)
        
        # Read in separate thread to avoid blocking
        def read_text():
            try:
                self.tts_engine.say(content)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
            finally:
                # Use QTimer to update UI from main thread
                QTimer.singleShot(100, self.reset_tts_buttons)
        
        self.tts_thread = threading.Thread(target=read_text, daemon=True)
        self.tts_thread.start()
    
    def reset_tts_buttons(self):
        """Reset TTS buttons after reading"""
        self.is_reading = False
        self.tts_btn.setEnabled(True)
        self.stop_tts_btn.setEnabled(False)
        self.tts_lang_combo.setEnabled(True)
    
    def stop_tts(self):
        """Stop reading"""
        self.is_reading = False
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                print(f"Error stopping TTS: {e}")
        
        # Reset buttons immediately
        self.reset_tts_buttons()
    
    def load_courses(self):
        """Load courses from courses/ folder"""
        self.course_list.clear()
        self.courses = []
        
        if COURSES_DIR.exists():
            for course_file in COURSES_DIR.glob("*.md"):
                try:
                    with open(course_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        metadata = {
                            'title': course_file.stem,
                            'level': 'Beginner',
                            'author': 'Unknown',
                            'content': content,
                            'filename': course_file.name
                        }
                        
                        # Parse frontmatter
                        if content.startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                frontmatter = parts[1]
                                for line in frontmatter.split('\n'):
                                    if ':' in line:
                                        key, value = line.split(':', 1)
                                        key = key.strip().lower()
                                        value = value.strip()
                                        if key in metadata:
                                            metadata[key] = value
                                metadata['content'] = parts[2]
                        
                        self.courses.append(metadata)
                        
                        item = QListWidgetItem(f" {metadata['title']} ({metadata['level']})")
                        self.course_list.addItem(item)
                except Exception as e:
                    print(f"Error loading course {course_file.name}: {e}")
        
        if not self.courses:
            self.course_viewer.setPlainText("No courses available.\nAdd .md files to the 'courses/' folder.")
    
    def open_course(self, item):
        """Open course"""
        index = self.course_list.row(item)
        if 0 <= index < len(self.courses):
            course = self.courses[index]
            self.course_viewer.setPlainText(course['content'])
    
    def open_community(self):
        """Open community link"""
        webbrowser.open("https://huggingface.co/ED-RAY-AUTRA-PROJECT")

# ==================== QUIZ PAGE ====================

class QuizPage(QWidget):
    """Quiz page for exoplanet knowledge"""
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.questions = []
        self.current_question = 0
        self.score = 0
        self.answers = []
        self.init_ui()
        self.load_questions()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("❓ Exoplanet Quiz")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Score display
        self.score_label = QLabel("Score: 0/0")
        self.score_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.score_label)
        
        # Question display
        question_group = QGroupBox("Question")
        question_layout = QVBoxLayout()
        
        self.question_label = QLabel("Loading questions...")
        self.question_label.setWordWrap(True)
        self.question_label.setFont(QFont("Arial", 12))
        question_layout.addWidget(self.question_label)
        
        question_group.setLayout(question_layout)
        layout.addWidget(question_group)
        
        # Options
        self.options_group = QGroupBox("Options")
        self.options_layout = QVBoxLayout()
        self.option_buttons = []
        
        for i in range(4):
            btn = QPushButton()
            btn.setMinimumHeight(50)
            btn.clicked.connect(lambda checked, idx=i: self.check_answer(idx))
            self.option_buttons.append(btn)
            self.options_layout.addWidget(btn)
        
        self.options_group.setLayout(self.options_layout)
        layout.addWidget(self.options_group)
        
        # Explanation
        self.explanation_label = QLabel("")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("background-color: #e8f4f8; padding: 10px; border-radius: 5px;")
        self.explanation_label.setVisible(False)
        layout.addWidget(self.explanation_label)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.next_btn = QPushButton("Next Question ➡")
        self.next_btn.clicked.connect(self.next_question)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        
        self.restart_btn = QPushButton("🔄 Restart Quiz")
        self.restart_btn.clicked.connect(self.restart_quiz)
        nav_layout.addWidget(self.restart_btn)
        
        layout.addLayout(nav_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def load_questions(self):
        """Load questions from JSON file"""
        try:
            quiz_file = BASE_DIR / "quiz_questions.json"
            if quiz_file.exists():
                with open(quiz_file, 'r', encoding='utf-8') as f:
                    self.questions = json.load(f)
                self.display_question()
            else:
                self.question_label.setText("Quiz questions file not found!")
        except Exception as e:
            self.question_label.setText(f"Error loading quiz: {e}")
    
    def display_question(self):
        """Display current question"""
        if self.current_question < len(self.questions):
            q = self.questions[self.current_question]
            self.question_label.setText(f"Q{self.current_question + 1}: {q['question']}")
            
            for i, option in enumerate(q['options']):
                self.option_buttons[i].setText(option)
                self.option_buttons[i].setEnabled(True)
                self.option_buttons[i].setStyleSheet("")
            
            self.explanation_label.setVisible(False)
            self.next_btn.setEnabled(False)
            self.score_label.setText(f"Score: {self.score}/{len(self.questions)}")
        else:
            self.show_results()
    
    def check_answer(self, selected):
        """Check if answer is correct"""
        if self.current_question >= len(self.questions):
            return
        
        q = self.questions[self.current_question]
        correct = q['correct']
        
        # Disable all buttons
        for btn in self.option_buttons:
            btn.setEnabled(False)
        
        # Show correct/incorrect
        if selected == correct:
            self.option_buttons[selected].setStyleSheet("background-color: #4CAF50; color: white;")
            self.score += 1
        else:
            self.option_buttons[selected].setStyleSheet("background-color: #f44336; color: white;")
            self.option_buttons[correct].setStyleSheet("background-color: #4CAF50; color: white;")
        
        # Show explanation
        self.explanation_label.setText(f"✓ {q.get('explanation', 'Correct answer: ' + q['options'][correct])}")
        self.explanation_label.setVisible(True)
        
        self.next_btn.setEnabled(True)
        self.score_label.setText(f"Score: {self.score}/{len(self.questions)}")
    
    def next_question(self):
        """Move to next question"""
        self.current_question += 1
        self.display_question()
    
    def restart_quiz(self):
        """Restart the quiz"""
        self.current_question = 0
        self.score = 0
        self.answers = []
        self.display_question()
    
    def show_results(self):
        """Show final results"""
        percentage = (self.score / len(self.questions)) * 100
        self.question_label.setText(
            f"🎉 Quiz Completed!\n\n"
            f"Your Score: {self.score}/{len(self.questions)} ({percentage:.1f}%)\n\n"
            f"{'Excellent!' if percentage >= 80 else 'Good job!' if percentage >= 60 else 'Keep learning!'}"
        )
        
        for btn in self.option_buttons:
            btn.setVisible(False)
        
        self.explanation_label.setVisible(False)
        self.next_btn.setVisible(False)

# ==================== SIMULATION PAGE ====================

class SimulationPage(QWidget):
    """Orbital simulation and game page"""
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.simulation_running = False
        self.angle = 0
        self.speed = 1.0
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🌌 Orbital Simulation & Game")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title)
        
        # Tabs for Simulation and Game
        tabs = QTabWidget()
        
        # Simulation Tab
        sim_tab = QWidget()
        sim_layout = QVBoxLayout()
        
        # Controls
        controls_group = QGroupBox("Simulation Controls")
        controls_layout = QGridLayout()
        
        # Star parameters
        controls_layout.addWidget(QLabel("Star Mass (M☉):"), 0, 0)
        self.star_mass = QDoubleSpinBox()
        self.star_mass.setRange(0.1, 10.0)
        self.star_mass.setValue(1.0)
        self.star_mass.setSuffix(" M☉")
        controls_layout.addWidget(self.star_mass, 0, 1)
        
        controls_layout.addWidget(QLabel("Star Radius (R☉):"), 0, 2)
        self.star_radius = QDoubleSpinBox()
        self.star_radius.setRange(0.1, 5.0)
        self.star_radius.setValue(1.0)
        self.star_radius.setSuffix(" R☉")
        controls_layout.addWidget(self.star_radius, 0, 3)
        
        # Planet parameters
        controls_layout.addWidget(QLabel("Planet Mass (M⊕):"), 1, 0)
        self.planet_mass = QDoubleSpinBox()
        self.planet_mass.setRange(0.1, 100.0)
        self.planet_mass.setValue(1.0)
        self.planet_mass.setSuffix(" M⊕")
        controls_layout.addWidget(self.planet_mass, 1, 1)
        
        controls_layout.addWidget(QLabel("Orbital Distance (AU):"), 1, 2)
        self.orbital_distance = QDoubleSpinBox()
        self.orbital_distance.setRange(0.1, 10.0)
        self.orbital_distance.setValue(1.0)
        self.orbital_distance.setSuffix(" AU")
        controls_layout.addWidget(self.orbital_distance, 1, 3)
        
        # Speed control
        controls_layout.addWidget(QLabel("Simulation Speed:"), 2, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_speed)
        controls_layout.addWidget(self.speed_slider, 2, 1, 1, 2)
        
        self.speed_label = QLabel("1.0x")
        controls_layout.addWidget(self.speed_label, 2, 3)
        
        controls_group.setLayout(controls_layout)
        sim_layout.addWidget(controls_group)
        
        # Simulation canvas
        if PLOT_AVAILABLE:
            self.sim_canvas = MatplotlibCanvas(self, width=8, height=6)
            sim_layout.addWidget(self.sim_canvas)
        else:
            sim_layout.addWidget(QLabel("Matplotlib required for simulation"))
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Simulation")
        self.start_btn.clicked.connect(self.toggle_simulation)
        btn_layout.addWidget(self.start_btn)
        
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(reset_btn)
        
        sim_layout.addLayout(btn_layout)
        sim_tab.setLayout(sim_layout)
        tabs.addTab(sim_tab, "🌍 Orbital Simulation")
        
        # Game Tab
        game_tab = QWidget()
        game_layout = QVBoxLayout()
        
        game_title = QLabel("🎮 Spaceship Landing Game")
        game_title.setFont(QFont("Arial", 14, QFont.Bold))
        game_layout.addWidget(game_title)
        
        # Game canvas
        self.game_canvas = QLabel()
        self.game_canvas.setMinimumSize(800, 600)
        self.game_canvas.setStyleSheet("background-color: #000033; border: 2px solid #666;")
        self.game_canvas.setAlignment(Qt.AlignCenter)
        game_layout.addWidget(self.game_canvas)
        
        # Game info
        game_info_layout = QHBoxLayout()
        self.game_score_label = QLabel("Score: 0")
        self.game_fuel_label = QLabel("Fuel: 100%")
        self.game_speed_label = QLabel("Speed: 0 m/s")
        
        game_info_layout.addWidget(self.game_score_label)
        game_info_layout.addWidget(self.game_fuel_label)
        game_info_layout.addWidget(self.game_speed_label)
        game_layout.addLayout(game_info_layout)
        
        # Game controls
        game_controls = QLabel("Controls: ⬆ Thrust | ⬅ Rotate Left | ➡ Rotate Right | SPACE Start/Restart")
        game_controls.setAlignment(Qt.AlignCenter)
        game_controls.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        game_layout.addWidget(game_controls)
        
        # Start game button
        self.start_game_btn = QPushButton("🚀 Start Game")
        self.start_game_btn.clicked.connect(self.start_game)
        game_layout.addWidget(self.start_game_btn)
        
        game_tab.setLayout(game_layout)
        tabs.addTab(game_tab, "🎮 Landing Game")
        
        layout.addWidget(tabs)
        self.setLayout(layout)
        
        # Simulation timer
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        
        # Game timer
        self.game_timer = QTimer()
        self.game_timer.timeout.connect(self.update_game)
        
        # Game state
        self.game_active = False
        self.ship_x = 400
        self.ship_y = 50
        self.ship_vx = 0
        self.ship_vy = 0
        self.ship_angle = 0
        self.ship_fuel = 100
        self.game_score = 0
        
        # Initial simulation draw
        if PLOT_AVAILABLE:
            self.draw_simulation()
    
    def update_speed(self, value):
        """Update simulation speed"""
        self.speed = value / 10.0
        self.speed_label.setText(f"{self.speed:.1f}x")
    
    def toggle_simulation(self):
        """Start/stop simulation"""
        if self.simulation_running:
            self.sim_timer.stop()
            self.start_btn.setText("▶ Start Simulation")
            self.simulation_running = False
        else:
            self.sim_timer.start(50)  # 20 FPS
            self.start_btn.setText("⏸ Pause Simulation")
            self.simulation_running = True
    
    def reset_simulation(self):
        """Reset simulation"""
        self.angle = 0
        self.sim_timer.stop()
        self.start_btn.setText("▶ Start Simulation")
        self.simulation_running = False
        if PLOT_AVAILABLE:
            self.draw_simulation()
    
    def update_simulation(self):
        """Update simulation frame"""
        self.angle += self.speed * 0.05
        if self.angle >= 360:
            self.angle -= 360
        if PLOT_AVAILABLE:
            self.draw_simulation()
    
    def draw_simulation(self):
        """Draw orbital simulation"""
        if not PLOT_AVAILABLE:
            return
        
        self.sim_canvas.axes.clear()
        
        # Draw star
        star_size = self.star_radius.value() * 100
        self.sim_canvas.axes.scatter([0], [0], s=star_size, c='yellow', marker='o', edgecolors='orange', linewidths=2)
        self.sim_canvas.axes.text(0, -0.3, '⭐ Star', ha='center', fontsize=10)
        
        # Calculate planet position
        distance = self.orbital_distance.value()
        angle_rad = math.radians(self.angle)
        planet_x = distance * math.cos(angle_rad)
        planet_y = distance * math.sin(angle_rad)
        
        # Draw orbit
        orbit_angles = np.linspace(0, 2*np.pi, 100)
        orbit_x = distance * np.cos(orbit_angles)
        orbit_y = distance * np.sin(orbit_angles)
        self.sim_canvas.axes.plot(orbit_x, orbit_y, 'b--', alpha=0.3, linewidth=1)
        
        # Draw planet
        planet_size = self.planet_mass.value() * 20
        self.sim_canvas.axes.scatter([planet_x], [planet_y], s=planet_size, c='blue', marker='o', edgecolors='darkblue', linewidths=2)
        self.sim_canvas.axes.text(planet_x, planet_y - 0.3, '🌍 Planet', ha='center', fontsize=9)
        
        # Set limits
        max_dist = distance * 1.5
        self.sim_canvas.axes.set_xlim(-max_dist, max_dist)
        self.sim_canvas.axes.set_ylim(-max_dist, max_dist)
        self.sim_canvas.axes.set_aspect('equal')
        self.sim_canvas.axes.grid(True, alpha=0.2)
        self.sim_canvas.axes.set_xlabel('Distance (AU)')
        self.sim_canvas.axes.set_ylabel('Distance (AU)')
        self.sim_canvas.axes.set_title(f'Orbital Simulation - Angle: {self.angle:.1f}°')
        
        self.sim_canvas.draw()
    
    def start_game(self):
        """Start the landing game"""
        self.game_active = True
        self.ship_x = 400
        self.ship_y = 50
        self.ship_vx = random.uniform(-2, 2)
        self.ship_vy = 0
        self.ship_angle = 0
        self.ship_fuel = 100
        self.game_score = 0
        self.game_timer.start(33)  # ~30 FPS
        self.start_game_btn.setEnabled(False)
        self.game_canvas.setFocus()
    
    def update_game(self):
        """Update game state"""
        if not self.game_active:
            return
        
        # Physics
        gravity = 0.1
        self.ship_vy += gravity
        
        # Update position
        self.ship_x += self.ship_vx
        self.ship_y += self.ship_vy
        
        # Check boundaries
        if self.ship_x < 0 or self.ship_x > 800:
            self.end_game("Out of bounds!")
            return
        
        # Check landing
        if self.ship_y >= 550:
            speed = math.sqrt(self.ship_vx**2 + self.ship_vy**2)
            
            # Check if landed in green zone (x between 350 and 450)
            in_green_zone = 350 <= self.ship_x <= 450
            
            if in_green_zone:
                # Landed in green zone - calculate score based on performance
                base_score = 500
                speed_penalty = speed * 50
                angle_penalty = abs(self.ship_angle) * 5
                fuel_bonus = self.ship_fuel * 2
                
                self.game_score = int(base_score - speed_penalty - angle_penalty + fuel_bonus)
                
                # Determine landing quality
                if speed < 3 and abs(self.ship_angle) < 15:
                    self.end_game(f"🎉 PERFECT LANDING!\n\nScore: {self.game_score}\n\nExcellent piloting skills!")
                elif speed < 5 and abs(self.ship_angle) < 30:
                    self.end_game(f"✅ Good Landing!\n\nScore: {self.game_score}\n\nNice job, but you can do better!")
                else:
                    self.end_game(f"⚠️ Rough Landing!\n\nScore: {self.game_score}\n\nYou made it, but work on your technique!")
            else:
                # Missed the landing zone
                self.end_game("❌ MISSED THE LANDING ZONE!\n\nTry to land on the green pad!")
            return
        
        # Update display
        self.draw_game()
        self.game_fuel_label.setText(f"Fuel: {int(self.ship_fuel)}%")
        self.game_speed_label.setText(f"Speed: {math.sqrt(self.ship_vx**2 + self.ship_vy**2):.1f} m/s")
    
    def draw_game(self):
        """Draw game frame"""
        pixmap = QPixmap(800, 600)
        pixmap.fill(QColor(0, 0, 51))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw stars
        painter.setPen(QPen(Qt.white, 1))
        for _ in range(50):
            x = random.randint(0, 800)
            y = random.randint(0, 500)
            painter.drawPoint(x, y)
        
        # Draw planet surface
        painter.setBrush(QBrush(QColor(100, 100, 100)))
        painter.drawRect(0, 550, 800, 50)
        
        # Draw landing pad
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        painter.drawRect(350, 545, 100, 5)
        
        # Draw spaceship
        painter.save()
        painter.translate(int(self.ship_x), int(self.ship_y))
        painter.rotate(self.ship_angle)
        
        # Ship body
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        ship_points = [
            (0, -15),
            (-10, 15),
            (10, 15)
        ]
        painter.drawPolygon(*[QPoint(x, y) for x, y in ship_points])
        
        painter.restore()
        painter.end()
        
        self.game_canvas.setPixmap(pixmap)
    
    def end_game(self, message):
        """End the game"""
        self.game_active = False
        self.game_timer.stop()
        self.start_game_btn.setEnabled(True)
        self.game_score_label.setText(f"Score: {self.game_score}")
        QMessageBox.information(self, "Game Over", message)
    
    def keyPressEvent(self, event):
        """Handle key presses for game"""
        if not self.game_active:
            return
        
        if event.key() == Qt.Key_Up and self.ship_fuel > 0:
            # Apply thrust
            thrust = 0.3
            angle_rad = math.radians(self.ship_angle - 90)
            self.ship_vx += thrust * math.cos(angle_rad)
            self.ship_vy += thrust * math.sin(angle_rad)
            self.ship_fuel -= 1
        elif event.key() == Qt.Key_Left:
            self.ship_angle -= 5
        elif event.key() == Qt.Key_Right:
            self.ship_angle += 5

# ==================== SETTINGS PAGE ====================

class SettingsPage(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.init_ui()
    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Title and Community Button
        header_layout = QHBoxLayout()
        title = QLabel("⚙️ Settings")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Join Community Button
        community_btn = QPushButton("🌐 JOIN FUTURE COMMUNITY")
        community_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        community_btn.clicked.connect(self.join_community)
        header_layout.addWidget(community_btn)
        
        main_layout.addLayout(header_layout)
        
        # Create scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Language
        lang_group = QGroupBox("Language")
        lang_layout = QVBoxLayout()
        
        self.lang_combo = QComboBox()
        # Load all available languages
        if hasattr(self.main_window, 'languages'):
            self.lang_combo.addItems(sorted(self.main_window.languages.keys()))
        self.lang_combo.setCurrentText(self.main_window.current_language)
        self.lang_combo.currentTextChanged.connect(self.change_language)
        lang_layout.addWidget(self.lang_combo)
        
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)
        
        # Accessibility
        access_group = QGroupBox("Accessibility")
        access_layout = QVBoxLayout()
        
        # High contrast
        hc_layout = QHBoxLayout()
        self.high_contrast_check = QCheckBox("High Contrast")
        self.high_contrast_check.setChecked(self.main_window.accessibility_settings['high_contrast'])
        self.high_contrast_check.stateChanged.connect(self.on_high_contrast_changed)
        hc_layout.addWidget(self.high_contrast_check)
        self.hc_status = QLabel("✓" if self.main_window.accessibility_settings['high_contrast'] else "")
        self.hc_status.setStyleSheet("color: green; font-weight: bold;")
        hc_layout.addWidget(self.hc_status)
        hc_layout.addStretch()
        access_layout.addLayout(hc_layout)
        
        # Text-to-Speech
        tts_layout = QHBoxLayout()
        self.tts_check = QCheckBox("Text-to-Speech")
        self.tts_check.setChecked(self.main_window.accessibility_settings['text_to_speech'])
        self.tts_check.stateChanged.connect(self.on_tts_changed)
        tts_layout.addWidget(self.tts_check)
        self.tts_status = QLabel("✓" if self.main_window.accessibility_settings['text_to_speech'] else "")
        self.tts_status.setStyleSheet("color: green; font-weight: bold;")
        tts_layout.addWidget(self.tts_status)
        tts_layout.addStretch()
        access_layout.addLayout(tts_layout)
        
        # Screen reader support
        sr_layout = QHBoxLayout()
        self.screen_reader_check = QCheckBox("Screen Reader Support")
        self.screen_reader_check.setChecked(self.main_window.accessibility_settings['screen_reader'])
        self.screen_reader_check.stateChanged.connect(self.on_screen_reader_changed)
        sr_layout.addWidget(self.screen_reader_check)
        self.sr_status = QLabel("✓" if self.main_window.accessibility_settings['screen_reader'] else "")
        self.sr_status.setStyleSheet("color: green; font-weight: bold;")
        sr_layout.addWidget(self.sr_status)
        sr_layout.addStretch()
        access_layout.addLayout(sr_layout)
        
        # Keyboard navigation
        kn_layout = QHBoxLayout()
        self.keyboard_nav_check = QCheckBox("Keyboard Navigation")
        self.keyboard_nav_check.setChecked(self.main_window.accessibility_settings['keyboard_nav'])
        self.keyboard_nav_check.stateChanged.connect(self.on_keyboard_nav_changed)
        kn_layout.addWidget(self.keyboard_nav_check)
        self.kn_status = QLabel("✓" if self.main_window.accessibility_settings['keyboard_nav'] else "")
        self.kn_status.setStyleSheet("color: green; font-weight: bold;")
        kn_layout.addWidget(self.kn_status)
        kn_layout.addStretch()
        access_layout.addLayout(kn_layout)
        
        # Font size
        access_layout.addWidget(QLabel("Font Size:"))
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setRange(8, 24)
        self.font_slider.setValue(self.main_window.accessibility_settings['font_size'])
        self.font_label = QLabel(f"{self.main_window.accessibility_settings['font_size']}pt")
        self.font_slider.valueChanged.connect(self.on_font_size_changed)
        access_layout.addWidget(self.font_slider)
        access_layout.addWidget(self.font_label)
        
        # Apply button
        apply_btn = QPushButton("Apply Accessibility Settings")
        apply_btn.clicked.connect(self.apply_accessibility)
        access_layout.addWidget(apply_btn)
        
        access_group.setLayout(access_layout)
        layout.addWidget(access_group)
        
        # Theme
        theme_group = QGroupBox("Theme")
        theme_layout = QVBoxLayout()
        
        theme_btn_layout = QHBoxLayout()
        dark_btn = QPushButton(" Dark Mode")
        dark_btn.clicked.connect(lambda: self.main_window.change_theme('dark'))
        theme_btn_layout.addWidget(dark_btn)
        
        light_btn = QPushButton(" Light Mode")
        light_btn.clicked.connect(lambda: self.main_window.change_theme('light'))
        theme_btn_layout.addWidget(light_btn)
        
        theme_layout.addLayout(theme_btn_layout)
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Model
        model_group = QGroupBox("Model Update")
        model_layout = QVBoxLayout()
        
        model_layout.addWidget(QLabel(f"Source: {HF_REPO}"))
        
        config_path = MODEL_DIR / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_layout.addWidget(QLabel(f"Version: {config.get('created_at_utc', 'N/A')}"))
        
        check_btn = QPushButton(" Check for Updates")
        check_btn.clicked.connect(self.check_updates)
        model_layout.addWidget(check_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        layout.addStretch()
        
        # Set content widget to scroll area
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # Bottom Community Button
        bottom_community_btn = QPushButton("🌐 JOIN FUTURE COMMUNITY")
        bottom_community_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        bottom_community_btn.clicked.connect(self.join_community)
        main_layout.addWidget(bottom_community_btn)
        
        self.setLayout(main_layout)
    
    def on_high_contrast_changed(self, state):
        """High contrast callback"""
        self.main_window.accessibility_settings['high_contrast'] = bool(state)
        self.hc_status.setText("✓" if bool(state) else "")
        # Announce change
        status = "enabled" if bool(state) else "disabled"
        self.main_window.speak(f"High contrast {status}")
        # Save immediately
        prefs = load_preferences()
        prefs['accessibility']['high_contrast'] = bool(state)
        save_preferences(prefs)
        # Apply immediately
        self.main_window.apply_accessibility_settings()
    
    def on_tts_changed(self, state):
        """Text-to-speech callback"""
        self.main_window.accessibility_settings['text_to_speech'] = bool(state)
        self.tts_status.setText("✓" if bool(state) else "")
        
        # Reinitialize or disable TTS engine
        if bool(state) and TTS_AVAILABLE:
            try:
                import pyttsx3
                self.main_window.tts_engine = pyttsx3.init()
                self.main_window.tts_engine.setProperty('rate', 150)
                self.main_window.tts_engine.setProperty('volume', 0.9)
                self.main_window.speak("Text to speech enabled")
            except:
                pass
        else:
            self.main_window.tts_engine = None
            
        # Save immediately
        prefs = load_preferences()
        prefs['accessibility']['text_to_speech'] = bool(state)
        save_preferences(prefs)
    
    def on_screen_reader_changed(self, state):
        """Screen reader callback"""
        self.main_window.accessibility_settings['screen_reader'] = bool(state)
        self.sr_status.setText("✓" if bool(state) else "")
        # Announce change
        status = "enabled" if bool(state) else "disabled"
        self.main_window.speak(f"Screen reader support {status}")
        # Save immediately
        prefs = load_preferences()
        prefs['accessibility']['screen_reader'] = bool(state)
        save_preferences(prefs)
    
    def on_keyboard_nav_changed(self, state):
        """Keyboard navigation callback"""
        self.main_window.accessibility_settings['keyboard_nav'] = bool(state)
        self.kn_status.setText("✓" if bool(state) else "")
        # Announce change
        status = "enabled" if bool(state) else "disabled"
        self.main_window.speak(f"Keyboard navigation {status}")
        # Save immediately
        prefs = load_preferences()
        prefs['accessibility']['keyboard_nav'] = bool(state)
        save_preferences(prefs)
    
    def on_font_size_changed(self, value):
        """Font size callback"""
        self.font_label.setText(f"{value}pt")
        self.main_window.accessibility_settings['font_size'] = value
        # Save immediately
        prefs = load_preferences()
        prefs['accessibility']['font_size'] = value
        save_preferences(prefs)
    
    def apply_accessibility(self):
        """Apply accessibility settings"""
        self.main_window.apply_accessibility_settings()
        QMessageBox.information(
            self,
            "Accessibility Settings",
            "Accessibility settings applied successfully!"
        )
    
    def change_language(self, language):
        """Change interface language"""
        if language == self.main_window.current_language:
            return  # No change needed
            
        self.main_window.current_language = language
        
        # Save preference
        prefs = load_preferences()
        prefs['language'] = language
        save_preferences(prefs)
        
        # Reload UI dynamically
        self.main_window.reload_ui_language()
        
        QMessageBox.information(
            self,
            "Language Changed",
            f"Language changed to: {language}\n\n"
            f"The interface has been updated!"
        )
    
    def join_community(self):
        """Open community page"""
        self.main_window.speak("Opening Hugging Face community page")
        try:
            webbrowser.open("https://huggingface.co/ED-RAY-AUTRA-PROJECT")
            QMessageBox.information(
                self,
                "Join Community",
                "Opening Hugging Face Community page...\n\n"
                "Join us to contribute and share your models!"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot open browser: {e}")
    
    def check_updates(self):
        """Check model updates"""
        try:
            # Check Internet connection
            response = requests.get("https://www.google.com", timeout=3)
            
            # Read local version
            local_config_path = MODEL_DIR / "config.json"
            if local_config_path.exists():
                with open(local_config_path, 'r') as f:
                    local_config = json.load(f)
                local_date = local_config.get('created_at_utc', '')
            else:
                local_date = ''
            
            # Download remote version
            remote_url = f"{HF_BASE_URL}/config.json"
            response = requests.get(remote_url, timeout=10)
            response.raise_for_status()
            remote_config = response.json()
            remote_date = remote_config.get('created_at_utc', '')

            # Parse ISO timestamps robustly
            def parse_dt(s):
                try:
                    if not s:
                        return None
                    # Support trailing 'Z'
                    if s.endswith('Z'):
                        s = s.replace('Z', '+00:00')
                    return datetime.fromisoformat(s)
                except Exception:
                    return None

            local_dt = parse_dt(local_date)
            remote_dt = parse_dt(remote_date)

            # Compare dates only if both parsed
            if remote_dt and (not local_dt or remote_dt > local_dt):
                reply = QMessageBox.question(
                    self,
                    "Update Available",
                    f"A new model version is available!\n\n"
                    f"Local version: {local_date[:10] if local_date else 'None'}\n"
                    f"Remote version: {remote_date[:10]}\n\n"
                    f"Do you want to download the update?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.download_model_update()
            else:
                QMessageBox.information(
                    self,
                    "No Update",
                    "You already have the latest model version!"
                )
        
        except requests.exceptions.ConnectionError:
            QMessageBox.warning(
                self,
                "Connection Error",
                "Cannot connect to Hugging Face.\nCheck your Internet connection."
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Verification error: {e}"
            )
    
    def download_model_update(self):
        """Download model update"""
        files_to_download = ['config.json', 'model.pth', 'preprocessor.joblib']
        
        progress = QProgressDialog("Downloading model...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()
        
        try:
            for i, filename in enumerate(files_to_download):
                url = f"{HF_BASE_URL}/{filename}"
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunks = []
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress.setValue(int((i + downloaded/total_size) / len(files_to_download) * 100))
                
                # Save file
                output_path = MODEL_DIR / filename
                with open(output_path, 'wb') as f:
                    f.write(b''.join(chunks))
            
            progress.setValue(100)
            
            QMessageBox.information(
                self,
                "Success",
                "Update downloaded successfully!\nRestart the application to load the new model."
            )
        
        except Exception as e:
            progress.close()
            QMessageBox.warning(
                self,
                "Error",
                f"Download error: {e}"
            )

# ==================== VISUALISATION 3D ====================

class Planet3DViewer(QWidget):
    """3D Visualization of an exoplanet"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls = QHBoxLayout()
        
        self.rotation_slider = QSlider(Qt.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.valueChanged.connect(lambda: self.update_plot())
        
        controls.addWidget(QLabel("Rotation:"))
        controls.addWidget(self.rotation_slider)
        
        # Canvas Matplotlib
        layout.addLayout(controls)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def generate_planet_cloud(self, features):
        """Generate 3D point cloud for planet"""
        # Extract relevant features
        radius = features.get('koi_prad', 1.0)  # Radius in Earth radii
        temp = features.get('koi_teq', 300)     # Equilibrium temperature
        
        # Generate points for a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Add noise based on temperature
        noise = np.random.normal(0, 0.01 * temp/300, x.shape)
        x += noise
        y += noise
        z += noise
        
        # Color based on temperature
        if temp < 200:
            color = (0.3, 0.5, 1.0)  # Cold blue
        elif temp < 400:
            color = (0.0, 0.8, 0.3)  # Temperate green
        elif temp < 600:
            color = (0.9, 0.7, 0.1)  # Hot orange
        else:
            color = (0.8, 0.2, 0.2)  # Very hot red
            
        return x, y, z, color
    

    
    def update_plot(self, features=None):
        """Update 3D visualization"""
        if features is None:
            features = {}
        
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        # Generate 3D point cloud
        x, y, z, color = self.generate_planet_cloud(features)
        
        # Rotation based on slider
        angle = np.radians(self.rotation_slider.value())
        rot_x = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
        
        # Apply rotation
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                vec = np.array([x[i,j], y[i,j], z[i,j]])
                vec_rot = np.dot(rot_x, vec)
                x[i,j], y[i,j], z[i,j] = vec_rot
        
        # Display surface
        ax.plot_surface(x, y, z, color=color, alpha=0.8, linewidth=0)
        
        # Axis configuration
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Exoplanet 3D View - Radius: {features.get("koi_prad", 1.0):.2f} R⊕')
        
        # Equal scale
        max_range = max(np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z))
        ax.set_xlim(-max_range/2, max_range/2)
        ax.set_ylim(-max_range/2, max_range/2)
        ax.set_zlim(-max_range/2, max_range/2)
        
        self.canvas.draw()

class Planet2DViewer(QWidget):
    """2D Visualization of an exoplanet"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label = QLabel()
        self.current_image = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
    
    def update_image(self, features):
        """Update 2D image"""
        img = self.generate_2d_texture(features)
        self.current_image = img  # Store for export
        
        # Convert PIL Image to QPixmap
        img.save('temp_planet.png')
        pixmap = QPixmap('temp_planet.png')
        self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))
        try:
            os.remove('temp_planet.png')
        except Exception:
            pass
    
    def generate_2d_texture(self, features):
        """Generate 2D texture for the planet"""
        return PlanetTextureUtils.generate_2d_texture(features)

class VisualizationPage(QWidget):
    """3D visualization page"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_features = {}
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Exoplanet Visualization")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Feature selection
        controls_layout.addWidget(QLabel("Radius (R⊕):"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 20.0)
        self.radius_spin.setValue(1.0)
        self.radius_spin.valueChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.radius_spin)
        
        controls_layout.addWidget(QLabel("Temperature (K):"))
        self.temp_spin = QSpinBox()
        self.temp_spin.setRange(100, 2000)
        self.temp_spin.setValue(300)
        self.temp_spin.valueChanged.connect(self.update_visualization)
        controls_layout.addWidget(self.temp_spin)
        
        # Random generation button
        random_btn = QPushButton("Random Planet")
        random_btn.clicked.connect(self.generate_random_planet)
        controls_layout.addWidget(random_btn)
        
        # Export button
        export_btn = QPushButton("Export Image")
        export_btn.clicked.connect(self.export_planet_image)
        controls_layout.addWidget(export_btn)
        
        layout.addLayout(controls_layout)
        
        # Visualizations
        viz_layout = QHBoxLayout()
        
        # 3D view
        self.viewer_3d = Planet3DViewer()
        viz_layout.addWidget(self.viewer_3d)
        
        # 2D view
        self.viewer_2d = Planet2DViewer()
        viz_layout.addWidget(self.viewer_2d)
        
        layout.addLayout(viz_layout, 1)
        
        # Information
        info_group = QGroupBox("Planet Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        self.setLayout(layout)
        self.update_visualization()
    
    def update_visualization(self):
        """Update visualizations"""
        features = {
            'koi_prad': self.radius_spin.value(),
            'koi_teq': self.temp_spin.value()
        }
        
        self.current_features = features
        
        # Update 3D view
        self.viewer_3d.update_plot(features)
        
        # Update 2D view
        self.viewer_2d.update_image(features)
        
        # Update information
        self.info_text.setPlainText(
            f"Radius: {features['koi_prad']:.2f} Earth radii\n"
            f"Temperature: {features['koi_teq']:.0f} K\n"
            f"Type: {self.get_planet_type(features)}"
        )
    
    def get_planet_type(self, features):
        """Determine planet type"""
        radius = features['koi_prad']
        temp = features['koi_teq']
        
        if radius < 0.8:
            return "Terrestrial"
        elif radius < 2.0:
            return "Super-Earth"
        elif radius < 4.0:
            return "Mini-Neptune"
        else:
            return "Gas Giant"
    
    def generate_random_planet(self):
        """Generate random planet"""
        self.radius_spin.setValue(random.uniform(0.5, 10.0))
        self.temp_spin.setValue(random.randint(100, 1500))
        self.update_visualization()
    
    def export_planet_image(self):
        """Export planet visualization as image"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Planet Image", "", 
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        
        if filename:
            try:
                # Get current planet image from 2D viewer
                if hasattr(self.viewer_2d, 'current_image') and self.viewer_2d.current_image:
                    self.viewer_2d.current_image.save(filename)
                    QMessageBox.information(self, "Success", f"Image saved to:\n{filename}")
                else:
                    QMessageBox.warning(self, "Error", "No planet image available")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save image: {e}")

# ==================== MAIN WINDOW ====================

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        # Load preferences
        prefs = load_preferences()
        self.current_language = prefs.get('language', 'English')
        self.accessibility_settings = prefs.get('accessibility', {
            'high_contrast': False,
            'text_to_speech': False,
            'font_size': 12,
            'screen_reader': False,
            'keyboard_nav': False
        })
        
        # Initialize TTS for accessibility
        self.tts_engine = None
        if self.accessibility_settings.get('text_to_speech', False) and TTS_AVAILABLE:
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
            except Exception as e:
                print(f"TTS initialization error: {e}")
        self.load_languages()
        self.check_model_on_startup()
        self.init_ui()
        
        # Apply light theme by default
        self.apply_light_theme()
        
        # Apply accessibility settings on startup
        if any(self.accessibility_settings.values()):
            QTimer.singleShot(500, self.apply_accessibility_settings)
    
    def init_ui(self):
        self.setWindowTitle("ED-RAY AUTRA - Exoplanet Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Menu bar
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('File')
        file_menu.addAction(self.create_action('Quit', self.close))
        
        view_menu = menubar.addMenu('View')
        view_menu.addAction(self.create_action('Light Mode', lambda: self.change_theme('light')))
        view_menu.addAction(self.create_action('Dark Mode', lambda: self.change_theme('dark')))
        
        help_menu = menubar.addMenu('Help')
        help_menu.addAction(self.create_action('About', self.show_about))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(200)
        sidebar_layout = QVBoxLayout()
        
        logo_label = QLabel("")
        logo_label.setFont(QFont("Arial", 48))
        logo_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        app_title = QLabel("ED-RAY\nAUTRA")
        app_title.setFont(QFont("Arial", 16, QFont.Bold))
        app_title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(app_title)
        
        sidebar_layout.addSpacing(20)
        
        # Navigation buttons
        self.nav_buttons = []
        pages = [
            ("Home", 0),
            ("Prediction", 1),
            ("Data", 2),
            ("Training", 3),
            ("Courses", 4),
            ("Quiz", 5),
            ("Simulation", 6),
            ("Visualization", 7),
            ("Settings", 8)
        ]
        
        for text, index in pages:
            btn = QPushButton(text)
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda checked, i=index, t=text: self.on_nav_button_clicked(i, t))
            # Accessibility
            btn.setAccessibleName(f"NavButton_{text}")
            btn.setToolTip(f"Go to {text} page")
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        sidebar_layout.addStretch()
        sidebar.setLayout(sidebar_layout)
        
        # Pages
        self.pages = QStackedWidget()
        
        # Home page
        self.home_page = HomePage()
        self.pages.addWidget(self.home_page)
        
        # Prediction page
        self.prediction_page = PredictionPage()
        self.prediction_page.prediction_made.connect(self.on_prediction_made)
        self.pages.addWidget(self.prediction_page)
        
        # Dataset page
        self.dataset_page = DatasetPage()
        self.pages.addWidget(self.dataset_page)
        
        # Training page
        self.training_page = TrainingPage()
        self.pages.addWidget(self.training_page)
        
        # Courses page
        self.courses_page = CoursesPage(main_window=self)
        self.pages.addWidget(self.courses_page)
        
        # Quiz page
        self.quiz_page = QuizPage(main_window=self)
        self.pages.addWidget(self.quiz_page)
        
        # Simulation page
        self.simulation_page = SimulationPage(main_window=self)
        self.pages.addWidget(self.simulation_page)
        
        # Visualization page
        self.visualization_page = VisualizationPage()
        self.pages.addWidget(self.visualization_page)
        
        # Settings Page
        self.settings_page = SettingsPage(self)
        self.pages.addWidget(self.settings_page)
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.pages, 1)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_action(self, text, slot):
        """Create menu action"""
        action = QAction(text, self)
        action.triggered.connect(slot)
        return action
    
    def speak(self, text):
        """Speak text if TTS is enabled"""
        if self.accessibility_settings.get('text_to_speech', False) and self.tts_engine:
            try:
                # Run in separate thread to avoid blocking
                def speak_thread():
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    except:
                        pass
                
                thread = threading.Thread(target=speak_thread, daemon=True)
                thread.start()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def on_nav_button_clicked(self, index, text):
        """Handle navigation button click with TTS"""
        self.speak(f"Button clicked: {text}")
        self.change_page(index)
    
    def change_page(self, index):
        """Change current page"""
        self.pages.setCurrentIndex(index)
        
        # Announce page change
        page_names = ["Home", "Prediction", "Data", "Training", "Courses", "Quiz", "Simulation", "Visualization", "Settings"]
        if 0 <= index < len(page_names):
            self.speak(f"Now on {page_names[index]} page")
        self.statusBar().showMessage(f"Page {index + 1}")
    
    def change_theme(self, theme):
        """Change theme"""
        if not QT_MATERIAL_AVAILABLE:
            print("qt-material disabled in the app code.")
            QMessageBox.warning(
                self, 
                "Theme Error", 
                "qt-material is disabled in code or it's not installed.. If qt-material is enabled in code: \n\nInstall with:\npip install qt-material\n\nThen restart the application."
            )
            return
        
        try:
            app = QApplication.instance()
            if theme == 'dark':
                apply_stylesheet(app, theme='dark_blue.xml')
            else:
                apply_stylesheet(app, theme='light_blue.xml')
            
            # Save preference
            prefs = load_preferences()
            prefs['theme'] = theme
            save_preferences(prefs)
            
            # Update status
            self.statusBar().showMessage(f"Theme changed to {theme} mode")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot change theme: {e}\n\nTry reinstalling qt-material.")
    
    def check_model_on_startup(self):
        """Check if model exists on startup"""
        config_path = MODEL_DIR / "config.json"
        model_path = MODEL_DIR / "model.pth"
        preprocessor_path = MODEL_DIR / "preprocessor.joblib"
        
        if not all([config_path.exists(), model_path.exists(), preprocessor_path.exists()]):
            reply = QMessageBox.question(
                None,
                "Model Not Found",
                "No model found in the 'model' folder.\n\n"
                "Would you like to download the model from Hugging Face?\n"
                "(This may take a few minutes)",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.download_initial_model()
    
    def download_initial_model(self):
        """Download initial model"""
        files_to_download = ['config.json', 'model.pth', 'preprocessor.joblib']
        
        # Modal progress dialog to prevent accidental closure
        progress = QProgressDialog("Downloading model...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()
        
        try:
            for i, filename in enumerate(files_to_download):
                url = f"{HF_BASE_URL}/{filename}"
                
                # Check if file already exists
                output_path = MODEL_DIR / filename
                if output_path.exists():
                    print(f"File {filename} already exists, skipping...")
                    continue
                    
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunks = []
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            overall_progress = int((i + downloaded/total_size) / len(files_to_download) * 100)
                            progress.setValue(overall_progress)
                            if progress.wasCanceled():
                                raise Exception("Download canceled by user")
                            QApplication.processEvents()
                
                # Ensure directory exists before saving
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(b''.join(chunks))
            
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(
                self,
                "Success",
                "Model downloaded successfully!"
            )
        
        except requests.exceptions.ConnectionError:
            progress.close()
            QMessageBox.critical(
                self,
                "Connection Error",
                "Cannot connect to Hugging Face.\n\n"
                "Please check your internet connection and try again."
            )
        except requests.exceptions.Timeout:
            progress.close()
            QMessageBox.critical(
                self,
                "Timeout Error",
                "Download timed out.\n\n"
                "Please check your internet connection and try again."
            )
        except requests.exceptions.HTTPError as e:
            progress.close()
            QMessageBox.critical(
                self,
                "HTTP Error",
                f"HTTP error {e.response.status_code}: {e.response.reason}\n\n"
                "Please check the repository URL or try again later."
            )
        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Download Error",
                f"Failed to download model: {e}\n\n"
                "Please download manually from:\n{HF_BASE_URL}"
            )
    
    def load_languages(self):
        """Load available languages"""
        self.languages = {}
        if LANGUAGES_DIR.exists():
            for lang_file in LANGUAGES_DIR.glob("*.json"):
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        lang_name = lang_file.stem
                        self.languages[lang_name] = json.load(f)
                except Exception as e:
                    print(f"Error loading language {lang_file.name}: {e}")
    
    def get_text(self, key, default=""):
        """Get translated text"""
        if self.current_language in self.languages:
            return self.languages[self.current_language].get(key, default or key)
        return default or key
    
    def reload_ui_language(self):
        """Reload UI with new language - Complete refresh"""
        # Update window title
        self.setWindowTitle(self.get_text("app_title", "ED-RAY AUTRA - Exoplanet Detection System"))
        
        # Update navigation buttons
        nav_labels = [
            self.get_text("menu_home", "🏠 Home"),
            self.get_text("menu_prediction", "🔮 Prediction"),
            self.get_text("menu_dataset", "📊 Data"),
            self.get_text("menu_training", "🧠 Training"),
            self.get_text("menu_courses", "📚 Courses"),
            self.get_text("menu_visualization", "🌍 Visualization"),
            self.get_text("menu_settings", "⚙️ Settings")
        ]
        
        for i, btn in enumerate(self.nav_buttons):
            if i < len(nav_labels):
                # Clear old text completely before setting new
                btn.setText("")
                btn.setText(nav_labels[i])
        
        # Force update of all pages by recreating them
        try:
            current_index = self.pages.currentIndex()
            
            # Clear and recreate pages to avoid language mixing
            # Note: This is a simplified approach - full recreation would be more complex
            # For now, we'll update key elements
            
            # Update status bar
            self.statusBar().showMessage(self.get_text("home_welcome", "Language updated successfully"))
            
            # Show message to user
            QTimer.singleShot(500, lambda: self.statusBar().showMessage(
                self.get_text("home_welcome", "Ready")
            ))
            
        except Exception as e:
            print(f"Error updating UI language: {e}")
            self.statusBar().showMessage("Language update completed")
    
    def apply_light_theme(self):
        """Apply light theme by default"""
        app = QApplication.instance()
        palette = QPalette()
        
        # Light theme colors
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        app.setPalette(palette)
    
    def apply_accessibility_settings(self):
        """Apply accessibility settings"""
        app = QApplication.instance()
        
        # Font size - Apply to main window and children
        font_size = self.accessibility_settings.get('font_size', 12)
        if font_size < 8:
            font_size = 12
        
        # Apply font to all widgets recursively
        font = QFont("Arial", font_size)
        for widget in self.findChildren(QWidget):
            widget.setFont(font)
        
        # High contrast mode
        if self.accessibility_settings.get('high_contrast', False):
            palette = QPalette()
            # Background colors
            palette.setColor(QPalette.Window, QColor(0, 0, 0))
            palette.setColor(QPalette.Base, QColor(0, 0, 0))
            palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
            palette.setColor(QPalette.Button, QColor(40, 40, 40))
            
            # Text colors
            palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
            palette.setColor(QPalette.Text, QColor(255, 255, 255))
            palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
            palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
            
            # Highlight colors (bright yellow for maximum contrast)
            palette.setColor(QPalette.Highlight, QColor(255, 255, 0))
            palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
            
            # Link colors
            palette.setColor(QPalette.Link, QColor(100, 200, 255))
            palette.setColor(QPalette.LinkVisited, QColor(200, 100, 255))
            
            app.setPalette(palette)
            self.statusBar().showMessage("High contrast mode enabled")
        else:
            # Reset to normal light theme palette
            palette = QPalette()
            
            # Light theme colors (default)
            palette.setColor(QPalette.Window, QColor(240, 240, 240))
            palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
            palette.setColor(QPalette.Base, QColor(255, 255, 255))
            palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
            palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
            palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
            palette.setColor(QPalette.Text, QColor(0, 0, 0))
            palette.setColor(QPalette.Button, QColor(240, 240, 240))
            palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
            palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
            palette.setColor(QPalette.Link, QColor(0, 0, 255))
            palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            
            app.setPalette(palette)
            self.statusBar().showMessage("Normal light theme restored")
        
        # Text-to-Speech
        if self.accessibility_settings['text_to_speech']:
            if not TTS_AVAILABLE:
                QMessageBox.warning(
                    self,
                    "Text-to-Speech",
                    "pyttsx3 module not installed.\n\nInstall it with:\npip install pyttsx3\n\nThen restart the application."
                )
            else:
                try:
                    if not hasattr(self, 'tts_engine') or self.tts_engine is None:
                        self.tts_engine = pyttsx3.init()
                        self.tts_engine.setProperty('rate', 150)
                        self.tts_engine.setProperty('volume', 0.9)
                    self.tts_engine.say("Text to speech activated")
                    self.tts_engine.runAndWait()
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "TTS Error",
                        f"Text-to-Speech error: {e}\n\nPlease check your system's TTS support."
                    )
        
        # Screen reader support
        if self.accessibility_settings['screen_reader']:
            # Add accessible attributes
            self.setAccessibleName("ED-RAY AUTRA Main Window")
            self.setAccessibleDescription("Exoplanet Detection and Analysis System")
        
        # Keyboard navigation - Enhanced
        if self.accessibility_settings['keyboard_nav']:
            self.setFocusPolicy(Qt.StrongFocus)
            # Enable Tab navigation for all interactive widgets
            for widget in self.findChildren(QWidget):
                if isinstance(widget, (QPushButton, QLineEdit, QTextEdit, QComboBox, 
                                      QSpinBox, QDoubleSpinBox, QSlider, QCheckBox)):
                    widget.setFocusPolicy(Qt.StrongFocus)
                    # Add visual focus indicator
                    widget.setStyleSheet(widget.styleSheet() + """
                        *:focus {
                            border: 2px solid #FFD700;
                            background-color: rgba(255, 215, 0, 0.1);
                        }
                    """)
            
            # Add keyboard shortcuts for main pages
            from PyQt5.QtWidgets import QShortcut
            from PyQt5.QtGui import QKeySequence
            
            shortcuts = [
                ("Ctrl+1", 0, "Home"),
                ("Ctrl+2", 1, "Prediction"),
                ("Ctrl+3", 2, "Data"),
                ("Ctrl+4", 3, "Training"),
                ("Ctrl+5", 4, "Courses"),
                ("Ctrl+6", 5, "Visualization"),
                ("Ctrl+7", 6, "Settings"),
            ]
            
            for key, index, name in shortcuts:
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(lambda i=index: self.change_page(i))
                # Store shortcuts to prevent garbage collection
                if not hasattr(self, 'shortcuts'):
                    self.shortcuts = []
                self.shortcuts.append(shortcut)
            
            self.statusBar().showMessage("Keyboard navigation enabled (Ctrl+1-7 for pages)")
    
    def on_prediction_made(self, result):
        """Handle prediction result"""
        self.statusBar().showMessage(f"Prediction: {result}")
        self.speak(f"Prediction result: {result}")
    
    def show_about(self):
        """Show About dialog"""
        QMessageBox.about(
            self,
            "About ED-RAY AUTRA",
            "<h2>ED-RAY AUTRA</h2>"
            "<p>Exoplanet Detection and Analysis System</p>"
            "<p><b>RAY AUTRA TEAM</b></p>"
            "<p>NASA SPACE APP CHALLENGE 2025</p>"
              "<p>"
            '<a href="https://github.com/ray1-cmd/ED-RAY-AUTRA-PROJECT">GitHub Project LINK</a>'
            "</p>"
            ""
        )

# ==================== SPLASH SCREEN ====================

class SplashScreen(QSplashScreen):
    """Custom splash screen with branding"""
    def __init__(self):
        # Create a pixmap for the splash screen
        splash_pix = QPixmap(600, 400)
        splash_pix.fill(QColor(15, 23, 42))  # Dark blue background
        super().__init__(splash_pix, Qt.WindowStaysOnTopHint)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setEnabled(False)
        
    def drawContents(self, painter):
        """Draw splash screen content"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background gradient
        from PyQt5.QtGui import QLinearGradient
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(15, 23, 42))
        gradient.setColorAt(1, QColor(30, 41, 59))
        painter.fillRect(0, 0, 600, 400, QBrush(gradient))
        
        # Main title - ED-RAY AUTRA
        painter.setPen(QColor(255, 255, 255))
        title_font = QFont("Arial", 32, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(0, 80, 600, 50, Qt.AlignCenter, "ED-RAY AUTRA")
        
        # Planet icon/emoji
        icon_font = QFont("Arial", 48)
        painter.setFont(icon_font)
        painter.drawText(0, 130, 600, 60, Qt.AlignCenter, "🪐")
        
        # Slogan
        painter.setPen(QColor(100, 181, 246))  # Light blue
        slogan_font = QFont("Arial", 18, QFont.Bold)
        painter.setFont(slogan_font)
        painter.drawText(0, 200, 600, 30, Qt.AlignCenter, "DETECTION - LEARNING")
        
        # NASA Space App Challenge
        painter.setPen(QColor(200, 200, 200))
        subtitle_font = QFont("Arial", 14)
        painter.setFont(subtitle_font)
        painter.drawText(0, 250, 600, 25, Qt.AlignCenter, "NASA SPACE APP CHALLENGE 2025")
        
        # Team name
        painter.setPen(QColor(255, 193, 7))  # Gold
        team_font = QFont("Arial", 16, QFont.Bold)
        painter.setFont(team_font)
        painter.drawText(0, 290, 600, 25, Qt.AlignCenter, "RAY AUTRA TEAM")
        
        # Motto
        painter.setPen(QColor(150, 150, 150))
        motto_font = QFont("Arial", 12, QFont.StyleItalic)
        painter.setFont(motto_font)
        painter.drawText(0, 330, 600, 20, Qt.AlignCenter, "INCLUDE ALL")
        
        # Loading indicator
        painter.setPen(QColor(100, 181, 246))
        loading_font = QFont("Arial", 10)
        painter.setFont(loading_font)
        painter.drawText(0, 370, 600, 20, Qt.AlignCenter, "Loading...")

# ==================== MAIN ====================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Load preferences first
    prefs = load_preferences()
    
    # Apply Material Design theme based on preferences
    theme = prefs.get('theme', 'dark')
    if QT_MATERIAL_AVAILABLE:
        try:
            if theme == 'dark':
                apply_stylesheet(app, theme='dark_blue.xml')
            else:
                apply_stylesheet(app, theme='light_blue.xml')
        except Exception as e:
            print(f"Theme error: {e}")
    else:
        print("qt-material disabled in the app code, using default theme")
    
    # Show splash screen for 3 seconds
    splash = SplashScreen()
    splash.show()
    app.processEvents()
    
    # Simple timer approach - no complex event loops
    import time
    start_time = time.time()
    while time.time() - start_time < 3.0:
        app.processEvents()
        time.sleep(0.01)  # Small delay to prevent CPU hogging
    
    # Close splash and show main window
    window = MainWindow()
    window.show()
    splash.close()
    
    sys.exit(app.exec_())
