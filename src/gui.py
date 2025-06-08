import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os

class TireClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tire Classifier")
        self.root.geometry("900x700")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["New", "Worn"]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.setup_ui()
        
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill='both', expand=True, padx=40, pady=40)
        
        header_frame = tk.Frame(main_container, bg='#1e1e1e')
        header_frame.pack(fill='x', pady=(0, 30))
        
        title_label = tk.Label(
            header_frame, 
            text="Tire Condition Classifier", 
            font=("Segoe UI", 28, "normal"), 
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame, 
            text="Upload an image to analyze tire condition", 
            font=("Segoe UI", 12), 
            bg='#1e1e1e',
            fg='#888888'
        )
        subtitle_label.pack(pady=(5, 0))
        
        content_frame = tk.Frame(main_container, bg='#1e1e1e')
        content_frame.pack(fill='both', expand=True)
        
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='flat', bd=0)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 20))
        
        image_container = tk.Frame(left_panel, bg='#2d2d2d')
        image_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.image_label = tk.Label(
            image_container, 
            text="Drop your image here\nor click upload below",
            font=("Segoe UI", 14),
            bg='#2d2d2d',
            fg='#666666',
            justify='center'
        )
        self.image_label.pack(expand=True)
        
        right_panel = tk.Frame(content_frame, bg='#1e1e1e', width=300)
        right_panel.pack(side='right', fill='y', padx=(20, 0))
        right_panel.pack_propagate(False) 
        
        controls_frame = tk.Frame(right_panel, bg='#1e1e1e')
        controls_frame.pack(fill='x', pady=(20, 0))
        
        self.upload_btn = tk.Button(
            controls_frame,
            text="Upload Image",
            command=self.select_image,
            font=("Segoe UI", 12, "normal"),
            bg='#007acc',
            fg='white',
            border=0,
            relief='flat',
            cursor='hand2',
            padx=30,
            pady=15,
            activebackground='#005a9e',
            activeforeground='white'
        )
        self.upload_btn.pack(fill='x', pady=(0, 15))
        
        self.load_model_btn = tk.Button(
            controls_frame,
            text="Load Custom Model",
            command=self.load_model,
            font=("Segoe UI", 10),
            bg='#333333',
            fg='#cccccc',
            border=0,
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=8,
            activebackground='#444444',
            activeforeground='white'
        )
        self.load_model_btn.pack(fill='x')
        
        status_frame = tk.Frame(right_panel, bg='#1e1e1e')
        status_frame.pack(fill='x', pady=(30, 0))
        
        status_title = tk.Label(
            status_frame,
            text="Status",
            font=("Segoe UI", 12, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        status_title.pack(anchor='w')
        
        self.status_label = tk.Label(
            status_frame, 
            text="Ready to analyze", 
            font=("Segoe UI", 10),
            bg='#1e1e1e',
            fg='#888888',
            wraplength=250,
            justify='left'
        )
        self.status_label.pack(anchor='w', pady=(5, 0))
        
        results_frame = tk.Frame(right_panel, bg='#1e1e1e')
        results_frame.pack(fill='x', pady=(40, 0))
        
        results_title = tk.Label(
            results_frame,
            text="Results",
            font=("Segoe UI", 12, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        results_title.pack(anchor='w')
        
        self.result_card = tk.Frame(results_frame, bg='#2d2d2d', relief='flat', bd=0)
        self.result_card.pack(fill='x', pady=(10, 0))
        
        self.result_label = tk.Label(
            self.result_card,
            text="No analysis yet",
            font=("Segoe UI", 14),
            bg='#2d2d2d',
            fg='#666666',
            padx=20,
            pady=20
        )
        self.result_label.pack()
        
        self.progress = ttk.Progressbar(
            right_panel, 
            mode='indeterminate',
            length=250,
            style='Custom.Horizontal.TProgressbar'
        )
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.Horizontal.TProgressbar', 
                       background='#007acc',
                       troughcolor='#333333',
                       borderwidth=0,
                       lightcolor='#007acc',
                       darkcolor='#007acc')
        
    def get_model(self, num_classes=2):
        """Tworzy model ResNet18"""
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
        
    def load_model(self):
        """Ładuje wytrenowany model"""
        try:
            model_path = filedialog.askopenfilename(
                title="Select model file",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
            )
            
            if not model_path:
                return
                
            self.model = self.get_model(num_classes=2)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            self.status_label.config(
                text="✓ Custom model loaded successfully", 
                fg='#4CAF50'
            )
            self.load_model_btn.config(text="✓ Custom Model Loaded", bg='#4CAF50', fg='white')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_label.config(
                text="✗ Error loading model", 
                fg='#f44336'
            )
    
    def select_image(self):
        """Wybiera i klasyfikuje zdjęcie"""
        try:
            image_path = filedialog.askopenfilename(
                title="Select tire image",
                filetypes=[
                    ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                    ("All Files", "*.*")
                ]
            )
            
            if not image_path:
                return
            
            self.progress.pack(pady=(20, 0))
            self.progress.start()
            self.status_label.config(text="Analyzing image...", fg='#007acc')
            self.root.update()
            
            self.display_image(image_path)
            
            if self.model:
                prediction, confidence = self.classify_image(image_path)
                self.display_result(prediction, confidence)
            else:
                self.result_label.config(
                    text="Model not loaded\nPlease load a model first",
                    fg='#f44336'
                )
                self.status_label.config(text="Model required for analysis", fg='#f44336')
            
            self.progress.stop()
            self.progress.pack_forget()
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            self.status_label.config(text="Error processing image", fg='#f44336')
    
    def display_image(self, image_path):
        """Wyświetla obraz w GUI"""
        image = Image.open(image_path)
        
        display_width, display_height = 400, 400
        image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo, text="", bg='#2d2d2d')
        self.image_label.image = photo
    
    def classify_image(self, image_path):
        """Klasyfikuje obraz"""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        return predicted.item(), confidence.item()
    
    def display_result(self, prediction, confidence):
        """Wyświetla wynik klasyfikacji"""
        class_name = self.class_names[prediction]
        confidence_percent = confidence * 100
        
        if prediction == 0:  
            color = '#4CAF50'
            status_icon = '✓'
            bg_color = '#1b5e20'
        else:  
            color = '#ff5722'
            status_icon = '⚠'
            bg_color = '#bf360c'
        
        result_text = f"{status_icon} {class_name}\n{confidence_percent:.1f}% confidence"
        self.result_label.config(
            text=result_text, 
            fg='white',
            bg=color,
            font=("Segoe UI", 12, "bold")
        )
        self.result_card.config(bg=color)
        
        self.status_label.config(
            text=f"Analysis complete: {class_name.lower()} tire detected",
            fg=color
        )

def main():
    default_model_path = "model_best.pt"
    
    root = tk.Tk()
    app = TireClassifierGUI(root)
    
    if os.path.exists(default_model_path):
        try:
            app.model = app.get_model(num_classes=2)
            app.model.load_state_dict(torch.load(default_model_path, map_location=app.device))
            app.model.to(app.device)
            app.model.eval()
            
            app.status_label.config(
                text="✓ Default model loaded and ready", 
                fg='#4CAF50'
            )
            app.load_model_btn.config(text="✓ Default Model Loaded", bg='#4CAF50', fg='white')
        except:
            app.status_label.config(
                text="Default model found but failed to load", 
                fg='#ff9800'
            )
    else:
        app.status_label.config(
            text="No default model found - load custom model", 
            fg='#ff9800'
        )
    
    root.mainloop()

if __name__ == "__main__":
    main()