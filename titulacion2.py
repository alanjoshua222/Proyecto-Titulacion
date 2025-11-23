import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class DentalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paso 1: Separación Diente/Encía (Blanco y Negro)")
        # Ajuste de geometría solicitado
        self.root.geometry("1350x650") 
        self.root.configure(bg="#2c3e50")

        self.original_cv_image = None
        self.binary_mask = None

        # --- INTERFAZ ---
        top_frame = tk.Frame(root, bg="#34495e", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Arial", 12, "bold"), "bg": "#2980b9", "fg": "white", "padx": 20}
        tk.Button(top_frame, text="📂 Cargar Imagen", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=20)

        # Panel de imágenes
        img_frame = tk.Frame(root, bg="#2c3e50")
        img_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Izquierda: Original
        self.lbl_original = tk.Label(img_frame, text="Original (Letterbox)", bg="black", fg="white")
        self.lbl_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        # Derecha: La Máscara B/N
        self.lbl_binary = tk.Label(img_frame, text="Máscara Binaria (Detección)", bg="black", fg="white")
        self.lbl_binary.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

    # --- FUNCIÓN NUEVA: REDIMENSIONADO CON RELLENO NEGRO ---
    def resize_with_padding(self, img, target_size=(640, 480)):
        """
        Redimensiona la imagen para que quepa dentro de target_size 
        sin deformarse, rellenando el espacio sobrante con negro.
        Esto mantiene la interfaz gráfica fija.
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # 1. Calcular la escala para que la imagen quepa completa (fit)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 2. Redimensionar la imagen real
        resized = cv2.resize(img, (new_w, new_h))
        
        # 3. Crear un lienzo negro del tamaño EXACTO del recuadro
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 4. Calcular coordenadas para centrar la imagen
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # 5. Pegar la imagen en el centro del lienzo
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

    def resize_with_padding(self, img, target_size=(640, 480)):
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas
    
    def draw_grid(self, img, grid_size=40, color=(200,200,200)):
        img_grid = img.copy()
        h, w = img_grid.shape[:2]

        for x in range(0, w, grid_size):
            cv2.line(img_grid, (x,0), (x,h), color, 1)

        for y in range(0, h, grid_size):
            cv2.line(img_grid, (0,y), (w,y), color, 1)

        return img_grid
    
    def overlay_mask(slef, original, mask):
        red_layer = np.zeros_like(original)
        red_layer[:] = (0,0,255)

        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(original, original, mask=mask_inv)
        fg = cv2.bitwise_and(red_layer, red_layer, mask=mask)
        combined = cv2.add(bg, fg)

        return cv2.addWeighted(original, 0.7, combined, 0.3, 0) 

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return

        img = cv2.imread(path)
        if img is None: return

        self.original_cv_image = self.resize_with_padding(img, target_size=(640,480))
        self.create_tooth_mask(self.original_cv_image)

    def create_tooth_mask(self, img):
        # 1. Lógica de Detección (Tu código)
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0,0,60])
        upper_white = np.array([180,90,255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([20,50,50])
        upper_yellow = np.array([40,255,255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

        kernel = np.ones((10,10), np.uint8)
    
        binary_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        self.binary_mask = binary_clean

        # --- VISUALIZACIÓN CON CUADRÍCULA ---
        
        # 1. Panel Izquierdo: Original + Grid Gris
        # IMPORTANTE: Asegúrate de que esta línea esté alineada con las de arriba
        original_grid = self.draw_grid(self.original_cv_image, color=(150, 150, 150))
        self.show_image(original_grid, self.lbl_original)

        # 2. Panel Derecho: Máscara B/N + Grid Gris
        binary_bgr = cv2.cvtColor(self.binary_mask, cv2.COLOR_GRAY2BGR)
        
        binary_grid = self.draw_grid(binary_bgr, color=(150, 150, 150))
        
        self.show_image(binary_grid, self.lbl_binary)

    def show_image(self, cv_img, label_widget, is_gray=False):
        if is_gray:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()