import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

class DentalAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Placa bacteriana")
 
        self.root.geometry("1350x650") 
        self.root.configure(bg="#2c3e50")

        # Variables de estado
        self.original_cv_image = None
        self.processed_image = None  # Aquí guardaremos la imagen final para descargar
        self.binary_mask = None
       
        # --- UI SETUP ---
        top_frame = tk.Frame(root, bg="#34495e", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"font": ("Arial", 12, "bold"), "bg": "#2980b9", "fg": "white", "padx": 20}
        
        # Botón Cargar
        tk.Button(top_frame, text="📂 Cargar Imagen", command=self.load_image, **btn_style).pack(side=tk.LEFT, padx=20)
        
        # CORRECCIÓN 1: Asignamos el botón a self.btn_save para poder activarlo después
        self.btn_save = tk.Button(top_frame, text="💾 Descargar Imagen", command=self.save_image, state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.LEFT, padx=20) 

        img_frame = tk.Frame(root, bg="#2c3e50")
        img_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        self.lbl_original = tk.Label(img_frame, text="Original (Letterbox)", bg="black", fg="white")
        self.lbl_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

        self.lbl_binary = tk.Label(img_frame, text="Máscara Binaria (Detección)", bg="black", fg="white")
        self.lbl_binary.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)


    def resize_with_padding(self, img, target_size=(640, 480)):
        """Redimensiona manteniendo el aspecto y rellena con negro (Letterboxing)"""
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
    
    # Corregido typo 'slef' -> 'self'
    def overlay_mask(self, original, mask):
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

        # Reiniciar estado al cargar nueva imagen
        self.btn_save.config(state=tk.DISABLED)
        
        self.original_cv_image = self.resize_with_padding(img, target_size=(640,480))
        self.create_tooth_mask(self.original_cv_image)

    def create_tooth_mask(self, img):
        # 1. Preprocesamiento
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 2. Máscaras de color
        lower_white = np.array([0,0,60])
        upper_white = np.array([180,90,255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        lower_yellow = np.array([20,50,50])
        upper_yellow = np.array([40,255,255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

        # 3. Limpieza morfológica
        kernel = np.ones((9,9), np.uint8)
        binary_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=3)

        # 4. Encontrar contorno más grande (Diente principal)
        contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(binary_clean)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5000:
                cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        self.binary_mask = final_mask

        # Mostrar Original con Grid
        original_grid = self.draw_grid(self.original_cv_image, color=(150, 150, 150))
        self.show_image(original_grid, self.lbl_original)

        # 5. Recorte y Detección de Placa
        tooth_cutout_grid = cv2.bitwise_and(self.original_cv_image, self.original_cv_image, mask=self.binary_mask)
        
        # Análisis de placa en el recorte
        hsv_cutout = cv2.cvtColor(tooth_cutout_grid, cv2.COLOR_BGR2HSV)
        lower_plaque = np.array([20,50,80])
        upper_plaque = np.array([35,255,220])
        mask_plaque = cv2.inRange(hsv_cutout, lower_plaque, upper_plaque)
        contours_plaque, _ = cv2.findContours(mask_plaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_plaque_area = 0
        for c in contours_plaque:
            total_plaque_area += cv2.contourArea(c)

        # Dibujar contornos de placa sobre el recorte
        cv2.drawContours(tooth_cutout_grid, contours_plaque, -1, (255,255,0), 2) # Amarillo cyan para contraste
        
        # Agregar grid al resultado final
        result_grid = self.draw_grid(tooth_cutout_grid, color=(150,150,150))
        
        # CORRECCIÓN 2: Guardar la imagen procesada en self para poder descargarla después
        self.processed_image = result_grid
        
        self.show_image(result_grid, self.lbl_binary)
        print(f"Área de placa detectada: {total_plaque_area} px")

        # CORRECCIÓN 3: Activar el botón (ahora sí funciona porque self.btn_save existe)
        self.btn_save.config(state=tk.NORMAL)

    def show_image(self, cv_img, label_widget, is_gray=False):
        if is_gray:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk

    def save_image(self):
        # Verificamos si hay imagen procesada
        if self.processed_image is None: 
            messagebox.showwarning("Aviso", "No hay imagen procesada para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")],
            title="Guardar Resultado"
        )

        if file_path:
            # Guardamos self.processed_image en lugar de self.original_cv_image
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Éxito", f"Imagen guardada en:\n{file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()