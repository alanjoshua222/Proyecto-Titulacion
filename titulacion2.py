import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class DentalAnalyzerApp:
    def __init__(self, root):
        # --- CONFIGURACIÓN DE LA VENTANA ---
        self.root = root
        self.root.title("Software de Detección de Placa (Solo Interfaz)")
        self.root.geometry("1200x700")
        self.root.configure(bg="#2c3e50") # Fondo Azul Oscuro

        # Variable para guardar la imagen cargada en memoria
        self.current_image = None

        # --- INTERFAZ GRÁFICA (GUI) ---
        
        # 1. Panel Superior (Barra de Botones)
        top_frame = tk.Frame(root, bg="#34495e", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Estilo común para los botones
        btn_style = {"font": ("Arial", 12, "bold"), "bg": "#2980b9", "fg": "white", "padx": 20, "pady": 5}
        
        # Botón Cargar
        self.btn_load = tk.Button(top_frame, text="📂 Cargar Imagen", command=self.load_image, **btn_style)
        self.btn_load.pack(side=tk.LEFT, padx=20)

        # Botón Guardar (Deshabilitado por ahora)
        self.btn_save = tk.Button(top_frame, text="💾 Guardar Resultado", state=tk.DISABLED, **btn_style)
        self.btn_save.pack(side=tk.LEFT, padx=20)

        # 2. Panel Central (Área de Visualización)
        img_frame = tk.Frame(root, bg="#2c3e50")
        img_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Marco Izquierdo: Imagen Original
        self.lbl_img_original = tk.Label(img_frame, text="[ Imagen Original ]", bg="black", fg="gray", font=("Arial", 10))
        self.lbl_img_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)

        # Marco Derecho: Imagen Analizada (Vacío por ahora)
        self.lbl_img_processed = tk.Label(img_frame, text="[ Imagen Analizada ]", bg="black", fg="gray", font=("Arial", 10))
        self.lbl_img_processed.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10)

        # 3. Panel Inferior (Barra de Estado / Resultados)
        info_frame = tk.Frame(root, bg="#ecf0f1", height=100)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_stats = tk.Label(info_frame, text="Bienvenido. Por favor cargue una imagen.", 
                                  font=("Helvetica", 14), bg="#ecf0f1", fg="#2c3e50")
        self.lbl_stats.pack(pady=20)

    # --- LÓGICA BÁSICA (SOLO CARGA Y VISUALIZACIÓN) ---
    def load_image(self):
        # 1. Abrir explorador de archivos
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        
        if not file_path:
            return # Si el usuario cancela, no hacemos nada

        try:
            # 2. Cargar la imagen con PIL (Pillow)
            img_pil = Image.open(file_path)
            
            # 3. Redimensionar para que quepa en la ventana (Thumbnail)
            # Usamos 500x500 como referencia para mantener el diseño ordenado
            img_pil = img_pil.resize((500, 500), Image.Resampling.LANCZOS)
            
            # 4. Convertir a formato compatible con Tkinter
            img_tk = ImageTk.PhotoImage(img_pil)

            # 5. Mostrar en la etiqueta izquierda (Original)
            self.lbl_img_original.config(image=img_tk, text="") 
            self.lbl_img_original.image = img_tk # ¡Importante! Guardar referencia para que no se borre

            # 6. Actualizar el texto de abajo
            self.lbl_stats.config(text=f"Imagen cargada exitosamente: {file_path.split('/')[-1]}")
            
            # Nota: Aquí iría la llamada al procesamiento en el futuro...
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

# --- PUNTO DE ENTRADA ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DentalAnalyzerApp(root)
    root.mainloop()