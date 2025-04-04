import numpy as np  
import matplotlib.pyplot as plt  #
import tkinter as tk  
from tkinter import ttk  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  
from matplotlib.animation import FuncAnimation 
from matplotlib.patches import FancyArrowPatch 

class Arrow(FancyArrowPatch):
    def __init__(self, x, y, dx, dy, *args, **kwargs):
        """
            x, y: Coordenadas de origen del vector
            dx, dy: Componentes del vector (desplazamiento)
            *args, **kwargs: Argumentos adicionales para personalización
        """
        super().__init__((x, y), (x+dx, y+dy), *args, **kwargs)
        # Configura el estilo visual de la punta de la flecha
        self.set_arrowstyle('->,head_width=5,head_length=10')

class ParabolicSimulator:

    def __init__(self, root):
        self.root = root
        self.setup_ui()  # Configura todos los elementos de la interfaz
        self.setup_variables()  # Inicializa las variables de estado
        
    def setup_ui(self):
        # Configuración básica de la ventana principal
        self.root.title("Movimiento Parabólico")
        self.root.geometry("1300x900")
        
        # Marco principal que contendrá todos los demás elementos
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Marco izquierdo para controles y resultados
        left_frame = ttk.Frame(main_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)  # Fija el ancho del marco
        
        # Marco derecho para la animación
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Sección de parámetros de entrada
        control_frame = ttk.LabelFrame(left_frame, text="Parámetros de Entrada", padding=10)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Campo para velocidad inicial
        ttk.Label(control_frame, text="Velocidad inicial (m/s):").pack(pady=2)
        self.v0_entry = ttk.Entry(control_frame)
        self.v0_entry.pack(fill=tk.X, pady=2)
        self.v0_entry.insert(0, "30")  # Valor por defecto
        
        # Campo para ángulo de lanzamiento
        ttk.Label(control_frame, text="Ángulo de lanzamiento (°):").pack(pady=2)
        self.angle_entry = ttk.Entry(control_frame)
        self.angle_entry.pack(fill=tk.X, pady=2)
        self.angle_entry.insert(0, "45")  # Valor por defecto
        
        # Campo para altura inicial
        ttk.Label(control_frame, text="Altura inicial (m):").pack(pady=2)
        self.y0_entry = ttk.Entry(control_frame)
        self.y0_entry.pack(fill=tk.X, pady=2)
        self.y0_entry.insert(0, "0")  # Valor por defecto
        
        # Campo para masa del proyectil
        ttk.Label(control_frame, text="Masa del proyectil (kg):").pack(pady=2)
        self.mass_entry = ttk.Entry(control_frame)
        self.mass_entry.pack(fill=tk.X, pady=2)
        self.mass_entry.insert(0, "1.0")  # Valor por defecto
        
        # Sección de opciones de visualización
        options_frame = ttk.LabelFrame(control_frame, text="Opciones de Visualización", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Checkboxes para mostrar/ocultar vectores
        self.vel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Mostrar vector velocidad", variable=self.vel_var).pack(anchor=tk.W)
        
        self.acc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Mostrar vector aceleración", variable=self.acc_var).pack(anchor=tk.W)
        
        self.force_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Mostrar vector fuerza", variable=self.force_var).pack(anchor=tk.W)
        
        # Control deslizante para escala de vectores
        ttk.Label(options_frame, text="Escala de vectores:").pack(anchor=tk.W)
        self.scale_slider = tk.Scale(options_frame, from_=1, to=20, orient=tk.HORIZONTAL)
        self.scale_slider.pack(fill=tk.X)
        self.scale_slider.set(8)  # Valor por defecto
        
        # Control deslizante para velocidad de simulación
        ttk.Label(options_frame, text="Velocidad de simulación:").pack(anchor=tk.W)
        self.speed_slider = tk.Scale(options_frame, from_=1, to=10, orient=tk.HORIZONTAL)
        self.speed_slider.pack(fill=tk.X)
        self.speed_slider.set(5)  # Valor por defecto
        
        # Marco para botones de control
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Botones principales
        ttk.Button(btn_frame, text="Iniciar Simulación", command=self.run_simulation).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="Pausar/Reanudar", command=self.toggle_pause).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="Reiniciar", command=self.safe_reset).pack(side=tk.LEFT, expand=True)
        
        # Sección de resultados en tiempo real
        self.results_frame = ttk.LabelFrame(left_frame, text="Resultados en Tiempo Real", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Etiquetas para mostrar los resultados
        self.result_labels = {
            'tiempo': ttk.Label(self.results_frame, text="Tiempo: 0.00 s"),
            'distancia': ttk.Label(self.results_frame, text="Distancia: 0.00 m"),
            'altura': ttk.Label(self.results_frame, text="Altura: 0.00 m"),
            'velocidad': ttk.Label(self.results_frame, text="Velocidad: 0.00 m/s"),
            'alcance': ttk.Label(self.results_frame, text="Alcance máximo: 0.00 m"),
            'altura_max': ttk.Label(self.results_frame, text="Altura máxima: 0.00 m"),
            'tiempo_vuelo': ttk.Label(self.results_frame, text="Tiempo total: 0.00 s")
        }
        
        # Coloca todas las etiquetas en la interfaz
        for label in self.result_labels.values():
            label.pack(anchor=tk.W, pady=2)
        
        # Marco para la animación
        self.animation_frame = ttk.Frame(right_frame)
        self.animation_frame.pack(fill=tk.BOTH, expand=True)
    
    def setup_variables(self):
        self.animation = None  # Objeto de animación
        self.is_paused = False  # Estado de pausa
        self.canvas = None  # Lienzo para el gráfico
        self.fig = None  # Figura de matplotlib
        self.ax = None  # Ejes de matplotlib
        self.simulation_running = False  # Bandera de simulación activa
        self.current_frame = 0  # Frame actual de la animación
        self.sim_data = None  # Almacenamiento de datos de simulación
        
    def get_input_values(self):
        """
        Obtiene y valida los valores ingresados por el usuario.
        
        Returns:
            tuple: (v0, angle, y0, mass) o (None, None, None, None) si hay error
        """
        try:
            # Intenta convertir los valores a float
            v0 = float(self.v0_entry.get())
            angle = float(self.angle_entry.get())
            y0 = float(self.y0_entry.get())
            mass = float(self.mass_entry.get())
            return v0, angle, y0, mass
        except ValueError:
            # Maneja el error si los valores no son numéricos
            self.show_error("Por favor ingrese valores numéricos válidos")
            return None, None, None, None
        
    def show_error(self, message):
        """Muestra un mensaje de error en todas las etiquetas de resultados."""
        for label in self.result_labels.values():
            label.config(text=f"Error: {message}")
        
    def update_results(self, t, x, y, v, a, alcance, altura_max, tiempo_vuelo):
        """
        Actualiza las etiquetas de resultados con los valores actuales.
        
        Args:
            t: Array de tiempos
            x, y: Arrays de posiciones
            v: Tupla con componentes de velocidad (vx, vy)
            a: Tupla con componentes de aceleración (ax, ay)
            alcance: Distancia horizontal máxima
            altura_max: Altura máxima alcanzada
            tiempo_vuelo: Tiempo total de vuelo
        """
        if self.current_frame < len(t):
            # Calcula los valores actuales
            current_time = t[self.current_frame]
            current_dist = x[self.current_frame]
            current_height = y[self.current_frame]
            current_speed = np.sqrt(v[0][self.current_frame]**2 + v[1][self.current_frame]**2)
            
            # Actualiza todas las etiquetas
            self.result_labels['tiempo'].config(text=f"Tiempo: {current_time:.2f} s")
            self.result_labels['distancia'].config(text=f"Distancia: {current_dist:.2f} m")
            self.result_labels['altura'].config(text=f"Altura: {current_height:.2f} m")
            self.result_labels['velocidad'].config(text=f"Velocidad: {current_speed:.2f} m/s")
            self.result_labels['alcance'].config(text=f"Alcance máximo: {alcance:.2f} m")
            self.result_labels['altura_max'].config(text=f"Altura máxima: {altura_max:.2f} m")
            self.result_labels['tiempo_vuelo'].config(text=f"Tiempo total: {tiempo_vuelo:.2f} s")
        
    def calculate_trajectory(self, v0, angle, y0):
        """
        Args:
            v0: Velocidad inicial (m/s)
            angle: Ángulo de lanzamiento (grados)
            y0: Altura inicial (m)
            
        Returns:
            tuple: (x, y, t, v, a, alcance, altura_max, tiempo_vuelo)
        """
        g = 9.81  # Aceleración gravitatoria (m/s²)
        angle_rad = np.radians(angle)  # Convierte ángulo a radianes
        
        # Componentes de la velocidad inicial
        v0x = v0 * np.cos(angle_rad)  # Componente horizontal
        v0y = v0 * np.sin(angle_rad)  # Componente vertical
        
        # Cálculo del tiempo máximo de vuelo (considerando altura inicial)
        t_max = (v0y + np.sqrt(v0y**2 + 2*g*y0)) / g
        
        # Generación de arrays de tiempo y posición
        t = np.linspace(0, t_max, 150)  # 150 puntos temporales
        x = v0x * t  # Posiciones horizontales (movimiento uniforme)
        y = y0 + v0y * t - 0.5 * g * t**2  # Posiciones verticales (movimiento acelerado)
        
        # Cálculo de velocidad y aceleración
        vx = np.full_like(t, v0x)  # Velocidad horizontal constante
        vy = v0y - g * t  # Velocidad vertical con aceleración
        ax = np.zeros_like(t)  # Aceleración horizontal nula
        ay = np.full_like(t, -g)  # Aceleración vertical constante (gravedad)
        
        # Características importantes del movimiento
        alcance = x[-1]  # Distancia horizontal máxima
        altura_max = np.max(y)  # Altura máxima alcanzada
        tiempo_vuelo = t[-1]  # Tiempo total de vuelo
        
        return x, y, t, (vx, vy), (ax, ay), alcance, altura_max, tiempo_vuelo
    
    def run_simulation(self):
        """Inicia o reanuda la simulación con los parámetros actuales."""
        if self.is_paused:
            self.resume_simulation()
            return
            
        # Obtiene los valores de entrada
        v0, angle, y0, mass = self.get_input_values()
        if None in [v0, angle, y0, mass]:
            return  # Si hay error, no continúa
            
        self.safe_reset()  # Limpia cualquier simulación previa
        self.simulation_running = True
        
        # Calcula la trayectoria
        self.sim_data = self.calculate_trajectory(v0, angle, y0)
        x, y, t, v, a, alcance, altura_max, tiempo_vuelo = self.sim_data
        
        # Configura el gráfico
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(0, max(x)*1.1)  # Límites con 10% de margen
        self.ax.set_ylim(0, max(y)*1.1)
        self.ax.set_xlabel("Distancia (m)")
        self.ax.set_ylabel("Altura (m)")
        self.ax.grid(True, alpha=0.3)  # Cuadrícula semitransparente
        self.ax.set_title("Simulación de Movimiento Parabólico")
        
        # Elementos gráficos iniciales
        self.line, = self.ax.plot([], [], 'b-', lw=2, alpha=0.7)  # Línea de trayectoria
        self.point, = self.ax.plot([], [], 'ro', ms=10)  # Punto del proyectil
        
        # Vectores (flechas)
        scale = self.scale_slider.get()  # Factor de escala para visualización
        
        # Creación de flechas para los vectores
        self.vel_arrow = Arrow(0, 0, 0, 0, color='#2ecc71', lw=2, alpha=0.8)  # Verde
        self.acc_arrow = Arrow(0, 0, 0, 0, color='#e74c3c', lw=2, alpha=0.8)  # Rojo
        self.force_arrow = Arrow(0, 0, 0, 0, color='#9b59b6', lw=2, alpha=0.8)  # Morado
        
        # Añade las flechas al gráfico
        self.ax.add_patch(self.vel_arrow)
        self.ax.add_patch(self.acc_arrow)
        self.ax.add_patch(self.force_arrow)
        
        # Actualiza la visibilidad según las opciones
        self.update_vector_visibility()
        
        # Configura la leyenda del gráfico
        self.ax.legend(handles=[
            plt.Line2D([0], [0], color='#2ecc71', lw=2, label='Velocidad'),
            plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Aceleración'),
            plt.Line2D([0], [0], color='#9b59b6', lw=2, label='Fuerza (F=ma)')
        ], loc='upper right')
        
        # Función que se ejecuta en cada frame de la animación
        def update(frame):
            """Actualiza la animación para cada frame."""
            if self.is_paused or not self.simulation_running:
                return self.point, self.line, self.vel_arrow, self.acc_arrow, self.force_arrow
                
            self.current_frame = frame
            
            # Actualiza posición del proyectil y trayectoria
            self.point.set_data([x[frame]], [y[frame]])
            self.line.set_data(x[:frame+1], y[:frame+1])
            
            # Obtiene el factor de escala actual
            scale = self.scale_slider.get()
            
            # Actualiza vector velocidad si está visible
            if self.vel_var.get():
                self.vel_arrow.set_positions(
                    (x[frame], y[frame]), 
                    (x[frame] + v[0][frame]/scale, y[frame] + v[1][frame]/scale)
                )
                self.vel_arrow.set_visible(True)
            
            # Actualiza vector aceleración si está visible
            if self.acc_var.get():
                self.acc_arrow.set_positions(
                    (x[frame], y[frame]), 
                    (x[frame] + a[0][frame]/scale, y[frame] + a[1][frame]/scale)
                )
                self.acc_arrow.set_visible(True)
            
            # Actualiza vector fuerza si está visible (F = m*a)
            if self.force_var.get():
                self.force_arrow.set_positions(
                    (x[frame], y[frame]), 
                    (x[frame] + a[0][frame]*mass/scale, y[frame] + a[1][frame]*mass/scale)
                )
                self.force_arrow.set_visible(True)
            
            # Actualiza los resultados numéricos
            self.update_results(t, x, y, v, a, alcance, altura_max, tiempo_vuelo)
            
            return self.point, self.line, self.vel_arrow, self.acc_arrow, self.force_arrow
        
        # Configura la velocidad de la animación
        speed = self.speed_slider.get()
        interval = max(5, 50 - (speed * 5))  # Calcula el intervalo entre frames
        
        # Crea la animación
        self.animation = FuncAnimation(
            self.fig, 
            update, 
            frames=len(t),  # Número total de frames
            interval=interval,  # Tiempo entre frames (ms)
            blit=True  # Optimización para renderizado
        )
        
        # Integra el gráfico en la interfaz de tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.animation_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_vector_visibility(self):
        """Actualiza la visibilidad de los vectores según las opciones seleccionadas."""
        if hasattr(self, 'vel_arrow'):
            self.vel_arrow.set_visible(self.vel_var.get())
        if hasattr(self, 'acc_arrow'):
            self.acc_arrow.set_visible(self.acc_var.get())
        if hasattr(self, 'force_arrow'):
            self.force_arrow.set_visible(self.force_var.get())
        
    def toggle_pause(self):
        """Alterna el estado de pausa de la simulación."""
        self.is_paused = not self.is_paused
        
    def resume_simulation(self):
        """Reanuda la simulación pausada."""
        self.is_paused = False
        
    def safe_reset(self):
        """Reinicia completamente la simulación, limpiando todos los recursos."""
        self.simulation_running = False
        self.current_frame = 0
        
        # Intenta detener la animación de manera segura
        try:
            if self.animation and self.animation.event_source:
                self.animation.event_source.stop()
        except (AttributeError, RuntimeError):
            pass
            
        # Elimina el canvas si existe
        if self.canvas:
            try:
                self.canvas.get_tk_widget().destroy()
            except (tk.TclError, AttributeError):
                pass
                
        # Cierra la figura de matplotlib si existe
        if self.fig:
            try:
                plt.close(self.fig)
            except:
                pass
                
        # Reinicia el estado
        self.is_paused = False
        self.animation = None
        self.canvas = None
        self.fig = None
        self.ax = None
        
        # Restablece las etiquetas de resultados
        for label in self.result_labels.values():
            label.config(text=label.cget("text").split(":")[0] + ": 0.00")


if __name__ == "__main__":
    """Punto de entrada principal de la aplicación."""
    root = tk.Tk()  # Crea la ventana principal
    app = ParabolicSimulator(root)  # Crea la instancia del simulador
    root.mainloop()  # Inicia el bucle principal de la interfaz