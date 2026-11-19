import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


class PhasePortraitAnalyzer:
    def __init__(self, dx_str, dy_str, x_range=(-5, 5), y_range=(-5, 5)):
        self.dx_str = dx_str
        self.dy_str = dy_str
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        # Definir símbolos matemáticos
        self.x_sym, self.y_sym = sp.symbols('x y')

        # Parsear las cadenas de texto a expresiones simbólicas
        try:
            self.f_sym = sp.sympify(dx_str)
            self.g_sym = sp.sympify(dy_str)
        except sp.SympifyError as e:
            raise ValueError(f"Error al interpretar las ecuaciones: {e}")

        # Calcular el Jacobiano simbólico exacto
        self.J_sym = sp.Matrix([self.f_sym, self.g_sym]).jacobian([self.x_sym, self.y_sym])

        # Compilar funciones numéricas rápidas usando lambdify
        self.f_num = sp.lambdify((self.x_sym, self.y_sym), self.f_sym, modules=['numpy'])
        self.g_num = sp.lambdify((self.x_sym, self.y_sym), self.g_sym, modules=['numpy'])
        self.J_num = sp.lambdify((self.x_sym, self.y_sym), self.J_sym, modules=['numpy'])

    def system(self, t, state):
        x, y = state
        # Manejo seguro de operaciones vectorizadas o escalares
        dx = self.f_num(x, y)
        dy = self.g_num(x, y)
        # Si una ecuación es constante (ej. dx/dt = 1), expandir al tamaño de la malla
        if np.ndim(x) > 0:
            if np.ndim(dx) == 0:
                dx = np.full_like(x, dx, dtype=float)
            if np.ndim(dy) == 0:
                dy = np.full_like(y, dy, dtype=float)
        return [dx, dy]

    def find_equilibria(self, grid_density=8):
        """Busca puntos de equilibrio iniciando desde una cuadrícula de semillas."""
        x_seeds = np.linspace(self.x_min, self.x_max, grid_density)
        y_seeds = np.linspace(self.y_min, self.y_max, grid_density)
        equilibria = []

        for xs in x_seeds:
            for ys in y_seeds:
                try:
                    sol, info, ier, _ = fsolve(
                        lambda p: [float(self.f_num(p[0], p[1])), float(self.g_num(p[0], p[1]))],
                        [xs, ys],
                        full_output=True
                    )
                    if ier == 1:  # Convergencia exitosa
                        # Verificar que esté dentro de los límites (con un pequeño margen)
                        if (self.x_min - 1 <= sol[0] <= self.x_max + 1 and
                                self.y_min - 1 <= sol[1] <= self.y_max + 1):
                            # Verificar residuo cercano a cero
                            res = np.hypot(self.f_num(sol[0], sol[1]), self.g_num(sol[0], sol[1]))
                            if res < 1e-5:
                                # Evitar duplicados
                                if not any(np.allclose(sol, eq, atol=1e-3) for eq in equilibria):
                                    equilibria.append(sol)
                except Exception:
                    continue
        return equilibria

    def classify_point(self, point):
        """Calcula valores propios del Jacobiano y clasifica la estabilidad."""
        J_eval = np.array(self.J_num(point[0], point[1]), dtype=float)
        eigvals = np.linalg.eigvals(J_eval)
        l1, l2 = eigvals
        re1, re2 = np.real(l1), np.real(l2)
        im1, im2 = np.imag(l1), np.imag(l2)

        if np.isclose(im1, 0, atol=1e-5) and np.isclose(im2, 0, atol=1e-5):
            if re1 * re2 < 0:
                kind = "Punto Silla (Inestable)"
            elif re1 < 0 and re2 < 0:
                kind = "Nodo Estable (Sumidero)"
            elif re1 > 0 and re2 > 0:
                kind = "Nodo Inestable (Fuente)"
            else:
                kind = "Punto No Hiperbólico / Degenerado"
        else:
            if np.isclose(re1, 0, atol=1e-4):
                kind = "Centro (Estabilidad Neutral)"
            elif re1 < 0:
                kind = "Espiral Estable (Foco Atractor)"
            else:
                kind = "Espiral Inestable (Foco Repulsor)"

        return eigvals, kind

    def plot(self, density=25, t_max=15.0, style="quiver", save_path=None):
        """Genera el retrato de fase interactivo."""
        x = np.linspace(self.x_min, self.x_max, density)
        y = np.linspace(self.y_min, self.y_max, density)
        X, Y = np.meshgrid(x, y)
        DX, DY = self.system(0, [X, Y])
        magnitude = np.hypot(DX, DY)

        # Malla de alta resolución para isoclinas
        x_fine = np.linspace(self.x_min, self.x_max, 400)
        y_fine = np.linspace(self.y_min, self.y_max, 400)
        Xf, Yf = np.meshgrid(x_fine, y_fine)
        DXf, DYf = self.system(0, [Xf, Yf])

        fig, ax = plt.subplots(figsize=(10, 8), dpi=110)

        # 1. Campo vectorial
        if style == "stream":
            strm = ax.streamplot(X, Y, DX, DY, color=magnitude, cmap='viridis', density=1.5, linewidth=1.1)
            cbar = plt.colorbar(strm.lines, ax=ax, pad=0.02)
        else:
            mag_safe = np.where(magnitude == 0, 1.0, magnitude)
            quiv = ax.quiver(
                X, Y, DX / mag_safe, DY / mag_safe, magnitude,
                cmap='viridis', pivot='mid', scale=32, width=0.003, alpha=0.85
            )
            cbar = plt.colorbar(quiv, ax=ax, pad=0.02)
        cbar.set_label('Velocidad $\sqrt{(dx/dt)^2 + (dy/dt)^2}$')

        # 2. Isoclinas nulas
        ax.contour(Xf, Yf, DXf, levels=[0], colors='#e63946', linestyles='--', linewidths=1.5)
        ax.contour(Xf, Yf, DYf, levels=[0], colors='#1d3557', linestyles='--', linewidths=1.5)
        ax.plot([], [], color='#e63946', linestyle='--', linewidth=1.5, label='Isoclina $dx/dt = 0$')
        ax.plot([], [], color='#1d3557', linestyle='--', linewidth=1.5, label='Isoclina $dy/dt = 0$')

        # 3. Puntos de equilibrio
        equilibria = self.find_equilibria()
        print("\n" + "=" * 60)
        print(f"SISTEMA: dx/dt = {self.dx_str}  |  dy/dt = {self.dy_str}")
        print("=" * 60)
        if not equilibria:
            print("No se encontraron puntos de equilibrio en el dominio especificado.")
        for idx, eq in enumerate(equilibria):
            eigvals, kind = self.classify_point(eq)
            print(f"[{idx+1}] Punto: ({eq[0]:.4f}, {eq[1]:.4f})")
            print(f"    Valores propios: {eigvals[0]:.4f}, {eigvals[1]:.4f} -> {kind}")
            lbl = 'Punto de equilibrio' if idx == 0 else ""
            ax.plot(eq[0], eq[1], 'o', color='red', markeredgecolor='black', markersize=9, zorder=5, label=lbl)

        # Función auxiliar para integrar una trayectoria desde (x0, y0)
        def draw_trajectory(x0, y0, add_label=False):
            sol_f = solve_ivp(self.system, [0, t_max], [x0, y0], t_eval=np.linspace(0, t_max, 500), rtol=1e-7)
            sol_b = solve_ivp(self.system, [0, -t_max / 2], [x0, y0], t_eval=np.linspace(0, -t_max / 2, 250), rtol=1e-7)
            lbl = 'Trayectoria' if add_label else ""
            ax.plot(sol_f.y[0], sol_f.y[1], color='black', linewidth=1.4, label=lbl)
            ax.plot(sol_b.y[0], sol_b.y[1], color='black', linewidth=1.0, alpha=0.45)
            ax.plot(x0, y0, 'o', color='orange', markersize=4, zorder=4)

        # 4. Trayectorias iniciales por defecto (en una rejilla ligera)
        seed_x = np.linspace(self.x_min * 0.8, self.x_max * 0.8, 4)
        seed_y = np.linspace(self.y_min * 0.8, self.y_max * 0.8, 4)
        first = True
        for sx in seed_x:
            for sy in seed_y:
                if not np.isclose(sx, 0) or not np.isclose(sy, 0):
                    draw_trajectory(sx, sy, add_label=first)
                    first = False

        # 5. Interactividad: Clic en la gráfica para añadir trayectorias
        def on_click(event):
            if event.inaxes == ax and event.button == 1:
                draw_trajectory(event.xdata, event.ydata)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', on_click)

        # Estética
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.axhline(0, color='gray', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$y$', fontsize=12)
        ax.set_title(
            f"Retrato de Fase: $dx/dt = {sp.latex(self.f_sym)}$, $dy/dt = {sp.latex(self.g_sym)}$\n"
            "(Haz clic izquierdo en cualquier punto para trazar una trayectoria)",
            fontsize=11
        )
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', framealpha=0.9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"\nGráfica guardada exitosamente en: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generador interactivo de Retratos de Fase 2D.")
    parser.add_argument("--dx", type=str, help="Ecuación para dx/dt (ej. 'y')")
    parser.add_argument("--dy", type=str, help="Ecuación para dy/dt (ej. '-sin(x) - 0.5*y')")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-5.0, 5.0], help="Límites del eje X (min max)")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-5.0, 5.0], help="Límites del eje Y (min max)")
    parser.add_argument("--tmax", type=float, default=15.0, help="Tiempo máximo de integración")
    parser.add_argument("--style", choices=["quiver", "stream"], default="quiver", help="Estilo del campo vectorial")
    parser.add_argument("--save", type=str, default=None, help="Ruta para guardar la imagen (ej. retrato.png)")

    args = parser.parse_args()

    # Modo interactivo si no se pasan ecuaciones por argumentos
    if not args.dx or not args.dy:
        print("=== GENERADOR DE RETRATOS DE FASE 2D ===")
        print("Escribe las ecuaciones en función de 'x' e 'y'.")
        print("Funciones soportadas: sin, cos, tan, exp, log, sqrt, abs, etc.")
        print("Ejemplo (Péndulo amortiguado): dx/dt = y  |  dy/dt = -sin(x) - 0.3*y\n")
        
        dx_input = input("Ingresa dx/dt = ").strip() or "y"
        dy_input = input("Ingresa dy/dt = ").strip() or "-sin(x) - 0.3*y"
        
        xlim_str = input("Límites en X [default: -5 5]: ").strip()
        ylim_str = input("Límites en Y [default: -5 5]: ").strip()
        
        xlim = [float(v) for v in xlim_str.split()] if xlim_str else [-5.0, 5.0]
        ylim = [float(v) for v in ylim_str.split()] if ylim_str else [-5.0, 5.0]
    else:
        dx_input = args.dx
        dy_input = args.dy
        xlim = args.xlim
        ylim = args.ylim

    try:
        analyzer = PhasePortraitAnalyzer(dx_input, dy_input, x_range=xlim, y_range=ylim)
        analyzer.plot(t_max=args.tmax, style=args.style, save_path=args.save)
    except Exception as e:
        print(f"\n[Error]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
