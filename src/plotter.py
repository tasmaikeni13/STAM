import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import numpy as np
from .config import Config
from scipy.interpolate import griddata

class LivePlotter:
    def __init__(self, X, Y, train_Z, val_Z, grad_x, grad_y, val_grad_x, val_grad_y):
        self.X = X
        self.Y = Y
        self.train_Z = train_Z
        self.val_Z = val_Z
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.val_grad_x = val_grad_x
        self.val_grad_y = val_grad_y
        
        plt.ion()
        self.fig = plt.figure(figsize=(20, 20))
        
        self.ax1 = self.fig.add_subplot(221, projection='3d')
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223, projection='3d')
        self.ax4 = self.fig.add_subplot(224)
        
        # 1. Train 3D
        self.surf_train = self.ax1.plot_surface(X, Y, train_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')
        self.line_train3d, = self.ax1.plot([], [], [], color='black', linewidth=2, marker='o', markersize=3)
        self.ax1.set_title(f'Train Loss Landscape ({Config.DATASET})')
        
        # 2. Train 2D
        self.ax2.contourf(X, Y, train_Z, levels=50, cmap=Config.COLORMAP, alpha=0.8)
        step = 4
        self.ax2.quiver(X[::step, ::step], Y[::step, ::step], -grad_x[::step, ::step], -grad_y[::step, ::step], color='white', alpha=0.5)
        self.line_train2d, = self.ax2.plot([], [], color='black', linewidth=2, marker='o', markersize=3)
        self.ax2.set_title('Train Gradient Flow & Optimizer Trajectory')
        
        # 3. Val 3D
        self.surf_val = self.ax3.plot_surface(X, Y, val_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')
        self.line_val3d, = self.ax3.plot([], [], [], color='black', linewidth=2, marker='o', markersize=3)
        self.ax3.set_title(f'Val Loss Landscape ({Config.DATASET})')
        
        # 4. Val 2D
        self.ax4.contourf(X, Y, val_Z, levels=50, cmap=Config.COLORMAP, alpha=0.8)
        self.ax4.quiver(X[::step, ::step], Y[::step, ::step], -val_grad_x[::step, ::step], -val_grad_y[::step, ::step], color='white', alpha=0.5)
        self.line_val2d, = self.ax4.plot([], [], color='black', linewidth=2, marker='o', markersize=3)
        self.ax4.set_title('Val Gradient Flow & Optimizer Trajectory')
        
        self.traj_x = []
        self.traj_y = []
        
    def update(self, x, y):
        self.traj_x.append(x)
        self.traj_y.append(y)
        
        # Interpolate Z for current point
        points = np.column_stack((self.X.flatten(), self.Y.flatten()))
        tz = griddata(points, self.train_Z.flatten(), (x, y), method='linear')
        vz = griddata(points, self.val_Z.flatten(), (x, y), method='linear')
        
        if np.isnan(tz): tz = np.nanmean(self.train_Z)
        if np.isnan(vz): vz = np.nanmean(self.val_Z)
        
        # Append to Z trajectories (we must maintain full Z history for 3D plot.set_3d_properties)
        if not hasattr(self, 'train_z_history'):
            self.train_z_history = []
            self.val_z_history = []
        self.train_z_history.append(tz)
        self.val_z_history.append(vz)
        
        self.line_train3d.set_data(self.traj_x, self.traj_y)
        self.line_train3d.set_3d_properties(self.train_z_history)
        
        self.line_train2d.set_data(self.traj_x, self.traj_y)
        
        self.line_val3d.set_data(self.traj_x, self.traj_y)
        self.line_val3d.set_3d_properties(self.val_z_history)
        
        self.line_val2d.set_data(self.traj_x, self.traj_y)
        
        plt.draw()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close(self.fig)

def plot_landscapes(X, Y, train_Z, val_Z, traj_x, traj_y, grad_x, grad_y, val_grad_x, val_grad_y, traj_grad_x, traj_grad_y):
    """
    Renders the STAM visualizations.
    """
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    
    points = np.column_stack((X.flatten(), Y.flatten()))
    train_traj_z = griddata(points, train_Z.flatten(), (traj_x, traj_y), method='linear')
    val_traj_z = griddata(points, val_Z.flatten(), (traj_x, traj_y), method='linear')
    
    train_traj_z = np.nan_to_num(train_traj_z, nan=np.nanmean(train_Z))
    val_traj_z = np.nan_to_num(val_traj_z, nan=np.nanmean(val_Z))

    # --- 1. Static Plot ---
    print("Generating STAM static plot...")
    fig = plt.figure(figsize=(20, 20))
    
    # Train 3D
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X, Y, train_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')
    ax1.plot(traj_x, traj_y, train_traj_z, color='black', linewidth=2, marker='o', markersize=3)
    ax1.set_title(f'Train Loss Landscape ({Config.DATASET})')
    
    # Train 2D
    ax2 = fig.add_subplot(222)
    ax2.contourf(X, Y, train_Z, levels=50, cmap=Config.COLORMAP, alpha=0.8)
    step = 4
    ax2.quiver(X[::step, ::step], Y[::step, ::step], -grad_x[::step, ::step], -grad_y[::step, ::step], color='white', alpha=0.5)
    ax2.plot(traj_x, traj_y, color='black', linewidth=2, marker='o', markersize=3)
    ax2.set_title('Train Gradient Flow & Optimizer Trajectory')

    # Val 3D
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_surface(X, Y, val_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')
    ax3.plot(traj_x, traj_y, val_traj_z, color='black', linewidth=2, marker='o', markersize=3)
    ax3.set_title(f'Val Loss Landscape ({Config.DATASET})')
    
    # Val 2D
    ax4 = fig.add_subplot(224)
    ax4.contourf(X, Y, val_Z, levels=50, cmap=Config.COLORMAP, alpha=0.8)
    ax4.quiver(X[::step, ::step], Y[::step, ::step], -val_grad_x[::step, ::step], -val_grad_y[::step, ::step], color='white', alpha=0.5)
    ax4.plot(traj_x, traj_y, color='black', linewidth=2, marker='o', markersize=3)
    ax4.set_title('Val Gradient Flow & Optimizer Trajectory')
    
    save_path_static = os.path.join(Config.PLOT_DIR, f'stam_landscape_{Config.DATASET}.png')
    plt.savefig(save_path_static, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static visualization to {save_path_static}")

    # --- 2. Animated GIF ---
    print("Generating animated GIF...")
    fig_anim = plt.figure(figsize=(20, 20))
    
    ax1_anim = fig_anim.add_subplot(221, projection='3d')
    ax2_anim = fig_anim.add_subplot(222)
    ax3_anim = fig_anim.add_subplot(223, projection='3d')
    ax4_anim = fig_anim.add_subplot(224)
    
    # Initial setups
    surf_train = [ax1_anim.plot_surface(X, Y, train_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')]
    line_train3d, = ax1_anim.plot([], [], [], color='black', linewidth=2, marker='o', markersize=4)
    ax1_anim.set_xlim(X.min(), X.max()); ax1_anim.set_ylim(Y.min(), Y.max()); ax1_anim.set_zlim(train_Z.min(), train_Z.max())
    
    ax2_anim.contourf(X, Y, train_Z, levels=50, cmap=Config.COLORMAP, alpha=0.8)
    ax2_anim.quiver(X[::step, ::step], Y[::step, ::step], -grad_x[::step, ::step], -grad_y[::step, ::step], color='white', alpha=0.5)
    line_train2d, = ax2_anim.plot([], [], color='black', linewidth=2, marker='o', markersize=4)
    
    surf_val = [ax3_anim.plot_surface(X, Y, val_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')]
    line_val3d, = ax3_anim.plot([], [], [], color='black', linewidth=2, marker='o', markersize=4)
    ax3_anim.set_xlim(X.min(), X.max()); ax3_anim.set_ylim(Y.min(), Y.max()); ax3_anim.set_zlim(val_Z.min(), val_Z.max())
    
    ax4_anim.contourf(X, Y, val_Z, levels=50, cmap=Config.COLORMAP, alpha=0.8)
    ax4_anim.quiver(X[::step, ::step], Y[::step, ::step], -val_grad_x[::step, ::step], -val_grad_y[::step, ::step], color='white', alpha=0.5)
    line_val2d, = ax4_anim.plot([], [], color='black', linewidth=2, marker='o', markersize=4)
    
    def update_plot(frame):
        xt = traj_x[frame]
        yt = traj_y[frame]
        
        idx_x = (np.abs(X[0] - xt)).argmin()
        idx_y = (np.abs(Y[:, 0] - yt)).argmin()
        g_global_x = grad_x[idx_y, idx_x]
        g_global_y = grad_y[idx_y, idx_x]
        
        noise_x = traj_grad_x[frame] - g_global_x
        noise_y = traj_grad_y[frame] - g_global_y
        
        tilt = noise_x * (X - xt) + noise_y * (Y - yt)
        dampening = np.exp(-((X - xt)**2 + (Y - yt)**2) / 1.0)
        
        warped_Z = train_Z + tilt * dampening
        
        surf_train[0].remove()
        surf_train[0] = ax1_anim.plot_surface(X, Y, warped_Z, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')
        
        # We also warp val for consistency in animation (optional, but looks better)
        warped_vZ = val_Z + tilt * dampening
        surf_val[0].remove()
        surf_val[0] = ax3_anim.plot_surface(X, Y, warped_vZ, cmap=Config.COLORMAP, alpha=0.8, edgecolor='none')
        
        line_train3d.set_data(traj_x[:frame+1], traj_y[:frame+1])
        line_train3d.set_3d_properties(train_traj_z[:frame+1])
        
        line_train2d.set_data(traj_x[:frame+1], traj_y[:frame+1])
        
        line_val3d.set_data(traj_x[:frame+1], traj_y[:frame+1])
        line_val3d.set_3d_properties(val_traj_z[:frame+1])
        
        line_val2d.set_data(traj_x[:frame+1], traj_y[:frame+1])
        
        ax1_anim.view_init(elev=30., azim=frame * 2)
        ax3_anim.view_init(elev=30., azim=frame * 2)
        
        return line_train3d, surf_train[0], line_train2d, line_val3d, surf_val[0], line_val2d

    anim = FuncAnimation(fig_anim, update_plot, frames=len(traj_x), interval=100, blit=False)
    save_path_gif = os.path.join(Config.PLOT_DIR, f'stam_landscape_{Config.DATASET}.gif')
    anim.save(save_path_gif, writer='pillow', fps=10)
    plt.close()
    print(f"Saved animated GIF to {save_path_gif}")
