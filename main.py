import torch
import os
from src.config import Config
from src.data_loader import get_dataloaders
from src.models import get_model
from src.train import train_model_generator
from src.landscape import compute_basis, project_trajectory, evaluate_landscape
from src.plotter import plot_landscapes, LivePlotter

def main():
    print(f"Using device: {Config.DEVICE}")
    print(f"Dataset: {Config.DATASET}")
    
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    
    trajectory = []
    trajectory_grads = []
    
    gen = train_model_generator(model, train_loader, val_loader)
    
    basis_computed = False
    theta_0, v1, v2 = None, None, None
    live_plotter = None
    
    for is_epoch_end, epoch, batch_idx, params, grads, metrics in gen:
        # Always record trajectory and grads for every yield
        trajectory.append(params)
        trajectory_grads.append(grads)
        
        # Live update (only after burn-in basis is ready)
        if basis_computed and Config.LIVE_TRAINING and live_plotter:
            x, y = project_trajectory(torch.stack([params]), theta_0, v1, v2)
            live_plotter.update(x[0], y[0])
                
        # After first epoch, compute burn-in basis for live tracking
        if is_epoch_end and epoch == 0:
            print("Burn-in phase complete. Computing STAM basis for live tracking...")
            traj_tensor = torch.stack(trajectory)
            theta_0, v1, v2 = compute_basis(traj_tensor)
            
            traj_x, traj_y = project_trajectory(traj_tensor, theta_0, v1, v2)
            margin = Config.GRID_MARGIN
            # Use a large margin for the live preview (will be recomputed for final output)
            burn_in_range_x = traj_x.max() - traj_x.min()
            burn_in_range_y = traj_y.max() - traj_y.min()
            x_min = traj_x.min() - max(burn_in_range_x * 2, margin)
            x_max = traj_x.max() + max(burn_in_range_x * 5, margin)
            y_min = traj_y.min() - max(burn_in_range_y * 5, margin)
            y_max = traj_y.max() + max(burn_in_range_y * 5, margin)
            coords_x = (x_min, x_max)
            coords_y = (y_min, y_max)
            
            X_live, Y_live, train_Z_live, val_Z_live, grad_x_live, grad_y_live, val_grad_x_live, val_grad_y_live, _ = evaluate_landscape(
                model, train_loader, val_loader, theta_0, v1, v2, coords_x, coords_y
            )
            
            basis_computed = True
            
            if Config.LIVE_TRAINING:
                live_plotter = LivePlotter(X_live, Y_live, train_Z_live, val_Z_live, grad_x_live, grad_y_live, val_grad_x_live, val_grad_y_live)
                # Plot the burn-in trajectory quickly
                for x, y in zip(traj_x, traj_y):
                    live_plotter.update(x, y)

    if live_plotter:
        live_plotter.close()
        
    # === FINAL OUTPUT: Recompute basis and landscape using the FULL trajectory ===
    print("Training finished. Recomputing STAM basis on full trajectory...")
    full_traj = torch.stack(trajectory)
    full_grads = torch.stack(trajectory_grads)
    
    # Recompute SVD on the complete trajectory for maximum accuracy
    theta_0, v1, v2 = compute_basis(full_traj)
    traj_x, traj_y = project_trajectory(full_traj, theta_0, v1, v2)
    
    # Set grid bounds from the FULL trajectory
    margin = Config.GRID_MARGIN
    x_min, x_max = traj_x.min() - margin, traj_x.max() + margin
    y_min, y_max = traj_y.min() - margin, traj_y.max() + margin
    coords_x = (x_min, x_max)
    coords_y = (y_min, y_max)
    
    print("Evaluating final landscape anchors on full-trajectory basis...")
    X, Y, train_Z, val_Z, grad_x, grad_y, val_grad_x, val_grad_y, _ = evaluate_landscape(
        model, train_loader, val_loader, theta_0, v1, v2, coords_x, coords_y
    )
    
    traj_grad_x = torch.matmul(full_grads.to(torch.float32).to(Config.DEVICE), v1).cpu().numpy()
    traj_grad_y = torch.matmul(full_grads.to(torch.float32).to(Config.DEVICE), v2).cpu().numpy()
    
    print("Generating final GIF/PNG...")
    plot_landscapes(X, Y, train_Z, val_Z, traj_x, traj_y, grad_x, grad_y, val_grad_x, val_grad_y, traj_grad_x, traj_grad_y)
    
    print("STAM Pipeline Completed Successfully.")

if __name__ == '__main__':
    main()
