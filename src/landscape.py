import torch
import torch.nn as nn
import numpy as np
from .config import Config
from scipy.interpolate import Rbf

def compute_basis(trajectory):
    """
    Computes an orthonormal basis for the parameter trajectory using SVD.
    This anchors the 2D projection plane to the true maximum variance of the optimizer's path.
    """
    theta_0 = trajectory[0]
    D = trajectory - theta_0
    D = D.to(torch.float32)
    print(f"Computing SVD on displacement matrix of shape {D.shape}...")
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    v1 = Vh[0]
    v2 = Vh[1]
    return theta_0, v1, v2

def project_trajectory(trajectory, theta_0, v1, v2):
    """
    Projects the high-dimensional trajectory onto the STAM 2D basis.
    """
    D = trajectory - theta_0
    x = torch.matmul(D, v1)
    y = torch.matmul(D, v2)
    return x.cpu().numpy(), y.cpu().numpy()

def evaluate_landscape(model, train_loader, val_loader, theta_0, v1, v2, coords_x, coords_y):
    """
    Evaluates the training and validation loss landscapes, as well as the gradient 
    vector field over a sparse anchor grid. Uses RBF interpolation to upsample 
    the result to a high-resolution render grid.
    """
    sparse_size = Config.SPARSE_GRID_RESOLUTION
    render_size = Config.RENDER_RESOLUTION
    
    sparse_x = np.linspace(coords_x[0], coords_x[1], sparse_size)
    sparse_y = np.linspace(coords_y[0], coords_y[1], sparse_size)
    SX, SY = np.meshgrid(sparse_x, sparse_y)
    
    sparse_train_Z = np.zeros((sparse_size, sparse_size))
    sparse_val_Z = np.zeros((sparse_size, sparse_size))
    sparse_grad_x = np.zeros((sparse_size, sparse_size))
    sparse_grad_y = np.zeros((sparse_size, sparse_size))
    sparse_val_grad_x = np.zeros((sparse_size, sparse_size))
    sparse_val_grad_y = np.zeros((sparse_size, sparse_size))
    
    criterion = nn.CrossEntropyLoss()
    print(f"Evaluating STAM landscape anchors ({sparse_size}x{sparse_size})...")
    total_points = sparse_size * sparse_size
    
    num_eval_batches = 20
    train_batches = []
    val_batches = []
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    for _ in range(num_eval_batches):
        try: train_batches.append(next(train_iter))
        except StopIteration: break
            
    for _ in range(num_eval_batches):
        try: val_batches.append(next(val_iter))
        except StopIteration: break
            
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    point_idx = 0
    for i in range(sparse_size):
        for j in range(sparse_size):
            alpha = sparse_x[j]
            beta = sparse_y[i]
            
            theta_grid = theta_0 + alpha * v1 + beta * v2
            theta_grid = theta_grid.to(Config.DEVICE)
            
            offset = 0
            state_dict = model.state_dict()
            for name, param in model.named_parameters():
                numel = param.numel()
                param_data = theta_grid[offset:offset+numel].view(param.shape)
                state_dict[name].copy_(param_data)
                offset += numel
                
            model.load_state_dict(state_dict)
            model.train()
            
            t_loss = 0
            model.zero_grad()
            for inputs, labels in train_batches:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                t_loss += loss.item()
            sparse_train_Z[i, j] = t_loss / len(train_batches)
            
            # Extract high-dimensional gradient and project onto STAM basis
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1))
                else:
                    grad_vec.append(torch.zeros_like(param).view(-1))
            grad_vec = torch.cat(grad_vec)
            grad_vec = grad_vec / len(train_batches)
            
            sparse_grad_x[i, j] = torch.dot(grad_vec, v1.to(Config.DEVICE)).item()
            sparse_grad_y[i, j] = torch.dot(grad_vec, v2.to(Config.DEVICE)).item()
            
            # Now compute Validation gradients (requires train mode or manual backward)
            v_loss = 0
            model.zero_grad()
            for inputs, labels in val_batches:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                v_loss += loss.item()
            sparse_val_Z[i, j] = v_loss / len(val_batches)
            
            val_grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    val_grad_vec.append(param.grad.view(-1))
                else:
                    val_grad_vec.append(torch.zeros_like(param).view(-1))
            val_grad_vec = torch.cat(val_grad_vec)
            val_grad_vec = val_grad_vec / len(val_batches)
            
            sparse_val_grad_x[i, j] = torch.dot(val_grad_vec, v1.to(Config.DEVICE)).item()
            sparse_val_grad_y[i, j] = torch.dot(val_grad_vec, v2.to(Config.DEVICE)).item()
            
            point_idx += 1
            if point_idx % 10 == 0:
                print(f"Evaluated {point_idx}/{total_points} anchors")
                
    model.load_state_dict(original_state)
    
    # RBF Mathematical Upsampling
    print(f"Applying Radial Basis Function (RBF) upsampling to {render_size}x{render_size}...")
    sx_flat = SX.flatten()
    sy_flat = SY.flatten()
    
    rbf_train = Rbf(sx_flat, sy_flat, sparse_train_Z.flatten(), function='multiquadric', smooth=0.1)
    rbf_val = Rbf(sx_flat, sy_flat, sparse_val_Z.flatten(), function='multiquadric', smooth=0.1)
    rbf_gx = Rbf(sx_flat, sy_flat, sparse_grad_x.flatten(), function='linear')
    rbf_gy = Rbf(sx_flat, sy_flat, sparse_grad_y.flatten(), function='linear')
    rbf_vgx = Rbf(sx_flat, sy_flat, sparse_val_grad_x.flatten(), function='linear')
    rbf_vgy = Rbf(sx_flat, sy_flat, sparse_val_grad_y.flatten(), function='linear')
    
    render_x = np.linspace(coords_x[0], coords_x[1], render_size)
    render_y = np.linspace(coords_y[0], coords_y[1], render_size)
    X, Y = np.meshgrid(render_x, render_y)
    
    train_Z = rbf_train(X, Y)
    val_Z = rbf_val(X, Y)
    grad_x = rbf_gx(X, Y)
    grad_y = rbf_gy(X, Y)
    val_grad_x = rbf_vgx(X, Y)
    val_grad_y = rbf_vgy(X, Y)
    
    global_grad_magnitude = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    
    return X, Y, train_Z, val_Z, grad_x, grad_y, val_grad_x, val_grad_y, global_grad_magnitude
