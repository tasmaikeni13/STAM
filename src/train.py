import torch
import torch.nn as nn
import torch.optim as optim
from .config import Config

def get_current_state(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1).clone())
    return torch.cat(params)

def get_current_grads(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.data.view(-1).clone())
        else:
            grads.append(torch.zeros_like(param).view(-1))
    return torch.cat(grads)

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(val_loader), 100. * correct / total

def train_model_generator(model, train_loader, val_loader):
    """
    Yields (is_epoch_end, epoch, batch_idx, params, grads, metrics)
    so the caller can perform live plotting.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Initial state
    yield False, 0, 0, get_current_state(model), get_current_grads(model), None
    
    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Yield every 50 batches for trajectory tracking
            if batch_idx % 50 == 0 and (epoch > 0 or batch_idx > 0):
                yield False, epoch, batch_idx, get_current_state(model), get_current_grads(model), None
                
        # End of epoch validation
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        metrics = {
            'train_loss': running_loss / len(train_loader),
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        print(f"Epoch {epoch+1}/{Config.EPOCHS}, Train Loss: {metrics['train_loss']:.4f}, "
              f"Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.2f}%")
        
        # Yield epoch end
        yield True, epoch, len(train_loader), get_current_state(model), get_current_grads(model), metrics
