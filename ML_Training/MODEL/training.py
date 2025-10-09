from model import SimCLR
from torch import nn
import torchvision.models as models
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
# from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
import torch
# from data_loader import dataloader
# from generating_embeddings import generating_embeddings

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    import os # <- Add this import
    os.makedirs('saved_models', exist_ok=True) # <- Create save directory

    transform = SimCLRTransform(input_size=112, gaussian_blur=0.2)

    # Now point to the split folders you created!
    train_dataset = LightlyDataset(r"C:\Users\Besitzer\Desktop\Image_Dataset_Split\train", transform=transform)
    val_dataset = LightlyDataset(r"C:\Users\Besitzer\Desktop\Image_Dataset_Split\val", transform=transform)


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        drop_last=True, num_workers=8, persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, # No need to shuffle validation
        drop_last=False, num_workers=8, persistent_workers=True
    )

    resnet18 = models.resnet18()

    backbone = nn.Sequential(*list(resnet18.children())[:-1])
    optimal_out_dim = 256 
    model = SimCLR(backbone)

    optimal_temperature = 0.05316727044555556
    criterion = NTXentLoss(temperature=optimal_temperature)

    optimal_lr = 0.014720042723649098
    optimal_weight_decay = 8.471244577031029e-06
    optimal_momentum = 0.9214322894499637

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=optimal_lr,
        momentum=optimal_momentum,
        weight_decay=optimal_weight_decay
    )

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    best_val_loss = float('inf')
    num_epochs = 100

    for epoch in range(100):
        model.train()
        total_train_loss = 0
        
        for batch in train_dataloader:
            x0, x1 = batch[0]
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use mixed precision for faster training
            with torch.amp.autocast('cuda'):
                z0 = model(x0)
                z1 = model(x1)
                loss = criterion(z0, z1)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.detach().item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # ========== VALIDATION ==========
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                x0, x1 = batch[0]
                x0 = x0.to(device, non_blocking=True)
                x1 = x1.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    z0 = model(x0)
                    z1 = model(x1)
                    loss = criterion(z0, z1)
                
                total_val_loss += loss.detach().item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch:>02}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.5f}, "
              f"Val Loss: {avg_val_loss:.5f}")

        # ========== SAVE BEST MODEL ==========
        # Save based on validation loss (better indicator of generalization)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss = avg_train_loss
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': best_loss,
                'val_loss': best_val_loss,
                'config': {
                    'input_size': 224,
                    'batch_size': 512,
                    'lr': optimal_lr,
                    'temperature': optimal_temperature,
                    'out_dim': optimal_out_dim,
                    'weight_decay': optimal_weight_decay,
                    'momentum': optimal_momentum
                }
            }, 'saved_models/best_model.pth')
            
            print(f"--> New best model saved! Val Loss: {best_val_loss:.5f}")

    # ========== SAVE FINAL MODEL ==========
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'config': {
            'input_size': 224,
            'batch_size': 512,
            'lr': optimal_lr,
            'temperature': optimal_temperature,
            'out_dim': optimal_out_dim,
            'weight_decay': optimal_weight_decay,
            'momentum': optimal_momentum
        }
    }, 'saved_models/final_model.pth')
    
    print("Training completed!")
    print(f"Final Train Loss: {avg_train_loss:.5f}")
    print(f"Final Val Loss: {avg_val_loss:.5f}")
    print(f"Best Val Loss: {best_val_loss:.5f}")

    #     model.train()
    #     total_loss = 0
    #     for batch in train_dataloader:
    #         x0, x1 = batch[0]
    #         x0 = x0.to(device)
    #         x1 = x1.to(device)

    #         z0 = model(x0)
    #         z1 = model(x1)
    #         loss = criterion(z0, z1)
    #         total_loss += loss.detach()
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     avg_train_loss  = total_loss / len(train_dataloader)
    #     print(f"epoch: {epoch:>02}, loss: {avg_train_loss :.5f}")

    #     # --- VALIDATION PHASE ---
    #     model.eval()
    #     total_val_loss = 0
    #     with torch.no_grad():  # Disable gradients for validation!
    #         for batch in val_dataloader:
    #             x0, x1 = batch[0]
    #             x0, x1 = x0.to(device), x1.to(device)
    #             z0 = model(x0)
    #             z1 = model(x1)
    #             loss = criterion(z0, z1)
    #             total_val_loss += loss.detach()
    #     avg_val_loss = total_val_loss / len(val_dataloader)
    #     print(f"Epoch {epoch:>02} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    #     # --- SAVE THE BEST MODEL ---
    #     if avg_train_loss  < best_loss:
    #         best_loss = avg_train_loss 
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': best_loss,
    #         }, 'saved_models/best_model.pth')
    #         print(f"--> New best model saved with loss: {best_loss:.5f}")
    # # --- SAVE THE FINAL MODEL ---
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': avg_train_loss ,
    # }, 'saved_models/final_model.pth')
    # print("Final model saved.")