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


    transform = SimCLRTransform(input_size=32, gaussian_blur=0.0)

    # dataset = LightlyDataset(r"\Users\Besitzer\Desktop\Image_Dataset", transform=transform)
    dataset = LightlyDataset(r"\Users\Besitzer\Desktop\Image_Dataset", transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=True,
        drop_last=True, num_workers=8,  
            persistent_workers=True 
    )



    resnet18 = models.resnet18()

    backbone = nn.Sequential(*list(resnet18.children())[:-1])
    model = SimCLR(backbone)

    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    best_loss = float('inf')

    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        # --- SAVE THE BEST MODEL ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'saved_models/best_model.pth')
            print(f"--> New best model saved with loss: {best_loss:.5f}")
    # --- SAVE THE FINAL MODEL ---
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, 'saved_models/final_model.pth')
    print("Final model saved.")