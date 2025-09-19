# import torch.nn
import torchvision.models as models
from torch import nn
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
# from data_loader import dataloader
# from generating_embeddings import generating_embeddings


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)
    
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z 

# resnet18 = models.resnet18()

# backbone = nn.Sequential(*list(resnet18.children())[:-1])
# model = SimCLR(backbone)

# criterion = NTXentLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# for epoch in range(10):
#     total_loss = 0
#     for batch in dataloader:
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
#     avg_loss = total_loss / len(dataloader)
#     print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

# # model.eval()
# # embeddings, filenames = generating_embeddings(model, dataloader)


