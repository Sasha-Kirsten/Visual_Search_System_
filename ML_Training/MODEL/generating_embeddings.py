import torch
import torchvision

def generating_embeddings(model, dataloader):
    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)
    embeddings = torch.cat(embeddings, 0)
    # embeddings = normalize(embeddings)
    embeddings = torchvision.transforms.Normalize(embeddings)
