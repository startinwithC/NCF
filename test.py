import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("./models/NeuMF-end.pth")

print(model)

user = torch.tensor(5623)
item = torch.tensor(352)
user, item = user.to(device), item.to(device)
model.eval()

with torch.no_grad():
    output = model(user, item)
    output = output.item()
print(output)