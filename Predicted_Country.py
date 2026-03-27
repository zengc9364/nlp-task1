import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
abc = abc + abc.lower() + "_"
N_LETTERS = len(abc)
MAX_NAME_LENGTH = 20
DATA_FOLDER = r"D:\VScodeProject\NLP\names"  

COUNTRY_LIST = [
    "Arabic", "Chinese", "Czech", "Dutch", "English",
    "French", "German", "Greek", "Irish", "Italian",
    "Japanese", "Korean", "Polish", "Portuguese", "Russian",
    "Scottish", "Spanish", "Vietnamese"
]

def letter_index(lett):
    if lett in abc:
        return abc.find(lett)
    return abc.find("_")

def word_to_tensor(word):
    tensor = torch.zeros(MAX_NAME_LENGTH, N_LETTERS)
    for li, letter in enumerate(word[:MAX_NAME_LENGTH]):
        tensor[li][letter_index(letter)] = 1
    return tensor

class NamesDataset(Dataset):
    def __init__(self, data_folder):
        self.names, self.label_indices = self._load_data(data_folder)
        self.num_countries = len(COUNTRY_LIST)
        self.countries = COUNTRY_LIST

    def _load_data(self, data_folder):
        all_names = []
        all_label_indices = []
        print("data index：")
        for country_idx, country_name in enumerate(COUNTRY_LIST):
            file_path = os.path.join(data_folder, f"{country_name}.txt")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                names = [line.strip() for line in f if line.strip()]
            all_names.extend(names)
            all_label_indices.extend([country_idx] * len(names))
            print(f"  [{country_idx}] {country_name}: {len(names)} names")
        return all_names, all_label_indices

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label_idx = self.label_indices[idx]
        name_tensor = word_to_tensor(name)
        country_tensor = torch.zeros(self.num_countries)
        country_tensor[label_idx] = 1
        return name_tensor, country_tensor

class NameClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.fc(out)
        return out

def train_model(model, train_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, y_onehot in train_loader:
            x, y_onehot = x.to(device), y_onehot.to(device)
            y = torch.argmax(y_onehot, dim=1)  

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(y_pred, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

def predict_name(model, name, dataset):
    model.eval()
    with torch.no_grad():
        x = word_to_tensor(name).unsqueeze(0).to(device)
        output = model(x)
        pred_idx = torch.argmax(output, dim=1).item()
    return dataset.countries[pred_idx]

if __name__ == '__main__':
    dataset = NamesDataset(DATA_FOLDER)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    print(f"\nDataset loaded: {len(dataset)} names | 18 countries")

    model = NameClassifier(
        input_size=N_LETTERS,
        hidden_size=256,
        num_classes=len(dataset.countries)
    ).to(device)

    print("\nStart training (20 epochs)")
    train_model(model, train_loader, epochs=20)

    print("\nPrediction Results:")
    test_names = ["Khoury", "Cui", "Smith", "Mohammed", "Ivanov", "Nguyen"]
    for name in test_names:
        result = predict_name(model, name, dataset)
        print(f"Name: {name:10} → Country: {result}")

#result
#data index：
#  [0] Arabic: 2000 names
#  [1] Chinese: 268 names
#  [2] Czech: 519 names
#  [3] Dutch: 297 names
#  [4] English: 3668 names
#  [5] French: 277 names
#  [6] German: 724 names
#  [7] Greek: 203 names
#  [8] Irish: 232 names
#  [9] Italian: 709 names
#  [10] Japanese: 991 names
#  [11] Korean: 94 names
#  [12] Polish: 139 names
#  [13] Portuguese: 74 names
#  [14] Russian: 9408 names
#  [15] Scottish: 100 names
#  [16] Spanish: 298 names
#  [17] Vietnamese: 73 names

#Dataset loaded: 20074 names | 18 countries

#Start training (20 epochs)
#Epoch  1 | Loss: 0.0126 | Accuracy: 0.5394
#Epoch  2 | Loss: 0.0092 | Accuracy: 0.6588
#Epoch  3 | Loss: 0.0081 | Accuracy: 0.6924
#Epoch  4 | Loss: 0.0074 | Accuracy: 0.7190
#Epoch  5 | Loss: 0.0069 | Accuracy: 0.7359
#Epoch  6 | Loss: 0.0064 | Accuracy: 0.7536
#Epoch  7 | Loss: 0.0060 | Accuracy: 0.7688
#Epoch  8 | Loss: 0.0057 | Accuracy: 0.7793
#Epoch  9 | Loss: 0.0053 | Accuracy: 0.7944
#Epoch 10 | Loss: 0.0051 | Accuracy: 0.7983
#Epoch 11 | Loss: 0.0049 | Accuracy: 0.8100
#Epoch 12 | Loss: 0.0046 | Accuracy: 0.8176
#Epoch 13 | Loss: 0.0044 | Accuracy: 0.8227
#Epoch 14 | Loss: 0.0042 | Accuracy: 0.8283
#Epoch 15 | Loss: 0.0041 | Accuracy: 0.8322
#Epoch 16 | Loss: 0.0039 | Accuracy: 0.8408
#Epoch 17 | Loss: 0.0038 | Accuracy: 0.8464
#Epoch 18 | Loss: 0.0037 | Accuracy: 0.8517
#Epoch 19 | Loss: 0.0035 | Accuracy: 0.8589
#Epoch 20 | Loss: 0.0034 | Accuracy: 0.8590

#Prediction Results:
#Name: Khoury     → Country: Arabic
#Name: Cui        → Country: Chinese
#Name: Smith      → Country: English
#Name: Mohammed   → Country: English
#Name: Ivanov     → Country: Russian
#Name: Nguyen     → Country: Vietnamese
