import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from transformer import Transformer as GeometricTransformer

# -------------------------------
# 1. Load Dataset
# -------------------------------
class GeometryDataset(Dataset):
    def __init__(self, csv_file, tokenizer, geometry_label, max_len=32):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[
            self.data["geometry_type"].str.casefold() == geometry_label.casefold()
        ]
        if self.data.empty:
            raise ValueError(f"No training sentences found for {geometry_label!r}.")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = str(self.data.iloc[idx]["sentence"])
        encoding = self.tokenizer(sentence,
                                  padding="max_length",
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        return input_ids

# -------------------------------
# 2. Simple Transformer Model
# -------------------------------
# We will use the Geometric Transformer (Transformer) model in transformer.py.

# -------------------------------
# 3. Training Setup
# -------------------------------
def train_model(
    csv_file,
    epochs=5,
    batch_size=16,
    lr=1e-4,
    geometry_type="euclidean",
    geometry_label=None,
):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if geometry_label is None:
        geometry_label = {
            "euclidean": "Euclidean",
            "hyperbolic": "Hyperbolic",
            "spherical": "Elliptic",
            "exponential": "Exponential",
            "parabolic": "Parabolic",
        }.get(geometry_type)
    if geometry_label is None:
        raise ValueError(
            f"No training label configured for geometry type {geometry_type!r}."
        )

    dataset = GeometryDataset(csv_file, tokenizer, geometry_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GeometricTransformer(src_vocab_size=tokenizer.vocab_size, tgt_vocab_size=tokenizer.vocab_size, geometry_type=geometry_type)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids in dataloader:
            input_ids = input_ids.to(device)

            optimizer.zero_grad()
            decoder_input_ids = input_ids[:, :-1]
            targets = input_ids[:, 1:].contiguous()
            tgt_mask = torch.tril(
                torch.ones(
                    decoder_input_ids.size(1),
                    decoder_input_ids.size(1),
                    device=device,
                    dtype=torch.bool,
                )
            )
            outputs = model(input_ids[:, :1], decoder_input_ids, tgt_mask=tgt_mask)

            loss = criterion(
                outputs.reshape(-1, tokenizer.vocab_size), targets.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # best practice
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model

# -------------------------------
# 5. Run Training
# -------------------------------
def train(train_model):
    euclidean_model = train_model("geometry_sentences.csv", epochs=5, batch_size=16, lr=5e-5, geometry_type="euclidean")
    hyperbolic_model = train_model("geometry_sentences.csv", epochs=5, batch_size=16, lr=5e-5, geometry_type="hyperbolic")
    spherical_model = train_model("geometry_sentences.csv", epochs=5, batch_size=16, lr=5e-5, geometry_type="spherical")
    exponential_model = train_model("geometry_sentences.csv", epochs=5, batch_size=16, lr=5e-5, geometry_type="exponential")
    parabolic_model = train_model("geometry_sentences.csv", epochs=5, batch_size=16, lr=5e-5, geometry_type="parabolic")

    torch.save(euclidean_model.state_dict(), "euclidean_transformer.pt")
    torch.save(hyperbolic_model.state_dict(), "hyperbolic_transformer.pt")
    torch.save(spherical_model.state_dict(), "spherical_transformer.pt")
    torch.save(exponential_model.state_dict(), "exponential_transformer.pt")
    torch.save(parabolic_model.state_dict(), "parabolic_transformer.pt")

    return euclidean_model, hyperbolic_model, spherical_model, exponential_model, parabolic_model

if __name__ == "__main__":
    train(train_model)