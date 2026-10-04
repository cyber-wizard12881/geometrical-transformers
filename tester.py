import torch
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer
from transformer import Transformer as GeometricTransformer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = {
    "Euclidean": ("euclidean", "euclidean_transformer.pt"),
    "Hyperbolic": ("hyperbolic", "hyperbolic_transformer.pt"),
    "Elliptic": ("spherical", "spherical_transformer.pt"),
    "Exponential": ("exponential", "exponential_transformer.pt"),
    "Parabolic": ("parabolic", "parabolic_transformer.pt"),
}

models = {}
for label, (geometry_type, checkpoint_path) in model_configs.items():
    model = GeometricTransformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        geometry_type=geometry_type,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    models[label] = model


def classify_sentence(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    decoder_input_ids = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    tgt_mask = torch.tril(
        torch.ones(
            decoder_input_ids.size(1),
            decoder_input_ids.size(1),
            device=device,
            dtype=torch.bool,
        )
    )

    losses = {}
    with torch.no_grad():
        for label, model in models.items():
            logits = model(input_ids[:, :1], decoder_input_ids, tgt_mask=tgt_mask)
            losses[label] = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
            ).item()

    return min(losses, key=losses.get), losses


if __name__ == "__main__":
    test_df = pd.read_csv("geometry_test_sentences.csv")
    correct_predictions = 0
    for _, row in test_df.iterrows():
        sentence = str(row["sentence"])
        actual_geometry = row["geometry_type"]
        predicted_geometry, losses = classify_sentence(sentence)
        correct_predictions += predicted_geometry == actual_geometry
        print(f"Sentence: {sentence}")
        print(f"Actual Geometry: {actual_geometry}")
        print(f"Predicted Geometry: {predicted_geometry}")
        print("Model losses:", {label: round(loss, 4) for label, loss in losses.items()})
        print("-" * 50)

    print(
        f"Accuracy: {correct_predictions}/{len(test_df)} "
        f"({correct_predictions / len(test_df):.1%})"
    )
