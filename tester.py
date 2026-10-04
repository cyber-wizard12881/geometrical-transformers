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
print("🔧 Loading geometry models")
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
    print(f"   ✅ {label:<12} ← {checkpoint_path}")
print(f"🖥️  Device: {device}\n")


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
    print("🧪 Evaluating test sentences\n" + "=" * 72)
    for _, row in test_df.iterrows():
        sentence = str(row["sentence"])
        actual_geometry = row["geometry_type"]
        predicted_geometry, losses = classify_sentence(sentence)
        is_correct = predicted_geometry == actual_geometry
        correct_predictions += is_correct
        result_icon = "✅" if is_correct else "❌"

        print(f"\n📝 {sentence}")
        print(f"   Expected:  {actual_geometry}")
        print(f"   Predicted: {predicted_geometry} {result_icon}")
        print("   📊 Model losses:")
        for label, loss in sorted(losses.items(), key=lambda item: item[1]):
            print(f"      {label:<12} {loss:.4f}")

    accuracy = correct_predictions / len(test_df)
    print("\n" + "=" * 72)
    print(
        f"🏁 Accuracy: {correct_predictions}/{len(test_df)} "
        f"({accuracy:.1%})"
    )
