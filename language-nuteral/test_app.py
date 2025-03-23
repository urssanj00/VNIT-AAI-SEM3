import torch
from app import model, tokenizer_en, vocab_en, vocab_hi

# Load trained model
model.load_state_dict(torch.load("translation_model.pth"))
model.eval()


# Function to convert tokens to text
def tokens_to_text(tokens, vocab):
    return " ".join([vocab.lookup_token(token) for token in tokens if
                     token not in {vocab["<sos>"], vocab["<eos>"], vocab["<pad>"], vocab["<unk>"]}])


# Function to test translation with greedy decoding
def test_translation(sentence, max_length=20):
    model.eval()

    eng_tokens = [vocab_en[token] for token in tokenizer_en(sentence)]
    eng_tokens = torch.tensor(eng_tokens, dtype=torch.long).unsqueeze(1)

    trg_tokens = [vocab_hi["<sos>"]]

    with torch.no_grad():
        for _ in range(max_length):
            trg_tensor = torch.tensor(trg_tokens, dtype=torch.long).unsqueeze(1)
            output_tokens = model(eng_tokens, trg_tensor)
            next_token = output_tokens.argmax(dim=-1).squeeze(1)[-1].item()

            if next_token == vocab_hi["<eos>"]:
                break

            trg_tokens.append(next_token)

    translated_text = tokens_to_text(trg_tokens[1:], vocab_hi)
    return translated_text if translated_text.strip() else "[Translation Failed]"


# Test Cases
test_sentences = [
    "Hello, how are you?",
    "Good morning!",
    "What is your name?",
    "I am fine."
]

for sentence in test_sentences:
    translation = test_translation(sentence)
    print(f"English: {sentence}")
    print(f"Translated Hindi: {translation}\n")
