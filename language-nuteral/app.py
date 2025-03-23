import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from indicnlp.tokenize import indic_tokenize

# Define model hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 4
DROPOUT = 0.1
HEADS = 8

# Expanded dataset (at least 10 examples)
dataset = [
    ("Hello, how are you?", "नमस्ते, आप कैसे हैं?"),
    ("Good morning!", "सुप्रभात!"),
    ("What is your name?", "आपका नाम क्या है?"),
    ("I am fine.", "मैं ठीक हूँ।"),
    ("Where do you live?", "आप कहाँ रहते हैं?"),
    ("Have a great day!", "आपका दिन शुभ हो!"),
    ("Nice to meet you.", "आपसे मिलकर अच्छा लगा।"),
    ("Thank you.", "धन्यवाद।"),
    ("See you later.", "फिर मिलते हैं।"),
    ("Welcome!", "स्वागत है!"),
    ("I love learning new languages.", "मुझे नई भाषाएँ सीखना पसंद है।"),
    ("The weather is nice today.", "आज मौसम अच्छा है।"),
    ("Can you help me?", "क्या आप मेरी मदद कर सकते हैं?"),
    ("I am very happy.", "मैं बहुत खुश हूँ।"),
    ("Let's go to the market.", "चलो बाजार चलते हैं।"),
]

# Tokenization
tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")

# ✅ Updated Hindi Tokenizer (Using IndicNLP)
def hindi_tokenizer(text):
    return list(indic_tokenize.trivial_tokenize(text, lang='hi'))

tokenizer_hi = hindi_tokenizer

# Build Vocabulary
def yield_tokens(data, tokenizer):
    for text, _ in data:
        yield tokenizer(text)

vocab_en = build_vocab_from_iterator(yield_tokens(dataset, tokenizer_en), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab_hi = build_vocab_from_iterator(yield_tokens(dataset, tokenizer_hi), specials=["<unk>", "<pad>", "<sos>", "<eos>"])

# Set unknown token index
vocab_en.set_default_index(vocab_en["<unk>"])
vocab_hi.set_default_index(vocab_hi["<unk>"])

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, num_layers, heads, dropout):
        super(Transformer, self).__init__()

        self.embedding_src = nn.Embedding(input_dim, embed_size)
        self.embedding_tgt = nn.Embedding(output_dim, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dropout=dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=heads, dropout=dropout)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, output_dim)

    def forward(self, src, trg):
        src = self.embedding_src(src)
        trg = self.embedding_tgt(trg)

        src = self.pos_encoding(src)
        trg = self.pos_encoding(trg)

        enc_out = self.encoder(src)
        out = self.decoder(trg, enc_out)
        return self.fc_out(out)

# Model Instantiation
model = Transformer(input_dim=len(vocab_en), output_dim=len(vocab_hi),
                    embed_size=EMBED_SIZE, num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT)

# Training Loop
def train_model(model, data, epochs=300):
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_hi["<pad>"])  # Ignore padding in loss
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        total_loss = 0
        for eng, hin in data:
            eng_tokens = [vocab_en[token] for token in tokenizer_en(eng)]
            hin_tokens = [vocab_hi["<sos>"]] + [vocab_hi[token] for token in tokenizer_hi(hin)] + [vocab_hi["<eos>"]]

            eng_tokens = torch.tensor(eng_tokens, dtype=torch.long).unsqueeze(1)
            hin_tokens = torch.tensor(hin_tokens, dtype=torch.long).unsqueeze(1)

            optimizer.zero_grad()
            output = model(eng_tokens, hin_tokens[:-1])  # Remove last token from target

            loss = criterion(output.view(-1, len(vocab_hi)), hin_tokens[1:].view(-1))  # Shift target for loss calc
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(data):.4f}")

# Train the Model
train_model(model, dataset, epochs=300)

# Save the trained model
torch.save(model.state_dict(), "translation_model.pth")
