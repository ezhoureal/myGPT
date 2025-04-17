import torch
import tiktoken
import tqdm
from model import Config, GPT

def inference(model: GPT, device: torch.device):
    BATCH = 5
    MAX_LEN = 30
    K = 50
    PROMPT = "Hello, LLM. What's your name?"

    enc = tiktoken.encoding_for_model('gpt2')
    tokens = enc.encode(PROMPT)
    x = torch.tensor(tokens, dtype=torch.long, device=device)  # Move tensor to device
    x = x.unsqueeze(0).repeat(BATCH, 1)  # (B, T)

    for i in tqdm.trange(MAX_LEN):
        logits, _ = model(x, None)
        logits = logits[:, -1, :].squeeze(1)
        assert logits.shape == (BATCH, Config.vocab_size)
        probs = torch.softmax(logits, dim=-1)
        (top_probs, top_idx) = torch.topk(probs, K)
        next_idx = torch.multinomial(top_probs, 1)  # sample 1 from top_k
        assert next_idx.shape == (BATCH, 1)
        next_token = torch.gather(top_idx, dim=1, index=next_idx)
        assert next_token.shape == (BATCH, 1)
        x = torch.cat((x, next_token), dim=1)[:, -Config.block_size:]

    response = enc.decode_batch(x.tolist())
    print(f'output = {response}')

torch.manual_seed(50)
if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
model = GPT.from_pretrained().to(device)
inference(model, device=device)