import tiktoken
import torch
class DataLoader:
    def __init__(self, device):
        with open("data.txt", "r") as f:
            self.txt = f.read()
            
        self.idx = 0
        self.enc = tiktoken.encoding_for_model('gpt2')
        self.device = device
    
    def get_batch(self, B: int, T: int):
        next_idx = self.idx + 1000
        chunk = self.txt[self.idx:next_idx]
        self.idx = next_idx
        tokens = self.enc.encode(chunk)
        buffer = torch.tensor(tokens[:B * T + 1], device=self.device)
        assert buffer.shape == (B * T + 1,), buffer.shape
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        return x, y