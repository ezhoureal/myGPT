import tiktoken
import torch
class DataLoader:
    def __init__(self, device):
        with open("data.txt", "r") as f:
            self.txt = f.read()
            
        self.idx = 0
        self.enc = tiktoken.encoding_for_model('gpt2')
        self.device = device
        self.tokens = self.enc.encode(self.txt)
    
    def get_batch(self, B: int, T: int):
        next_idx = self.idx + B * T + 1
        buffer = torch.tensor(self.tokens[self.idx:next_idx], device=self.device)
        self.idx = next_idx
        assert buffer.shape == (B * T + 1,), buffer.shape
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        return x, y