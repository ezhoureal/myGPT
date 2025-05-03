use std::iter::Map;

struct Merge {
    token1: u32,
    token2: u32
}
struct Tokenizer {
    vocab: Map<u32, String>,
    merges: Vec<Merge>
}

pub fn train(&self, file_name: String, special_tokens: &Vec<String>, vocab_size: usize) -> Tokenizer {
    let content = std::fs::read_to_string(file_name)
        .expect("Failed to read the file");
    
    let mut vocab = Map::<u32, String>::new

    return Tokenizer { vocab: vocab, merges: merges }
}

const PAT: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;
impl Tokenizer {
}