use std::collections::{HashMap, HashSet, LinkedList};

#[derive(Debug)]
struct BPETokenizer {
    vocab: HashMap<u32, String>
}

impl BPETokenizer {
    fn new() -> Self {
        BPETokenizer {
            vocab: HashMap::new(),
        }
    }

    fn get_freq(&self, texts: &Vec<LinkedList<u32>>) -> HashMap<(u32, u32), u32> {
        let mut freqs = HashMap::new();

        for text in texts {
            let mut chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
            chars.push("</w>".to_string()); // Add end-of-word token

            for pair in chars.windows(2) {
                if let [a, b] = pair {
                    *freqs.entry((a.clone(), b.clone())).or_insert(0) += 1;
                }
            }
        }

        freqs
    }

    fn merge_vocab(&mut self, text: &Vec<LinkedList<u32>>, pair: &(String, String), new_id: u32) {
        let (a, b) = pair;
        let merged = format!("{}{}", a, b);

        let mut new_vocab = HashMap::new();
        for (token, freq) in &self.vocab {
            let new_token = token.replace(&format!("{} {}", a, b), &merged);
            new_vocab.insert(new_token, *freq);
        }

        self.vocab = new_vocab;
    }

    fn train(&mut self, texts: &[String], num_merges: usize) {
        rewrite
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        chars.push("</w>".to_string()); // Add end-of-word token

        let mut tokenized = chars.clone();
        let mut merged = true;

        while merged {
            merged = false;
            let mut best_pair = None;

            for pair in tokenized.windows(2) {
                if let [a, b] = pair {
                    if self.vocab.contains_key(&format!("{}{}", a, b)) {
                        best_pair = Some((a.clone(), b.clone()));
                        break;
                    }
                }
            }

            if let Some((a, b)) = best_pair {
                merged = true;
                let merged_token = format!("{}{}", a, b);
                let mut new_tokens = Vec::new();
                let mut skip = false;

                for token in tokenized {
                    if skip {
                        skip = false;
                        continue;
                    }

                    if token == a && !skip {
                        skip = true;
                        new_tokens.push(merged_token.clone());
                    } else {
                        new_tokens.push(token);
                    }
                }

                tokenized = new_tokens;
            }
        }

        tokenized
    }
}

fn main() {
    let texts = vec![
        "low".to_string(),
        "lowest".to_string(),
        "newer".to_string(),
        "wider".to_string(),
    ];

    let mut tokenizer = BPETokenizer::new();
    tokenizer.train(&texts, 10);

    let tokenized = tokenizer.tokenize("lowest");
    println!("{:?}", tokenized);
}