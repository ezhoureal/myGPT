use std::collections::HashMap;

struct Merge {
    token_pair: (u32, u32),
    new_id: u32,
}
pub struct Tokenizer {
    vocab: HashMap<u32, Vec<u8>>,
    merges: Vec<Merge>,
    special_tokens: HashMap<String, u32>,
}

type PairFrequency = HashMap<(u32, u32), u32>;

fn get_freq(tokens: &Vec<Vec<u32>>) -> PairFrequency {
    let mut freq: PairFrequency = HashMap::new();
    for chunk in tokens {
        for pair in chunk.windows(2) {
            let pair = (pair[0], pair[1]);
            *freq.entry(pair).or_insert(0) += 1;
        }
    }
    return freq;
}

fn update_freq(freq: &mut PairFrequency, tokens: &Vec<Vec<u32>>, new_id: u32) {
    for chunk in tokens {
        for pair in chunk.windows(2) {
            if pair[0] == new_id || pair[1] == new_id {
                *freq.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
        }
    }
}

fn merge(pair_to_merge: &(u32, u32), new_id: u32, tokens: &Vec<Vec<u32>>) -> Vec<Vec<u32>> {
    tokens
        .into_iter()
        .map(|chunk| {
            let mut new_chunk = Vec::new();
            let mut i = 0;
            while i < chunk.len() {
                if i < chunk.len() - 1
                    && chunk[i] == pair_to_merge.0
                    && chunk[i + 1] == pair_to_merge.1
                {
                    new_chunk.push(new_id);
                    i += 2; // Skip the next element after a merge
                } else {
                    new_chunk.push(chunk[i]);
                    i += 1;
                }
            }
            new_chunk
        })
        .collect()
}

fn get_pre_tokens(file_name: &str, special_tokens: &Vec<String>) -> Vec<String> {
    let content = std::fs::read_to_string(file_name).expect("Failed to read the file");
    let special_tokens_pattern = special_tokens.join("|");
    let special_tokens_regex =
        regex::Regex::new(&special_tokens_pattern).expect("Failed to compile special tokens regex");
    let chunks: Vec<&str> = special_tokens_regex.split(&content).collect();

    const PAT: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"#;
    let re = regex::Regex::new(PAT).expect("Failed to compile regex");
    let pre_tokens: Vec<String> = chunks
        .iter()
        .flat_map(|segment| re.find_iter(segment).map(|mat| mat.as_str().to_string()))
        .collect();
    pre_tokens
}

impl Tokenizer {
    pub fn train_bpe(file_name: String, special_tokens: Vec<String>, vocab_size: u32) -> Self {
        let pre_tokens = get_pre_tokens(&file_name, &special_tokens);
        let mut vocab: HashMap<u32, Vec<u8>> = (0..256).map(|i| (i, vec![i as u8])).collect();
        let mut tokens: Vec<Vec<u32>> = pre_tokens
            .iter()
            .map(|chunk| chunk.as_bytes().iter().map(|&b| b as u32).collect())
            .collect();
        let mut merges: Vec<Merge> = Vec::new();
        let mut freq = get_freq(&tokens);

        for i in 256..(vocab_size - special_tokens.len() as u32) {
            if freq.is_empty() {
                break;
            }
            let pair_to_merge = freq
                .iter()
                .max_by_key(|element| element.1)
                .unwrap()
                .0
                .clone();
            let byte_pair = [&vocab[&pair_to_merge.0][..], &vocab[&pair_to_merge.1][..]].concat();
            vocab.insert(i, byte_pair);
            tokens = merge(&pair_to_merge, i, &tokens);
            merges.push(Merge {
                token_pair: pair_to_merge,
                new_id: i,
            });
            update_freq(&mut freq, &tokens, i);
            freq.remove(&pair_to_merge);
        }
        let special_token_map: HashMap<String, u32> = special_tokens
            .iter()
            .map(|special| {
                let id = vocab.len() as u32;
                vocab.insert(id, special.as_bytes().to_vec());
                (special.clone(), id)
            })
            .collect();
        Tokenizer {
            vocab,
            merges,
            special_tokens: special_token_map,
        }
    }

    pub fn load_from_file() -> Self {
        todo!()
    }

    pub fn encode(&self, content: String) -> Vec<u32> {
        // first convert special tokens and map other characters to bytes
        let mut tokens: Vec<Vec<u32>> = vec![Vec::new()];
        let special_token_map = &self.special_tokens;

        let mut chars = content.char_indices().peekable();
        while let Some((i, _)) = chars.peek() {
            let mut matched = false;

            for (token, &id) in special_token_map {
                if content[*i..].starts_with(token) {
                    tokens.push(vec![id]);
                    tokens.push(Vec::new());
                    for _ in 0..token.chars().count() {
                        chars.next();
                    }
                    matched = true;
                    break;
                }
            }

            if !matched {
                if let Some(last) = tokens.last_mut() {
                    if let Some((_, ch)) = chars.next() {
                        let mut buf = [0; 4];
                        let encoded = ch.encode_utf8(&mut buf);
                        for byte in encoded.as_bytes() {
                            last.push(*byte as u32);
                        }
                    }
                }
            }
        }

        // perform merges
        let mut freqs = get_freq(&tokens);
        for merge_record in &self.merges {
            if !freqs.contains_key(&merge_record.token_pair) {
                continue;
            }
            tokens = merge(&merge_record.token_pair, merge_record.new_id, &tokens);
            update_freq(&mut freqs, &tokens, merge_record.new_id);
        }
        tokens.into_iter().flatten().collect()
    }

    pub fn decode(&self, tokens: Vec<u32>) -> String {
        let pre_tokens: Vec<u8> = tokens
            .iter()
            .flat_map(|token| self.vocab[token].clone())
            .collect();
        String::from_utf8_lossy(&pre_tokens).to_string()
    }
}

#[cfg(test)]
mod tests {
    use parameterized::parameterized;

    use super::*;

    #[test]
    fn test_get_freq() {
        let tokens = vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5], vec![1, 2, 3, 4]];
        let freq = get_freq(&tokens);
        assert_eq!(freq[&(1, 2)], 2);
        assert_eq!(freq[&(2, 3)], 3);
        assert_eq!(freq[&(3, 4)], 3);
        assert_eq!(freq[&(4, 5)], 1);
    }

    #[test]
    fn test_update_freq() {
        let mut freq = HashMap::new();
        freq.insert((1, 2), 2);
        freq.insert((2, 3), 3);
        let tokens = vec![vec![1, 2, 3, 4]];
        update_freq(&mut freq, &tokens, 3);
        assert_eq!(freq[&(1, 2)], 2);
        assert_eq!(freq[&(2, 3)], 4);
        assert_eq!(freq[&(3, 4)], 1);
    }

    #[test]
    fn test_merge() {
        let tokens = vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5]];
        let pair_to_merge = (2, 3);
        let new_id = 6;
        let merged_tokens = merge(&pair_to_merge, new_id, &tokens);
        assert_eq!(merged_tokens, vec![vec![1, 6, 4], vec![6, 4, 5],]);
    }

    #[parameterized(train_text = {
        "hello world<pad>no hello", "rust is amazing<pad>code more"
    }, validation_text = {
        "hello<unk>Not me".to_string(), "rust<unk>rocks".to_string()
    }, file_name = {
        "text1.txt", "text2.txt"
    })]
    fn test_integrated(train_text: &str, validation_text: String, file_name: &str) {
        const VOCAB_SIZE: u32 = 512;
        let special_tokens: Vec<String> = vec!["<pad>".to_string(), "<unk>".to_string()];
        // Create a temporary file for testing
        std::fs::write(file_name, train_text).expect("Failed to write test file");

        let tokenizer =
            Tokenizer::train_bpe(file_name.to_string(), special_tokens.clone(), VOCAB_SIZE);

        assert!(tokenizer.vocab.len() <= VOCAB_SIZE as usize);
        assert!(!tokenizer.merges.is_empty());

        let tokens = tokenizer.encode(validation_text.clone());
        let result = tokenizer.decode(tokens);
        assert_eq!(validation_text, result);

        // Clean up the temporary file
        std::fs::remove_file(file_name).expect("Failed to remove test file");
    }
}
