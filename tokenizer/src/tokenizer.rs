use std::collections::HashMap;

pub struct Tokenizer {
    pub vocab: HashMap<u32, Vec<u8>>,
    pub merges: Vec<(u32, u32)>,
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
                if i < chunk.len() - 1 && chunk[i] == pair_to_merge.0 && chunk[i + 1] == pair_to_merge.1 {
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

pub fn train_bpe(file_name: String, special_tokens: &Vec<String>, vocab_size: u32) -> Tokenizer {
    let content = std::fs::read_to_string(file_name).expect("Failed to read the file");

    const PAT: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"#;
    let re = regex::Regex::new(PAT).expect("Failed to compile regex");
    let pre_tokens: Vec<&str> = re.find_iter(&content).map(|mat| mat.as_str()).collect();

    let mut vocab: HashMap<u32, Vec<u8>> = (0..256).map(|i| (i, vec![i as u8])).collect();
    let mut tokens: Vec<Vec<u32>> = pre_tokens
        .iter()
        .map(|chunk| chunk.as_bytes().iter().map(|&b| b as u32).collect())
        .collect();
    let mut merges: Vec<(u32, u32)> = Vec::new();
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
        println!("pair to merge = {:?}", pair_to_merge);
        let byte_pair = [&vocab[&pair_to_merge.0][..], &vocab[&pair_to_merge.1][..]].concat();
        vocab.insert(i, byte_pair);
        tokens = merge(&pair_to_merge, i, &tokens);
        merges.push(pair_to_merge);
        update_freq(&mut freq, &tokens, i);
        freq.remove(&pair_to_merge);
    }
    Tokenizer {
        vocab: vocab,
        merges: merges,
    }
}

impl Tokenizer {
    pub fn load_from_file(&self) {}

    pub fn encode(&self) {}

    pub fn decode(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_freq() {
        let tokens = vec![
            vec![1, 2, 3, 4],
            vec![2, 3, 4, 5],
            vec![1, 2, 3, 4],
        ];
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
        let tokens = vec![
            vec![1, 2, 3, 4],
            vec![2, 3, 4, 5],
        ];
        let pair_to_merge = (2, 3);
        let new_id = 6;
        let merged_tokens = merge(&pair_to_merge, new_id, &tokens);
        assert_eq!(merged_tokens, vec![
            vec![1, 6, 4],
            vec![6, 4, 5],
        ]);
    }

    #[test]
    fn test_train_bpe() {
        let file_name = "test.txt";
        let special_tokens = vec!["<pad>".to_string(), "<unk>".to_string()];
        let vocab_size = 300;

        // Create a temporary file for testing
        std::fs::write(file_name, "hello world hello").expect("Failed to write test file");

        let tokenizer = train_bpe(file_name.to_string(), &special_tokens, vocab_size);

        assert!(tokenizer.vocab.len() <= vocab_size as usize);
        assert!(!tokenizer.merges.is_empty());

        // Clean up the temporary file
        std::fs::remove_file(file_name).expect("Failed to remove test file");
    }
}
