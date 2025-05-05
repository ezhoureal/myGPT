use tokenizer::Tokenizer;

mod tokenizer;
fn main() {
    let tokenizer = Tokenizer::train_bpe(String::from("data/data.txt"), vec![String::from("<Special>")], 512);
    let validation_text = "What's up<end>";
    let encoded = tokenizer.encode(validation_text.to_string());
    assert_eq!(tokenizer.decode(encoded), validation_text);
}
