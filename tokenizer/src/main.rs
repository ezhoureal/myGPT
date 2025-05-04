use tokenizer::Tokenizer;

mod tokenizer;
fn main() {
    println!("Hello, world!");
    let tokenizer = Tokenizer::train_bpe(String::from("../data/small_set.txt"), vec![String::from("<Special>")], 300);
    tokenizer.encode("What's up<end>".to_string());
}
