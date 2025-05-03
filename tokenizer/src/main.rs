use tokenizer::train_bpe;

mod tokenizer;
fn main() {
    println!("Hello, world!");
    let tokenizer = train_bpe(String::from("data/small_set.txt"), &vec![String::from("<Special>")], 300);
    tokenizer.encode();
}
