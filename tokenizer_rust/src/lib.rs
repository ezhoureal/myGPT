use pyo3::prelude::*;
mod tokenizer;
use tokenizer::Tokenizer;
#[pyclass]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    pub fn new(file_name: String, special_tokens: Vec<String>, vocab_size: u32) -> Self {
        PyTokenizer {
            inner: Tokenizer::train_bpe(file_name, special_tokens, vocab_size),
        }
    }

    pub fn encode(&self, content: String) -> Vec<u32> {
        self.inner.encode(content)
    }

    pub fn decode(&self, tokens: Vec<u32>) -> String {
        self.inner.decode(tokens)
    }
}

#[pymodule]
#[pyo3(name="rs_tokenizer")]
fn rs_tokenizer(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}