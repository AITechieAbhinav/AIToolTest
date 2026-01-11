from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text, max_input_length=512, max_output_length=150):
    # Prepend "summarize: " as T5 expects task prefix
    input_text = "summarize: " + text.strip()

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=max_output_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example usage
if _name_ == "__main__":
    long_text = """
    The Eiffel Tower is one of the most iconic landmarks in the world, located in Paris, France. 
    Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized 
    by some of France's leading artists and intellectuals for its design. However, it has become a 
    global cultural icon of France and one of the most recognizable structures in the world.
    """

    print("Summary:\n", summarize_text(long_text))
