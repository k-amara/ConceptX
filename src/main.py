from transformers import XLNetTokenizer

# Use the exact model identifier
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

# Example usage
input_text = "This is a test sentence."
encoded_input = tokenizer(input_text, return_tensors="pt")
print(encoded_input)
