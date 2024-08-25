from transformers import LlamaTokenizer, LlamaForCausalLM
from local_token import HUGGINGFACE_TOKEN

model_name = "meta-llama/Llama-2-7b"
# Load the LLaMA model and tokenizer with the token
tokenizer = LlamaTokenizer.from_pretrained(model_name,use_auth_token=HUGGINGFACE_TOKEN)
model = LlamaForCausalLM.from_pretrained(model_name,use_auth_token=HUGGINGFACE_TOKEN)



import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text()
    return all_text


def generate_response(prompt, pdf_text, temperature=0.7, max_length=100):
    # Combine PDF text and the prompt
    combined_input = pdf_text + "\n\n" + prompt

    # Tokenize input
    inputs = tokenizer(combined_input, return_tensors="pt")

    # Generate output from LLaMA
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True
    )

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def main(pdf_path, prompt, temperature=0.7):
    # Step 1: Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Generate response from LLaMA
    response = generate_response(prompt, pdf_text, temperature=temperature)
    
    # Output the result
    print("Response:\n", response)

if __name__ == "__main__":
    # Example usage
    pdf_file = "data/bhagwad_geeta_Shankaracharya.pdf"
    user_prompt = "Can you summarize the content?"
    response_temperature = 0.7

    main(pdf_file, user_prompt, response_temperature)
