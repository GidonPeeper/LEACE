import numpy as np
import transformers
import torch
from datasets import Dataset
import argparse
import gc

def load_dataset(path):

    print(f"loading dataset from path: {path}", flush=True)
    ds = Dataset.load_from_disk(path).with_format(
    "torch", 
    )
    
    return ds['input_ids'], ds['tags']

def process_and_pad_sequences(input_ids, pos_tags, start_token=1, stop_token=2, pad_id=3, pad_pos=-1, mous=False, linguistic_feature=None):
    """
    Function that processes and pads sequences. It can handle an additional linguistic feature
    that affects the dimensions of pos_tags.
    """
    
    if len(input_ids) != pos_tags.shape[0]:
        raise ValueError("Length of input_ids and pos_tags do not match")

    if len(input_ids.shape) > 1:
        input_ids = input_ids.flatten()
    
    # Handle 3D pos_tags if linguistic_feature is 'distances'
    if linguistic_feature == "distances":
        original_pos_dim = pos_tags.shape[-1]
        pos_tags = pos_tags.reshape(-1, original_pos_dim)
    else:
        pos_tags = pos_tags.flatten()

    # Initialize lists to hold the sentences before padding and token counts
    sentences_input_ids = []
    sentences_pos_tags = []
    sentence_lengths = []
    
    # Start processing from the first start token
    start_idx = (input_ids == start_token).nonzero(as_tuple=True)[0]
    end_idx = (input_ids == stop_token).nonzero(as_tuple=True)[0]
    
    if len(start_idx) != len(end_idx):
        raise ValueError("Mismatch in the number of start and stop tokens")
    
    for start, end in zip(start_idx, end_idx):
        sentence_input_ids = input_ids[start:end + 1]
        sentence_pos_tags = pos_tags[start:end + 1]
        
        sentences_input_ids.append(sentence_input_ids)
        sentences_pos_tags.append(sentence_pos_tags)
        sentence_lengths.append(len(sentence_input_ids))

    max_tokens = max(sentence_lengths)
    
    padded_input_ids = []
    padded_pos_tags = []
    for sentence_input_ids, sentence_pos_tags in zip(sentences_input_ids, sentences_pos_tags):
        pad_length = max_tokens - len(sentence_input_ids)
        if pad_length > 0:
            sentence_input_ids = torch.cat([sentence_input_ids, torch.full((pad_length,), pad_id)])
            if linguistic_feature == "distances":
                padding_pos_tags = torch.full((pad_length, original_pos_dim), pad_pos)
            else:
                padding_pos_tags = torch.full((pad_length,), pad_pos)
            sentence_pos_tags = torch.cat([sentence_pos_tags, padding_pos_tags])

        padded_input_ids.append(sentence_input_ids)
        padded_pos_tags.append(sentence_pos_tags)

    padded_input_ids = torch.stack(padded_input_ids)
    padded_pos_tags = torch.stack(padded_pos_tags)

    assert padded_input_ids.shape[0] == padded_pos_tags.shape[0], "Mismatch in number of sentences"
    assert padded_input_ids.shape[1] == padded_pos_tags.shape[1], "Mismatch in number of tokens"

    if mous:
        assert padded_input_ids.shape == torch.Size([400, 24]), f"Shape of padded_input_ids is {padded_input_ids.shape} when it should be [400, 24]"

    return padded_input_ids, padded_pos_tags, sentence_lengths

def generate_and_save_padded_hidden_states(padded_input_ids, output_path, model_name="GroNLP/bert-base-dutch-cased", batch_size=10500):
    """
    Function that generates padded hidden states for a given model in batches.
    Args:
        padded_input_ids (torch.Tensor): Tensor of input ids padded to the same length.
        model_name (str): Name of the model to load.
        batch_size (int): The size of the batch for processing.

    """
    # Assert model and layers
    assert model_name == "GroNLP/bert-base-dutch-cased", "Only GroNLP/bert-base-dutch-cased is supported for now."


    # Load the model
    print(f"Loading model {model_name} to save the hidden states for all layers", flush=True)
    model = transformers.AutoModel.from_pretrained(model_name, return_dict=True)
    model.eval()

    # Initialize random seed for reproducibility
    torch.manual_seed(42)

    # Process in batches
    batch_idx = 0
    num_samples = padded_input_ids.shape[0]
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_ids = padded_input_ids[start_idx:end_idx]

        print(f"Processing batch {start_idx + 1} to {end_idx} of {num_samples}", flush=True)

        # Create attention mask (assuming padding id is 3)
        attention_mask = (batch_ids != 3).int()

        # Generate hidden states with no gradient calculation
        with torch.no_grad():
            outputs = model(batch_ids, output_hidden_states=True, attention_mask=attention_mask)
        
        # Extract and store the hidden states for specified layers
        batch_hidden_states = outputs.hidden_states

        for layer in range(len(batch_hidden_states)):
            # save hidden states with torch
            torch.save(batch_hidden_states[layer], f"{output_path}/hidden_states_layer_{layer}_batch_{batch_idx}.pt")


        del outputs, batch_hidden_states
        gc.collect()

        batch_idx += 1

    return 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and save hidden states for a given model.")
    parser.add_argument("input_path", type=str, help="Path to the input dataset.")
    parser.add_argument("output_path", type=str, help="Path to save the hidden states.")
    parser.add_argument("--model_name", type=str, default="GroNLP/bert-base-dutch-cased", help="Name of the model to load.")
    parser.add_argument("--batch_size", type=int, default=5250, help="Batch size for processing.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    input_ids, tags = load_dataset(args.input_path)
    padded_input_ids, padded_tags, sentence_lengths = process_and_pad_sequences(input_ids, tags)
    generate_and_save_padded_hidden_states(padded_input_ids, args.output_path, args.model_name, args.batch_size)

    print("Hidden states generated and saved successfully", flush=True)