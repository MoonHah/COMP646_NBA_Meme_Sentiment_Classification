import pandas as pd
import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from tqdm import tqdm

# 1. Read and filter data
csv_path = 'Multi-Class-Emotion-Classification-of-NBA-Memes/final_annotations.csv'
image_dir = 'Multi-Class-Emotion-Classification-of-NBA-Memes/images_jpg_hot_945'

df = pd.read_csv(csv_path)
# Keep the required columns, including original_id
df = df[['image_id', 'original_id', 'text', 'parsed_teams']]

# 2. Load Qwen2.5-VL-7B-Instruct model
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# Recommended to use bfloat16 in a GPU environment to save VRAM
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 3. Define the prompt so the model outputs standard JSON format
system_prompt = """
You are an expert AI assistant that analyzes NBA memes. You are provided with an image and its associated text.
Based on the image and text, you must output a JSON object containing exactly 5 keys with binary values (1 or 0) following these rules:

Rule 1: Sentiment Polarity (Single Choice)
Exactly ONE of the following must be 1, the other two must be 0:
- "human_polarity__positive": 1 if the meme expresses praise, celebration, recognition, or positive sentiment.
- "human_polarity__negative": 1 if the meme expresses criticism, mockery, disappointment, or negative sentiment.
- "human_polarity__neutral": 1 if the meme is descriptive, hard to judge, or has mixed sentiments without a clear lean.

Rule 2: Meme Type (Multi-label)
Any of these can be 1 or 0:
- "human_type__humor": 1 if it is clearly a joke, irony, typical internet meme, or funny expression.
- "human_type__highlight": 1 if it highlights a great performance, stats, achievements, game moments, or player highlights.

OUTPUT ONLY VALID JSON. Example:
{
  "human_polarity__positive": 0,
  "human_polarity__negative": 1,
  "human_polarity__neutral": 0,
}
"""

# List for saving results
results = []

# 4. Iterate through and process the full dataset
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Memes"):
    original_id = row['original_id']
    text = str(row['text']) if pd.notna(row['text']) else ""

    # Check whether the image exists (using original_id)
    image_path = os.path.join(image_dir, f"{original_id}.jpg")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": f"Text in the post: {text}\nPlease analyze the meme and output the JSON."},
            ],
        }
    ]

    # Prepare inputs
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse JSON
    try:
        # Try to extract the JSON string
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        json_str = output_text[start:end]
        parsed_json = json.loads(json_str)
        # Record original_id for later merging
        parsed_json['original_id'] = original_id
        results.append(parsed_json)
    except Exception as e:
        print(f"Error parsing JSON for image {original_id}: {e}\nOutput was: {output_text}")

# 5. Merge results back into dataframe and save as a new CSV
results_df = pd.DataFrame(results)

# Merge by original_id and keep the original information
final_df = pd.merge(df, results_df, on='original_id', how='inner')

# Reorder columns to make them clearer
columns_order = [
    'image_id', 'original_id', 'text', 'parsed_teams',
    'human_polarity__positive', 'human_polarity__negative', 'human_polarity__neutral',
    'human_type__humor', 'human_type__highlight'
]
final_df = final_df[columns_order]

display(final_df)

# Save as a new CSV file
output_csv_path = 'Multi-Class-Emotion-Classification-of-NBA-Memes/new_annotations.csv'
final_df.to_csv(output_csv_path, index=False)
print(f"\nResults saved to {output_csv_path}")
