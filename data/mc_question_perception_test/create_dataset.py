import json
import argparse
import os

def create_mllm_dataset(input_file, output_file=None):
    # If output_file is not specified, use input filename in current directory
    if output_file is None:
        output_file = os.path.basename(input_file)
        if os.path.exists(output_file):
            raise FileExistsError(f"Output file '{output_file}' already exists")
    
    # Get the SHARED_DATA environment variable
    shared_data = os.environ.get('SHARED_DATA')
    if shared_data is None:
        raise EnvironmentError("Environment variable SHARED_DATA is not set")
    
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Initialize output list
    output_data = []
    
    # Process each video entry
    for video_id in data:
        # Extract video number from video_id (e.g., "video_9431" -> "9431")
        video_num = video_id.split('_')[1]
        video_path = os.path.join(shared_data, "PerceptionTest/videos", f"video_{video_num}.mp4")
        
        # Process each multiple choice question
        for question in data[video_id]['mc_question']:
            # Create conversation entry with A, B, C options
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            options_str = ""
            for option, letter in zip(question['options'], letters):
                options_str += f"({letter}) {option}\n"
            entry = {
                "messages": [
                    {
                        "content": f"<video>{question['question']}\n{options_str}",
                        "role": "user"
                    },
                    {
                        "content": f"{question['options'][question['answer_id']]}",
                        "role": "assistant"
                    }
                ],
                "videos": [video_path]
            }
            output_data.append(entry)
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Convert video QA data to MLLM format')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file path')
    parser.add_argument('--output', type=str, help='Output JSON file path (optional)')
    args = parser.parse_args()
    
    create_mllm_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
