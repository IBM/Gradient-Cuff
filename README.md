# Gradient-Cuff
Repo for NeurIPS 2024 paper "Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes"

Project Page: [TrustSafeAI/GradientCuff-Jailbreak-Defense](https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense)

Paper preprint: [Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes
](https://arxiv.org/abs/2403.00867)

# Usage 

## Step 1: Prepare a JSON file to store jailbreak prompts

You need to store your jailbreak prompts in a format like [user_query.json](./user_query.json).


## Step 2: Run Gradient Cuff
### Parameters
The main file [main](./main) receives multiple parameters as input, the full list of these parameters is listed below:
```python
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id', 
        type=str, 
        choices=[
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "lmsys/vicuna-7b-v1.5",
            "lmsys/vicuna-13b-v1.5"
            ],
        help="the language model's model-id on hugging face."
        )
    parser.add_argument(
        '--user_query_path', 
        type=str,
        help="path of the JSON file to store the jailbreak prompts."
        )
    parser.add_argument(
        '--p_times', 
        type=int, 
        default=10,
        help="the number of the perturbation vectors. Corresponding to the 'P' in our paper."
        )
    parser.add_argument(
        '--sample_times', 
        type=int, 
        default=10,
        help="the number of the sampling times. Corresponding to the 'N' in our paper."
        )
    parser.add_argument(
        '--mu',
        type=int,
        default=0.02,
        help="the perturbation radius. Corresponding to the '\mu' in our paper."
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="Larger batch size can accelerate the defending process."
        )
    parser.add_argument(
        '--seed',
        type=int,
        default=13,
        help="the random seed. Fix it to make the results reproducible."
        )
    parser.add_argument(
        '--chat_temperature',
        type=float,
        default=0.6,
        help="sampling parameters when chatting with the protected language model."
        )
    parser.add_argument(
        '--chat_max_length',
        type=int,
        default=128,
        help="sampling parameters when chatting with the protected language model."
        )
    parser.add_argument(
        '--chat_p',
        type=float,
        default=0.9,
        help="sampling parameters when chatting with the protected language model."
        )
    parser.add_argument(
        '--threshold',
        type=float,
        default=100,
        help="the gradient norm threshold used in the second-stage defense. Corresponding to t in the paper."
        )
    parser.add_argument(
        '--hf_token',
        type=str,
        help="Your hugging face auth token. Some models like llama-2 cannot be downloaded if the user doesn't have the authentication."
        )
    parser.add_argument(
        '--device',
        type=str,
        default= "cuda:0",
        help="the device on which this program is running. You can choose 'cuda:x' or 'cpu' to specify the device."
        )
    parser.add_argument(
        '--save_path',
        type=str,
        default= "response.json",
        help='the path of the JSON file to store the model's responses to the input queries.'
        )

    args = parser.parse_args()
    assert args.model_id is not None
    assert args.user_query_path is not None
    
    return args
```
### Shell command

You may run the program using the command like:
```bash
python main.py --model_id meta-llama/Llama-2-7b-chat-hf --user_query_path user_query.json --save_path response.json --batch_size 25 --hf_token ${your_huggingace_token} --device cuda:0
```
If you want to remove the Gradient Cuff defense, just let N=0:
```bash
python main.py --model_id meta-llama/Llama-2-7b-chat-hf --N 0 --user_query_path user_query.json --save_path response.json --batch_size 25 --hf_token ${your_huggingace_token} --device cuda:0
```
### Collect responses

The model's response to the input queries would be saved in a JSON file like [response.json](./response.json)

# Cite
If you find Gradient Cuff helpful, please cite our paper in the following format:
```bash
@misc{hu2024gradient,
      title={Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes}, 
      author={Xiaomeng Hu and Pin-Yu Chen and Tsung-Yi Ho},
      year={2024},
      eprint={2403.00867},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
# Contact
Feel free to contact us ([Xiaomeng Hu](mailto:xmhu23@cse.cuhk.edu.hk), [Pin-Yu Chen](mailto:pin-yu.chen@ibm.com), [Tsung-Yi Ho](mailto:tyho@cse.cuhk.edu.hk)) if you have any questions about this project.
