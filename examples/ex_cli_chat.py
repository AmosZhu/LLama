"""
Author: Dizhong Zhu
Date: 20/02/2024
"""

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
):
    ## This only work on 7B model, as I need to use this for debug
    generator = Llama.build_simple(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Continuous interaction loop
    dialogs = [[]]

    while True:
        user_input = input("You: ")  # Capture user input
        if user_input.lower() in ['quit', 'exit']:  # Exit condition
            print("Exiting the chat.")
            break

        # Generate dialog based on the user input
        user_prompt = {"role": "user", "content": user_input}
        dialogs[0].append(user_prompt)

        # Generate a response
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")

            response_prompt = {"role": result['generation']['role'], "content": result['generation']['content']}
            dialog.append(response_prompt)

            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
