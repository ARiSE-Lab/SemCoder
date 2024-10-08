import transformers
import fire
import torch
import gradio as gr


def main(
    base_model="semcoder/semcoder",
    device="cuda:0",
    port=8080,
):
    pipeline = transformers.pipeline(
        "text-generation",
        model=base_model,
        torch_dtype=torch.float16,
        device=device
    )
    def evaluate_semcoder(
        instruction,
        temperature=1,
        max_new_tokens=2048,
    ):
        SEMCODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable <Code> according to <NL_Description>

<NL_Description>
{instruction}

<Code>
""" 
        prompt = SEMCODER_PROMPT.format(instruction=instruction)

        if temperature > 0:
            sequences = pipeline(
                prompt,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        else:
            sequences = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
            )
        for seq in sequences:
            print('==========================question=============================')
            print(prompt)
            generated_text = seq['generated_text'].replace(prompt, "")
            print('===========================answer=============================')
            print(generated_text)
            return generated_text

    gr.Interface(
        fn=evaluate_semcoder,
        inputs=[
            gr.components.Textbox(
                lines=3, label="Instruction", placeholder="Anything about Python code you want to ask SemCoder?"
            ),
            gr.components.Slider(minimum=0, maximum=1, value=1, label="Temperature"),
            gr.components.Slider(
                minimum=1, maximum=2048, step=1, value=512, label="Max tokens"
            ),
        ],
        outputs=[
            gr.components.Textbox(
                lines=30,
                label="Output",
            )
        ],
        title="SemCoder",
        description="This is a LLM playground for SemCoder!"
    ).queue().launch(share=True, server_port=port)

if __name__ == "__main__":
    fire.Fire(main)