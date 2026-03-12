Set up the Weave library
Install the CLI and Python library for interacting with Weave and Wandb.

pip install wandb weave
Next, login to wandb and paste your API key when prompted.

wandb login
Generate your API key. Use it to log in to the wandb library.

Generate
You can also set your API key with the following environment variable.

import os
os.environ['WANDB_API_KEY'] = 'your_api_key'
Log a trace with code or
Start tracking inputs and outputs of functions by decorating them with weave.op.
Run this sample code to see the new trace.

OpenAI
Anthropic
In this example, we're using a generated OpenAI API key which you can find here.
Using another provider? We support all major clients and frameworks.

# Ensure your dependencies are installed with:
# pip install openai weave

# Find your OpenAI API key at: https://platform.openai.com/api-keys
# Ensure that your OpenAI API key is available at:
# os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"

import os
import weave
from openai import OpenAI

# Find your wandb API key at: https://wandb.ai/authorize
weave.init('dlopezkluever-aiuteur/intro-example') # 🐝

@weave.op # 🐝 Decorator to track requests
def create_completion(message: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ],
    )
    return response.choices[0].message.content

message = "Tell me a joke."
create_completion(message)
Running your first evaluation or
Evaluate a simple JSON QA task with Weave evaluators. Pick your provider below and run the snippet to log results.

OpenAI
Anthropic
In this example, we're using a generated OpenAI API key which you can find here.
Using another provider? We support all major clients and frameworks.

# Ensure your dependencies are installed with:
# pip install openai pandas weave

# Find your OpenAI API key at: https://platform.openai.com/api-keys
# Ensure that your OpenAI API key is available at:
# os.environ['OPENAI_API_KEY'] = "<your_openai_api_key>"

import asyncio
import os
import re
from textwrap import dedent

import openai
import weave


class JsonModel(weave.Model):
    prompt: weave.Prompt = weave.StringPrompt(
        dedent("""
You are an assistant that answers questions about JSON data provided by the user. The JSON data represents structured information of various kinds, and may be deeply nested. In the first user message, you will receive the JSON data under a label called 'context', and a question under a label called 'question'. Your job is to answer the question with as much accuracy and brevity as possible. Give only the answer with no preamble. You must output the answer in XML format, between <answer> and </answer> tags.
""")
    )
    model: str = "gpt-4.1-nano"
    _client: openai.OpenAI

    def __init__(self):
        super().__init__()
        self._client = openai.OpenAI()

    @weave.op
    def predict(self, context: str, question: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt.format()},
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {question}",
                },
            ],
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content


@weave.op
def correct_answer_format(answer: str, output: str) -> dict[str, bool]:
    parsed_output = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
    if parsed_output is None:
        return {"correct_answer": False, "correct_format": False}
    return {"correct_answer": parsed_output.group(1) == answer, "correct_format": True}


if __name__ == "__main__":
    if not os.environ.get('OPENAI_API_KEY'):
        print("OPENAI_API_KEY is not set - make sure to export it in your environment or assign it in this script")
        exit(1)

    # Find your wandb API key at: https://wandb.ai/authorize
    weave.init("dlopezkluever-aiuteur/intro-example")

    jsonqa = weave.Dataset.from_uri(
        "weave:///wandb/json-qa/object/json-qa:v3"
    ).to_pandas()

    model = JsonModel()

    eval = weave.Evaluation(
        name="json-qa-eval",
        dataset=weave.Dataset.from_pandas(jsonqa),
        scorers=[correct_answer_format],
    )

    asyncio.run(eval.evaluate(model))
Show less
Get started with Playground
You can interactively develop, review, and test their prompts using our LLM playground which supports all major model providers.