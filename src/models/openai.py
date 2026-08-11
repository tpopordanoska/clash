import base64

from models.base_model import BaseModel
from openai import OpenAI
from utils.constants import OPENAI_API_KEY


class GPTModel(BaseModel):
    def __init__(self, **kwargs):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = kwargs.get(
            "model_name", "gpt-4.1-nano"
        )


    def prepare_model_input(self, sample, **kwargs):
        question_format = kwargs.get("question_format", "open_ended")
        wo_contr = kwargs.get("wo_contr", False)
        icl_samples = kwargs.get("icl_samples", [])
        if icl_samples:
            return self._prepare_icl_input(sample, icl_samples, question_format)

        prompt = self.construct_prompt(question_format, sample, wo_contr=wo_contr)

        return [
            {"type": "text", "text": prompt},
            self._image_content(sample["image_id"]),
        ]

    def _prepare_icl_input(self, sample, icl_samples, question_format):
        instruction, example_texts, target_text = self.construct_icl_prompt(
            question_format, sample, icl_samples
        )

        # Build content array: [instruction, (image, example_text)*, target_image, target_text]
        content = [{"type": "text", "text": instruction}]

        # Add each ICL example with its image
        for example_text, image_id in example_texts:
            content.append(self._image_content(image_id))
            content.append({"type": "text", "text": example_text})

        # Add target sample
        content.append(self._image_content(sample["image_id"]))
        content.append({"type": "text", "text": target_text})

        return content

    def _image_content(self, image_id):
        """Wrap a sample's image as an OpenAI image_url content block."""
        base64_image = self.encode_image(self.image_path(image_id))
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }

    def inference(self, model_input):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=500,
            messages=[{"role": "user", "content": model_input}],
        )

        return completion.choices[0].message.content

    def inference_as_judge(self, model_input):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": model_input}],
        )

        return response.choices[0].message.content

    # Function to encode the image
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
