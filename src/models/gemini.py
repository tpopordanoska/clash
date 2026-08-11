from google import genai

from models.base_model import BaseModel
from utils.constants import GEMINI_API_KEY


# pip install google-genai==0.5.0


class GeminiModel(BaseModel):
    def __init__(self, **kwargs):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = kwargs.get(
            "model_name", "gemini-2.5-flash-lite-preview-06-17"
        )

    def prepare_model_input(self, sample, **kwargs):
        question_format = kwargs.get("question_format", "open_ended")
        wo_contr = kwargs.get("wo_contr", False)

        icl_samples = kwargs.get("icl_samples", [])
        if icl_samples:
            return self._prepare_icl_input(question_format, sample, icl_samples)

        prompt = self.construct_prompt(question_format, sample, wo_contr=wo_contr)
        my_file = self._upload(sample["image_id"])

        return [my_file, prompt]

    def _upload(self, image_id):
        return self.client.files.upload(file=self.image_path(image_id))

    def _prepare_icl_input(self, question_format, sample, icl_samples):
        instruction, example_texts, target_text = self.construct_icl_prompt(question_format, sample, icl_samples)

        # Build content list: [instruction, (image, example_text)*, target_image, target_text]
        content = [instruction]

        # Add each ICL example with its image
        for example_text, image_id in example_texts:
            content.append(self._upload(image_id))
            content.append(example_text)

        # Add target sample
        content.append(self._upload(sample["image_id"]))
        content.append(target_text)

        return content

    def inference(self, model_input):
        response = self.client.models.generate_content(
            model=self.model_name, contents=model_input
        )

        return response.text

    def inference_as_judge(self, model_input):
        return self.inference(model_input)
