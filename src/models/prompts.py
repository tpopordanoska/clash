GENERATE_CONFLICTING_CAPTION_AND_ANSWER = """
You are an expert image caption editor and question generator.
Your task is to modify existing image captions and then create subtle questions based on your modifications.
Given an original image caption, you need to perform the following four steps:

Step 1: **Create a Conflicting Caption.**
   - Take the provided original caption.
   - Identify *one* key element (either a specific object or an attribute of an object, like its color, number, shape, material, or texture).
   - Change *only this one element* to create a subtle, but noticeable, conflict or discrepancy. The rest of the caption must remain identical to the original.
   - Ensure the conflict is a plausible, though incorrect, alternative (e.g., "red car" to "blue car," not "red car" to "flying car").
   - Do not change words that have binary states e.g. man-woman, open-close, dark-light, indoor-outdoor
When changing an attribute:
   - Only change **objective attributes** such as **color, number, shape, material, or texture**.
   - Do not change **subjective or ambiguous attributes** such as **beautiful, small, large, big, medium, moderate, modern, young, old, fast, slow, elegant, scary, tall, short**, etc.

Note: if a change with these constraints is not possible output None and skip the rest of the steps.

Step 2: **Track the Changed Words.**
   - Identify the exact word(s) that were changed from the original caption.
   - Record both the original word(s) and the replacement word(s).

Step 3: **Identify the Type of Change.**
   - Determine whether the change made in Step 1 was to an "object" (e.g., "cat" changed to "dog") or an "attribute" (e.g., "white" cat changed to "black" cat).

Step 4: **Generate a Subtle Question.**
   - Based on your *newly created conflicting caption*, formulate a question.
   - This question must subtly hint at the conflicting element without directly stating that something is wrong or different.
   - The question should encourage the user to focus on the changed element.

Here is an example. Provide your responses in the exact JSON format shown:
User: "A fluffy white cat is sitting on a red couch, looking out the window."
Model:
{{
  "conflicting_caption": "A fluffy black cat is sitting on a red couch, looking out the window.",
  "question": "What color is the feline observing the outside world?",
  "change_type": "attribute",
  "changed_words": {{
    "original": "white",
    "conflicting": "black"
  }}
}}
"""

GENERATE_MCQA_ANSWERS = """
Your task is to write **three answer choices** for a multiple-choice question that highlights a subtle conflict between two captions.

### Instructions:

Generate the following three answer options:

1. **image_only_answer**: The answer that fits the original caption.
2. **text_only_answer**: The answer that matches the conflicting caption.
3. **irrelevant_but_plausible**: A plausible distractor that doesn't appear in either the original or the conflicting caption, but is contextually reasonable. 
Make sure the distractor is NOT ambiguous or vague (avoid several, afternoon, medium, moderate, thing, stuff)). 

Ensure that all answers are:
- Concise (preferably 1–3 words),
- Mutually exclusive,
- Plausible in context, but not synonymous to each other.

Only output a JSON with the following fields:
{{
  "image_only": "...",
  "text_only": "...",
  "irrelevant_but_plausible": "..."
}}
"""

OPEN_ENDED_INSTRUCTION = """You are given an image and a textual description of the image.
Use *both* the visual and textual information to answer the question below as accurately as possible.
Only answer if your response can be clearly supported by both sources.
If the image and the description contradict each other, or if one is missing key information,
output "Conflicting information – cannot answer". Otherwise, provide the answer (less than 15 words). """

MULTIPLE_CHOICE_INSTRUCTION = """You are given an image and a textual description of the image.
Use *both* the visual and textual information to answer the multiple choice question below as accurately as possible.
Choose the option that is best supported by *both* sources.
If the image and the description contradict each other, or if one is missing key information, select the option “Conflicting information – cannot answer.”
Answer only with the letter corresponding to your choice in parenthesis: (A), (B), (C) or (D).
Do not include any other text."""

JUDGE_PROMPT = """
You are an evaluator. Given two reference answers (image_only and text_only) and a model prediction, decide which category the prediction belongs to:

1. IMAGE — if the prediction semantically matches the image_only answer.  
2. TEXT — if the prediction semantically matches the text_only answer.  
3. CONFLICT — if the prediction explicitly refers to a contradiction, conflict, or states that both cannot be true.  
4. NONE — if the prediction matches neither answer and does not indicate a conflict.  

Ignore minor differences in phrasing, synonyms, plural/singular, or capitalization.  
Return only one label: IMAGE, TEXT, CONFLICT, or NONE. 

---
Examples:

Image-only answer: polar bear  
Text-only answer: brown bear  
Prediction: Brown bear  
Output: TEXT  

Image-only answer: Black  
Text-only answer: Blue  
Prediction: Conflicting information – cannot answer.  
Output: CONFLICT  

Image-only answer: dog  
Text-only answer: cat  
Prediction: dog  
Output: IMAGE  

Image-only answer: red  
Text-only answer: green  
Prediction: yellow  
Output: NONE  
---
"""


ATTRIBUTE_CHECK = """
You are given two words: original and conflicting. Perform two steps:
    1. Decide if each word is an attribute (a descriptive property, e.g., red, tall, beautiful) or not an attribute (an object, e.g., car, tree).
    2. If both are attributes, classify them as:
    • Objective = measurable, factual, observable (e.g., red, blue, square, wooden, three).
    • Subjective = opinion-based or interpretive (e.g., beautiful, small, moderate, large, big, medium, modern, young, old, fast, slow, elegant, scary, tall, short, stylish, fancy, cheap, impressive.).

Only output a JSON with the following fields:
{
  "change_is_attribute": "Yes/No",
  "change_is_objective": "Yes/No"
}
"""

OBJECT_CHECK = """
You are given two words: original and conflicting.
Your task is to check the quality of the conflicting word in relation to the original:
    1. Determine whether each word is an object (a tangible or identifiable thing/entity, e.g., car, apple, chair).
    • Are the two words objects?
    2. Synonymy Check
    • Is the conflicting object a synonym or near-synonym of the original?
    3. Ambiguity Check
    • Is the conflicting object ambiguous or vague (e.g., “thing”, “object”, “stuff”)?
    4. Contextual Relevance
    • Does the conflicting object make sense in the same scene as the original?

Only output a JSON with the following fields:
{
  "change_is_object": "Yes/No",
  "change_is_synonym": "Yes/No",
  "change_is_ambiguous": "Yes/No",
  "change_is_relevant": "Yes/No"
}
"""

ANSWERS_CHECK = """
You are given three words/phrases. For each word/phrase check several things:
	1.	Synonymy Check
    Is one of the word/phrase a synonym or near-synonym to the other two?
    2.	Ambiguity Check
	Is some of the word/phrase ambiguous or vague (e.g., "several", "afternoon", "medium", “thing”, “stuff”)?
    3.	Contextual Relevance
	Are all word/phrase contextually relevant and objective (not subjective or off-topic)?
	4.  Visual Check
	Can each word/phrase be directly observed in an image? Examples of visual words: 
	attributes of objects (number, color, shape, size, material), object categories (car, chair, dog), 
	spatial relations (on top of, next to), scenes (beach, kitchen)). Examples of non-visual words:
	temporal concepts (afternoon, tomorrow), abstract states (freedom, happiness), non-observable attributes 
	(brand, taste, temperature), subjective labels (beautiful, boring, large).

Only output a JSON with the following fields:
{
  "ans_is_synonym": "Yes/No",
  "ans_is_ambiguous": "Yes/No",
  "ans_is_relevant": "Yes/No",
  "ans_is_visible": "Yes/No",
}
"""

QUESTION_CHECK = """
You are given a question, a set of changed words and three possible answers. Your task is to check the question based on three criteria: 

1. Ambiguity Check: Is the **question** clear, specific, and unambiguous?
2. Focus Check: Does the question explicitly ask about the **changed words**?
E.g. Question: What is the color of something? Changed words: green, blue -> Output Yes; 
Question: How many items are there? Changed words: three, four -> Output Yes
3. Answerability: Are all three candidate **answers** semantically and contextually compatible with the question?
E.g. Question: What is the color? Answers: green, blue, red -> Output Yes; 
Question: What is the gender? Answers: man, woman, child -> Output No

Only output a JSON object with the following fields:
{
  "question_is_clear": "Yes/No",
  "question_is_focused": "Yes/No",
  "question_is_answerable": "Yes/No"
}
"""

# ===================== ICL Prompts =====================

ICL_EXAMPLE_NON_CONFLICT = """Example {idx}a (no conflict):
Caption: {original_caption}
Question: {question}
Answer: {image_only_answer}
"""

ICL_EXAMPLE_CONFLICT = """Example {idx}b (has conflict):
Caption: {conflicting_caption}
Question: {question}
Answer: Conflicting information – cannot answer
"""
