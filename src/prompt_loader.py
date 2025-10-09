import yaml
from langchain_core.prompts import ChatPromptTemplate

def load_prompt_from_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    prompt_data = data.get("rag_assistant_prompt", {})

    # Extract structured fields
    role = prompt_data.get("role", "")
    style = "\n".join(prompt_data.get("style_or_tone", []))
    instruction = prompt_data.get("instruction", "")
    constraints = "\n".join(prompt_data.get("output_constraints", []))
    output_format = "\n".join(prompt_data.get("output_format", []))

    # Combine into a single template for the model
    template = f"""
{role}

Style and Tone Guidelines:
{style}

Task Instruction:
{instruction}

Constraints:
{constraints}

Expected Output Format:
{output_format}

Context:
{{context}}

Question:
{{question}}

Answer:
"""

    return ChatPromptTemplate.from_template(template)
