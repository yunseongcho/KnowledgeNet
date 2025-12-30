"""utils.py"""

import time
from datetime import datetime
from google import genai
from google.genai import types

from prompts.instructions import translator_instructions


def replace_info(txt: str, file_name: str, file, formatted_file_name: bool = False) -> str:
    """Replace format's information."""

    if formatted_file_name:
        yymm, journal, title = file_name.split("_")
        txt = txt.replace("title_gemini", title)
        txt = txt.replace("yymm_gemini", yymm)
        txt = txt.replace("journal_gemini", journal)
    else:
        txt = txt.replace("title_gemini", file_name)

    txt = txt.replace("file_gemini", file.name)
    txt = txt.replace("preview_gemini", f"![[{file_name}.pdf]]")

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d (%a) %p %I:%M")
    txt = txt.replace("date_gemini", formatted_date)
    return txt


def replace_equation(txt: str, color: str = "") -> str:
    """Change the formula in LaTex display math mode (ChatGPT) to markdown format with color."""

    if color:
        color = f"\\color{{{color}}}"
        txt = txt.replace("\\( ", f"${color}")
        txt = txt.replace(" \\)", "$ ")
        txt = txt.replace("\\(", f"${color}")
        txt = txt.replace("\\)", "$ ")
        txt = txt.replace("\\[", f"$${color}")
        txt = txt.replace("\\]", "$$")
        txt = txt.replace("**ANSWER**", "")
        return txt
    else:
        txt = txt.replace("\\( ", "$")
        txt = txt.replace(" \\)", "$ ")
        txt = txt.replace("\\(", "$")
        txt = txt.replace("\\)", "$ ")
        txt = txt.replace("\\[", "$$")
        txt = txt.replace("\\]", "$$")
        txt = txt.replace("**ANSWER**", "")
        return txt


def check_prompts_format_equal(prompts1: dict, prompts2: dict):
    """Check if prompts formats are equal."""

    if prompts1.keys() != prompts2.keys():
        raise ValueError("The first keys are not equal")

    for key in prompts1.keys():
        if prompts1[key].keys() != prompts2[key].keys():
            raise ValueError(f"Second keys which are the values of first key '{key}' are not equal")

    for key1 in prompts1.keys():
        for key2 in prompts1[key1].keys():
            if len(prompts1[key1][key2]) != len(prompts2[key1][key2]):
                raise ValueError(f"The number of question of {key1}{key2} are not equal")

    split_history_nums = [0]
    for key1 in prompts1.keys():
        num = split_history_nums[-1] + len(prompts1[key1])
        split_history_nums.append(num)

    if len(split_history_nums[:-1]) != len(prompts1.keys()):
        raise ValueError("The number of split_chat_history_nums are not equal to the number of prompts1 keys")

    return split_history_nums[:-1]


def process_prompts(
    prompts: dict,
    depth: int,
    counter: list[int],
    question_dict: dict,
    suffix: str,
) -> list[str, dict]:
    """This is a recursive function that writes prompts in prompt.py into the format.md file."""

    result = ""
    for key, value in prompts.items():
        if isinstance(value, dict):
            # Section header
            section_header = f"\n\n{'#' * depth} {key}\n"
            result += section_header

            # Process nested dictionary
            txt, question_dict = process_prompts(value, depth + 1, counter, question_dict, suffix)
            result += txt
        else:
            # Section header for the top level key if it's not a dict
            result += f"\n{'#' * depth} {key}\n"

            # Question block
            for question in value:
                txt = question.strip()
                question_block = f">[!question]\n>{txt}\n\n"
                result += question_block

                # Answer block
                result += ">[!answer]\n"

                # Identifier
                identifier = f"gpt_answer{counter[0]:02d}_{suffix}"
                result += f"{identifier}\n\n"
                counter[0] += 1

                question_dict[identifier] = question.replace("\n>", "\n").strip()

    return result.strip(), question_dict


"""
def upload_file(title: str, file_path: str):
    Upload the given file to Gemini AI and wait for it to process.

    Args:
        title (str): paper title
        file_path (str): file path

    Returns:
        file
    

    file = genai.upload_file(file_path, mime_type="application/pdf", display_name=title)
    file = genai.get_file(file.name)
    while file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(10)
        file = genai.get_file(file.name)
    if file.state.name != "ACTIVE":
        raise ValueError(f"File {file.name} failed to process")

    return file
"""


def get_answer_from_chat(chat_session, query: str, threshold_num: int = 5) -> str:
    i = 0
    while True:
        try:
            response = chat_session.send_message(query)
            break
        except Exception as e:
            time.sleep(30)
            i += 1
            if i > threshold_num:
                if "RECITATION" in str(e):
                    return "RECITATION"
                else:
                    raise ValueError(f"run failed {query} {e}") from e
            continue

    return response.text


def get_answer_from_model(client: genai.Client, query: str, configs: dict, threshold_num: int = 5) -> str:
    i = 0
    while True:
        try:
            response = client.models.generate_content(
                model=configs["Translator"]["model_name"],
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=translator_instructions,
                    temperature=configs["Translator"]["configs"]["temperature"],
                    top_p=configs["Translator"]["configs"]["top_p"],
                    top_k=configs["Translator"]["configs"]["top_k"],
                    max_output_tokens=configs["Translator"]["configs"]["max_output_tokens"],
                    response_mime_type=configs["Translator"]["configs"]["response_mime_type"],
                ),
            )
            break
        except Exception as e:
            time.sleep(30)
            i += 1
            if i > threshold_num:
                raise ValueError(f"run failed translate {e}") from e
            continue

    return response.text
