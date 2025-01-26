"""KnowledgeNet main module"""

import argparse
import copy
import json
import os
import shutil

from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfMerger
from tqdm import tqdm

from prompts.instructions import explainer_instructions, translator_instructions
from prompts.prompts import main_prompts_en, main_prompts_ko, supple_prompts_en, supple_prompts_ko
from utils import (
    replace_info,
    replace_equation,
    check_prompts_format_equal,
    process_prompts,
    upload_file,
    make_history,
    append_history,
    get_answer_from_chat,
    get_answer_from_model,
)


# Load configs
parser = argparse.ArgumentParser(description="explain and translate papers")
parser.add_argument("--configs_path", type=str, default="./configs/gemini-2.0-flash-exp.json", help="configs_path")
parser.add_argument("--inputs_root", type=str, default="./inputs", help="root folder of the input pdf files")
parser.add_argument("--outputs_root", type=str, default="./outputs", help="root folder of the input pdf files")
parser.add_argument("--format_path", type=str, default="./format.md", help="format markdown file")
parser.add_argument("--formatted_file_name", action="store_true", help="Is the file name formatted? ex) YYMM_Journal_Title")
parser.add_argument("--equation_color", type=str, default="", help="color of the equation")
args = parser.parse_args()

with open(args.configs_path, "r", encoding="utf-8") as configs_file:
    configs = json.load(configs_file)

# Define client
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# check input folder
main_root = os.path.join(args.inputs_root, "main")
supple_root = os.path.join(args.inputs_root, "supple")
processed_root = os.path.join(args.inputs_root, "processed")

if len(os.listdir(main_root)) == 0:
    raise FileNotFoundError("There is no paper in input folder")

# make directories of outputs
output_en_root = os.path.join(args.outputs_root, "English")
output_ko_root = os.path.join(args.outputs_root, "Korean")
output_file_root = os.path.join(args.outputs_root, "Files")

os.makedirs(output_en_root, exist_ok=True)
os.makedirs(output_ko_root, exist_ok=True)
os.makedirs(output_file_root, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# check prompts format
split_history_nums = check_prompts_format_equal(main_prompts_ko, main_prompts_en)
check_prompts_format_equal(supple_prompts_ko, supple_prompts_en)


# Define explainer and translator
explainer = genai.GenerativeModel(
    model_name=configs["Explainer"]["model_name"],
    generation_config=configs["Explainer"]["configs"],
    system_instruction=explainer_instructions,
)
translator = genai.GenerativeModel(
    model_name=configs["Translator"]["model_name"],
    generation_config=configs["Translator"]["configs"],
    system_instruction=translator_instructions,
)

main_paper_names = [name for name in os.listdir(main_root) if name.endswith(".pdf")]
main_paper_names.sort()

# paper's loop
for main_paper_name in main_paper_names:
    file_name = main_paper_name.split(".")[0]

    # Load main and supple papers
    main_path = os.path.join(main_root, main_paper_name)
    supple_path = os.path.join(supple_root, main_paper_name)

    main_file = upload_file(file_name + "-main", main_path)
    if os.path.exists(supple_path):
        supple_file = upload_file(file_name + "-supple", supple_path)

    # open format file and replace the information
    with open(args.format_path, "r", encoding="utf-8") as format_file:
        format_txt = format_file.read()
    result_en = replace_info(format_txt, file_name, main_file, args.formatted_file_name)
    result_ko = copy.deepcopy(result_en)

    # make the prompt dictionary of depth 2 question dictionary of depth 1
    # prepare questions and result txt for main paper
    txt_en_main, q_dict_en_main = process_prompts(main_prompts_en, depth=1, counter=[1], question_dict={}, suffix="main")
    txt_ko_main, q_dict_ko_main = process_prompts(main_prompts_ko, depth=1, counter=[1], question_dict={}, suffix="main")
    result_en += txt_en_main
    result_ko += txt_ko_main

    # prepare questions and result txt for main paper
    # if supple paper exists
    if os.path.exists(supple_path):
        txt_en_supple, q_dict_en_supple = process_prompts(
            supple_prompts_en,
            depth=1,
            counter=[1],
            question_dict={},
            suffix="supple",
        )
        txt_ko_supple, q_dict_ko_supple = process_prompts(
            supple_prompts_ko,
            depth=1,
            counter=[1],
            question_dict={},
            suffix="supple",
        )
        result_en = result_en + "\n\n" + txt_en_supple
        result_ko = result_ko + "\n\n" + txt_ko_supple

    # main question loop
    identifiers_main = list(q_dict_en_main.keys())
    identifiers_main.sort()

    history = [{"role": "user", "parts": [""]}]
    for i, identifier_main in enumerate(tqdm(identifiers_main, desc=file_name + "-main")):
        # make history (append file to chat)
        if i in split_history_nums:
            history = make_history(main_file)

        # ask explainer
        question = q_dict_en_main[identifier_main]
        chat_session = explainer.start_chat(history=history)
        answer_en_origin = get_answer_from_chat(chat_session, question)

        # check if the answer is recitation
        if answer_en_origin == "RECITATION":
            with open("./logs/recitation_error.txt", "a", encoding="utf-8") as f:
                f.write(f"{file_name}  {identifier_main}\n")
                f.write(f"{question}\n")
                f.write("--------------------\n")
        else:
            # append history (append answer to chat)
            history = append_history(history, question, "user")
            history = append_history(history, answer_en_origin, "model")

            # replace answer in result_en
            answer_en = replace_equation(answer_en_origin, args.equation_color)
            result_en = result_en.replace(identifier_main, answer_en)
            with open(os.path.join(output_en_root, f"{file_name}.md"), "w", encoding="utf-8") as output_file:
                output_file.write(result_en)

            # ask translator
            answer_ko = get_answer_from_model(translator, answer_en_origin)

            # replace answer in result_ko
            answer_ko = replace_equation(answer_ko, args.equation_color)
            result_ko = result_ko.replace(identifier_main, answer_ko)
            with open(os.path.join(output_ko_root, f"{file_name}.md"), "w", encoding="utf-8") as output_file:
                output_file.write(result_ko)

    genai.delete_file(main_file.name)

    # if supple paper exists
    if os.path.exists(supple_path):
        identifiers_supple = list(q_dict_en_supple.keys())
        identifiers_supple.sort()

        # make history for supplemental materials (append file to chat)
        history = make_history(supple_file)
        for identifier_supple in tqdm(identifiers_supple, desc=file_name + "-supple"):
            # ask explainer
            question = q_dict_en_supple[identifier_supple]
            chat_session = explainer.start_chat(history=history)
            answer_en_origin = get_answer_from_chat(chat_session, question)

            # check if the answer is recitation
            if answer_en_origin == "RECITATION":
                with open("./logs/recitation_error.txt", "a", encoding="utf-8") as f:
                    f.write(f"{file_name}  {identifier_supple}\n")
                    f.write(f"{question}\n")
                    f.write("--------------------\n")
            else:
                # append history (append answer to chat)
                history = append_history(history, question, "user")
                history = append_history(history, answer_en_origin, "model")

                # replace answer in result_en
                answer_en = replace_equation(answer_en_origin, args.equation_color)
                result_en = result_en.replace(identifier_supple, answer_en)
                with open(os.path.join(output_en_root, f"{file_name}.md"), "w", encoding="utf-8") as output_file:
                    output_file.write(result_en)

                # ask translator
                answer_ko = get_answer_from_model(translator, answer_en_origin)

                # replace answer in result_ko
                answer_ko = replace_equation(answer_ko, args.equation_color)
                result_ko = result_ko.replace(identifier_supple, answer_ko)
                with open(os.path.join(output_ko_root, f"{file_name}.md"), "w", encoding="utf-8") as output_file:
                    output_file.write(result_ko)

        genai.delete_file(supple_file.name)

    # merge main and supple papers
    if os.path.exists(supple_path):
        try:
            merger = PdfMerger()
            merger.append(main_path)
            merger.append(supple_path)

            with open(os.path.join(output_file_root, f"{file_name}.pdf"), "wb") as output_pdf:
                merger.write(output_pdf)
            merger.close()
        except Exception as e:
            with open("./logs/pdf_merge_error.txt", "a", encoding="utf-8") as f:
                f.write(f"{file_name}\n")
    else:
        shutil.copy(main_path, os.path.join(output_file_root, f"{file_name}.pdf"))

    # move processed files
    processed_main_path = os.path.join(processed_root, main_paper_name)
    shutil.move(main_path, processed_main_path)

    if os.path.exists(supple_path):
        processed_supple_path = os.path.join(processed_root, file_name + "- supplemental.pdf")
        shutil.move(supple_path, processed_supple_path)
