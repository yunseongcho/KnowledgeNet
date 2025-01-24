# KnowledgeNet (AI 논문 설명기)

논문을 이해하거나, 요약 및 문서화 할 때 조금 편해보고자 만들었습니다. Gemini-2.0-flash-exp 모델을 사용하여 논문과 논문에 관한 프롬프트들을 Chat 형식으로 입력하여 논문에 대한 요약, 설명이 포함된 Markdown 파일을 만듭니다.

## Project Structure

```
inputs/                      # 입력 pdf 파일들이 저장되는 디렉토리
    main/                    # 읽고자 하는 논문의 main paper 를 pdf 형식으로 넣습니다.
    supple/                  # 읽고자 하는 논문의 supplemental material 을 pdf 형식으로 넣습니다. main paper와 동일한 이름을 가져야합니다.
    processed/               # 문서화가 끝난 main paper 및 supple 들이 이곳으로 옮겨집니다.
    
outputs/                     # Markdown 설명 파일들이 저장되는 디렉토리
    English/                 # Markdown 영어 출력 결과
    Korean/                  # Markdown 한국어 출력 결과
    Files/                   # 문서화가 끝나면 main paper 와 supplemental material 을 합쳐 이곳에 저장합니다. 

main.py                      # main script
utils.py                     # utility 함수가 정의된 파일
prompts/                     
    instructions.py          # system_instruction 을 정의
    prompts.py               # prompt 를 정의

format.md                    # Markdown output 형식 정의 파일
configs/                     
    gemini-2.0-flash-exp.json # Gemini 모델 configuration 
requirements.txt             # 필요한 Python 라이브러리 목록

.env                         # GEMINI_API_KEY 입력
pyproject.toml               
```

## Installation

`conda` 환경을 생성하고 `requirements.txt` 를 설치합니다:

```sh
conda create -n knowledgenet python=3.11
conda activate knowledgenet
pip install -r requirements.txt
```

## Usage

1. https://aistudio.google.com/apikey 에서 API_KEY를 발급받은 뒤 `.env` 파일을 생성하고 `GEMINI_API_KEY = ""` 를 입력하세요. (2025년 1월 24일 기준, 무료 API KEY 로도 동작합니다.)
2. `inputs/main` 폴더에 분석할 논문을 pdf 파일로 저장하세요.  
    -  `"YYMM_Journal_Title.pdf"` 의 파일 이름으로 저장한다면 메타 정보(발행연월/저널,컨퍼런스)가 Markdown 파일에 저장됩니다. 해당 포맷으로 파일 이름을 저장할 경우 arguments 로 `--formatted_file_name` 을 넣어주세요.
3. (supplementary 가 있다면) `inputs/supple` 폴더에 pdf 파일로 저장하세요. 이 때, **main paper와 파일 이름이 동일해야합니다.** supplementary 가 없어도 동작합니다.


모든 준비가 끝났습니다.  main script를 실행합니다:

```sh
python main.py
```

또는

```sh
python main.py --formatted_file_name --equation_color "orange"
```

실행이 완료되면, `outputs/Korean` 폴더에 한국어 결과물이 저장됩니다. **옵시디언**을 사용한다면, 더욱 가독성 좋게 결과물을 확인할 수 있습니다.


## Project Files

이 프로젝트에 포함된 파일은 다음과 같습니다:

- `main.py`: The main script to run the project.
    - **arguments**  
    `--formatted_file_name`: main paper pdf 파일 이름이 `"YYMM_Journal_Title"` 로 되어있는 경우 넣어줍니다. 발행연도와 저널/컨퍼런스 등의 메타 정보를 output 파일에 저장해줍니다. (ex. `2310_ICCV_StyleGANEX`)  
    `--equation_color={color_name}`: 수식이 좀 더 눈에 잘 띄도록 색깔을 바꿔줍니다. (ex. `--equation_color=orange`)
- `configs/gemini-2.0-flash-exp.json`: Configuration file for the models(explainer, translator).
- `requirements.txt`: List of dependencies.
- `utils.py`: Utility functions used in the project.
- `prompts/instructions.py`: explainer와 translator의 system_instruction.
- `prompts/prompts.py`: explainer에 입력할 prompts.
- `format.md`: Markdown output 형식 정의 파일.

## Tips
- model은 explainer 와 translator 두 종류의 모델을 사용합니다. explainer 는 논문을 영어로 설명합니다. translator 는 영문 설명을 학술/기술/전문 용어 및 고유 명사를 제외하고 한국어로 번역해줍니다.
- explainer 와 translator 두 모델의 configuration 은 `configs/gemini-2.0-flash-exp.json` 을 참조하십시오. `model_name`, `temperature`, `top_p` 등을 변경할 수 있습니다.
- 결과물에 대한 프롬프트 엔지니어링을 원할 경우, `prompts` 폴더를 참조하십시오. `instruction.py` 에는 explainer와 translator의 system_instruction 이 들어있으며, `prompts.py` 에는 explainer에 물어보는 프롬프트들이 저장되어 있습니다.
    - **이 때, 딕셔너리 `main_prompts_ko` 와 `main_prompts_en` 의 구조는 동일해야만합니다.** 마찬가지로 `supple_prompts_ko` 와 `supple_prompts_en` 의 구조 또한 동일해야만 합니다. 현재, 두 딕셔너리들의 구조가 동일하지 않으면 동작하지 않습니다. 두 프롬프트 딕셔너리는 의미는 동일하나, 한국어로 되어있는지 영어로 되어있는지, 프롬프트 엔지니어링이 추가되었는지 등의 차이가 있습니다.

## License

CC BY-NC-ND 4.0


## Contact

For any questions or issues, please contact by issue.