"""main paper and supplemental material prompts"""

# -----------------------------------------
# Main paper prompt 구조
# -----------------------------------------
main_prompts_ko = {
    "Main Paper": {
        "Paper Summary": ["전체 내용을 빠짐없이 논문의 목차에 따라 구조화해 요약하고, 핵심 수식을 포함해 주세요."],
        "Introduction": [
            "1. 이 논문에서 다루는 핵심 task(정의, 입력/출력, 목표/중요성)를 명확히 제시해 주세요.\n>2. 저자들의 연구 동기가 되는 핵심 task의 challenge 또는 problem이 무엇인지, 기존 접근법의 한계점은 무엇인지, 설명해 주세요.\n>3. 이 문제를 해결하기 위해 저자들은 어떤 접근법을 제시했나요? 논문에서 언급된 전반적 해결책과 핵심 아이디어를 간략히 설명해 주세요.\n>4. 최종적으로 저자들이 밝힌 주요 기여점(새로운 이론적/실용적 성과, 성능 향상, 새로운 모델 구조, 문제 정의 등)은 무엇인지 자세히 알려주세요."
        ],
        "Related Works": [
            "1. 저자들이 사용한 분류 기준에 따라 이 논문에서 언급된 유사 또는 관련 연구들의 제목을 모두 나열하고 각 연구의 핵심 아이디어와 한계점을 정리해주세요.\n>2. 기존 연구들과 비교했을 때 본 논문의 새로운 접근법과 차별화 포인트를 구체적으로 설명해 주세요."
        ],
    },
    "Methodology": {
        "Preliminaries": [
            "1. 이 논문에서 제시하는 방법을 이해하기 위해 필요한 주요 용어·기호는 무엇인지, 각각의 의미와 함께 설명해주십시오.\n>2. 이 논문에서 제시하는 방법을 이해하기 위한 수학적 배경(수식·이론)을 차근차근 풀어서 설명해 주세요.\n>3. 이 논문에서 제시하는 방법을 이해하기 위한 필수적인 사전 연구(선행 논문)를 알기 쉽게 설명하고 정리해 주세요.\n>4. 이 개념들이 이후 모델 설명과 어떻게 연결되는지 밝혀주십시오."
        ],
        "Framework": [
            "1. 전체 시스템 또는 프레임워크가 어떤 모듈/블록으로 구성되어 있는지 다음의 항목들을 포함하여 자세히 설명해주십시오.\n>    - 전체 시스템 또는 프레임워크의 모든 구성요소의 Baseline 또는 Backbone 과 그에 대응하는 refereces 및 선택 이유.\n>    - 각 구성요소에서 저자들이 주장하는 구조적 개선점과 최종 네트워크 구조(레이어 구성, 주요 모듈, 입력과 출력 등) 및 역할과 기능\n>2. 전체 시스템 또는 프레임워크의 입력과 출력, 각 구성요소 간 연결 과정(데이터 흐름)을 단계별로 설명해주십시오.\n>3. 프레임워크나 모델 아키텍처를 나타내는 그림(Figure)이 있다면, 그 위치(번호)와 함께 그림에서 중요한 부분들을 구체적으로 소개해 주세요."
        ],
        "Training": [
            "1. Training은 어떤 단계로 이뤄지며(예: 사전 학습 후 파인튜닝, 단계별 Loss 결합, 커리큘럼 러닝 등), 각 단계의 역할과 의미는 무엇인지 알려주세요.\n>2. 커스텀 손실 함수를 포함하여 각 단계에서 사용되는 손실 함수는 무엇이고, 어떤 출력에 적용되고 Training 대상은 무엇인지, 목적과 기능은 무엇인지 수식을 들어 설명해 주세요. \n>3. 어떤 최적화(Optimization) 기법과 하이퍼파라미터(러닝 레이트 스케줄 등)를 사용했는지, 어떻게 설정했는지 자세히 적어 주세요.\n>4. 커리큘럼 러닝, 멀티태스크 러닝, 자기지도 학습, 반지도 학습, 정규화(Regularization)와 같은 특별 학습 기법이 적용되었다면, 각각의 목적과 방식을 자세히 밝혀 주세요."
        ],
        "Inference and Application": [
            "1. 전체 시스템 혹은 프레임워크의 추론(Inference)은 어떻게 이뤄지나요? Inference 단계의 입력과 출력, 데이터 흐름을 구체적으로 서술해 주세요.\n>2. 논문에서 제시하는 실제 응용 사례(Use Case)가 있다면, 모든 예시를 빠짐없이 자세히 설명해 주세요.\n>3. 또한, 저자들이 강조하는 접근법의 실용적 장점(실시간 처리, 메모리 효율, 확장성 등)을 정리해 주십시오."
        ],
        "Method Summary": [
            "지금까지 제시된 내용을 바탕으로, 프레임워크 구성, 데이터 흐름, 학습 전략, 손실 함수, 추론 및 활용 가능성을 모두 아우르는 풍부한 요약을 작성해 주세요."
        ],
    },
    "Experiments": {
        "Datasets": [
            "1. 실험에 사용된 모든 데이터셋에 대하여 각 데이터셋의 종류나 라벨, 데이터 양 등 주요 특징을 설명해 주세요.\n>    - 데이터셋들의 분할(훈련, 검증, 테스트 등) 방법이 논문에 명시되어 있다면 설명해주십시오.\n>2. 만약 논문에서 직접 데이터를 수집했다면, 그 수집 과정과 데이터셋의 특성(종류, 라벨, 분량 등)을 설명해 주세요.\n>3. 실험의 시나리오에서 각 데이터셋이 실험에서 어떤 역할(훈련, 평가, 응용 등)을 맡았는지 구체적으로 알려 주세요."
        ],
        "Implementation Details": [
            "1. 실험 설정 및 하이퍼파라미터(예: Learning Rate, Batch Size, Optimizer, Epoch 수 등)와 구현 방법 등 Implementation Detail 전반을 빠짐없이 알려 주세요.\n>2. 훈련 시 사용한 GPU의 수와 종류, 훈련 시간 등도 가능한 한 자세히 밝혀 주세요.\n>3. 논문에서 재현성(Reproducibility)을 위해 코드나 모델을 공개했다면, 어떤 내용을 안내하고 있는지 알려 주세요."
        ],
        "Quantitative Results": [
            "1. 정량적 평가(Quantitative Evaluation)를 위해 어떤 지표(Accuracy, Precision, Recall, F1-score, IoU 등)를 사용했는지 모두 나열하고 해당 지표들의 수식과 해석 방법을 설명해 주세요. (수식이 없다면 관련 참고문헌을 제시하셔도 됩니다.)\n>2. 정량 결과를 확인하기 위해 참조해야 할 표(Table)들은 무엇이며, 각각 어떤 비교 결과를 보여줍니까?\n>3. 이 표들에 대해 저자들은 어떤 해석을 내렸으며, 수치가 의미하는 바와 이 방법의 장단점은 무엇인지 자세히 기술해 주세요."
        ],
        "Qualitative Results": [
            "1. 정성적 평가(Qualitative Result)에 대한 내용을 보려면 어떤 그림(이미지, 그래프, 예시 출력 등)들을 확인해야 하며, 각각 무엇을 나타내나요? 빠짐없이 공유해 주세요.\n>2. 이 그림들에 대해 저자들은 어떻게 해석하고 있나요? 어떤 결론과 의미를 이끌어내며, 방법의 장단점은 무엇인가요? 자세히 알려 주세요.\n>3. 논문에서 실패 사례(Failure Case)나 극단적 예시(Edge Case)를 보여주고 있다면, 어떤 예시이며 그 원인은 무엇인가요?"
        ],
        "Ablation Study": [
            "1. 저자들이 수행한 Ablation Study 목록을 모두 말해 주세요. 모듈, 손실 항, 알고리즘 등 어떤 요소를 제거 혹은 변경해 실험했는지, 그 목적은 무엇인지 설명해 주세요.\n>2. 각 Ablation Study 결과는 어느 표나 그래프, 그림을 보면 되나요? 그 위치(번호)를 알려 주십시오.\n>3. 각 Ablation Study에 대하여, 저자들의 해석에 따르면 어떤 구성 요소가 성능 개선에 중요한 역할을 하나요? 제거 혹은 추가 시 어떤 변화가 있는지, 그 구체적 해석을 알려 주세요."
        ],
        "Results Summary": [
            "지금까지 논의된 정량/정성적 결과와 Ablation Study를 종합하여 풍부하고 체계적인 요약을 부탁드립니다. 또한, 이 방법론의 강점과 약점도 동시에 짚어 주세요."
        ],
    },
    "Conclusion": {
        "Limitations ans Future works": [
            "1. 이 논문에서 제안한 방법의 단점이나 아직 해결되지 않은 문제, 일반화에 대한 이슈 등 한계점을 상세히 설명해 주세요.\n>2. 논문에서 제시된 후속 연구 아이디어나 확장 가능 방향(더 큰 규모의 데이터셋 적용, 모델 구조 확대 등)에 대해 구체적으로 풀어 써 주세요."
        ],
        "Conclusion": [
            "이 논문의 결론을 정리해 주세요. 연구진이 주장하는 바와 이를 뒷받침하는 핵심 근거(Method 강점 및 실험 결과)는 무엇인가요?"
        ],
    },
}
# -----------------------------------------
# Supplemental material prompt 구조
# -----------------------------------------

supple_prompts_ko = {
    "Supplemental Material": {
        "Supplemental Summary": [
            "Supplemental의 전체 내용을 빠짐없이 보충자료의 목차에 따라 구조화해 요약하고, 핵심 수식을 포함해 주세요."
        ],
        "Proofs": [
            "첨부한 파일에 추가적인 수식 또는 lemma, 증명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>제공된 파일에 포함된 모든 lemma나 수식에 대한 증명(또는 유도과정)을 빠짐없이 나열하고, 각, lemma나 수식의 증명 또는 유도 과정을 차근차근 단계적으로 설명해주십시오."
        ],
        "Framework": [
            "첨부한 파일에 전체 시스템 또는 Framework에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 전체 시스템 또는 프레임워크가 어떤 모듈/블록으로 구성되어 있는지 다음의 항목들을 포함하여 자세히 설명해주십시오.\n>    - 전체 시스템 또는 프레임워크의 모든 구성요소의 Baseline 또는 Backbone 및 선택 이유.\n>    - 각 구성요소에서 저자들이 주장하는 구조적 개선점과 최종 네트워크 구조(레이어 구성, 주요 모듈, 입력과 출력 등) 및 역할과 기능\n>\n>2. 프레임워크나 모델 아키텍처를 나타내는 그림(Figure)이 있다면, 그 위치(번호)와 함께 그림에서 중요한 부분들을 구체적으로 소개해 주세요."
        ],
        "Training and Loss function": [
            "첨부한 파일에 Training 및 Loss function 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. Training은 어떤 단계로 이뤄지는지, 각 단계의 역할과 의미는 무엇인지 알려주세요.\n>2. 커스텀 손실 함수를 포함하여 각 단계에서 사용되는 손실 함수는 무엇이고, 어떤 출력에 적용되고 Training 대상은 무엇인지, 목적과 기능은 무엇인지 수식을 들어 설명해 주세요. 추가적인 증명이 있다면 포함하십시오.\n>3. 어떤 최적화(Optimization) 기법과 하이퍼파라미터(러닝 레이트 스케줄 등)를 사용했는지, 어떻게 설정했는지 자세히 적어 주세요."
        ],
        "Inference and Application": [
            "첨부한 파일에 Inference 및 Application 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 전체 시스템 혹은 프레임워크의 추론(Inference)은 어떻게 이뤄지나요? Inference 단계의 입력과 출력, 데이터 흐름을 구체적으로 서술해 주세요.\n>2. 보충자료에서 제시하는 실제 응용 사례(Use Case)가 있다면, 모든 예시를 빠짐없이 자세히 설명해 주세요.\n>3. 또한, 저자들이 강조하는 접근법의 실용적 장점(실시간 처리, 메모리 효율, 확장성 등)을 정리해 주십시오."
        ],
        "Datasets": [
            "첨부한 파일에 Datasets 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 실험에 사용된 모든 데이터셋에 대하여 각 데이터셋의 종류나 라벨, 데이터 양 등 주요 특징을 설명해 주세요.\n>    - 데이터셋들의 분할(훈련, 검증, 테스트 등) 방법이 보충자료에 명시되어 있다면 설명해주십시오.\n>2. 만약 이 연구에서 직접 데이터를 수집했다면, 그 수집 과정과 데이터셋의 특성(종류, 라벨, 분량 등)을 설명해 주세요.\n>3. 실험의 시나리오에서 각 데이터셋이 실험에서 어떤 역할(훈련, 평가, 응용 등)을 맡았는지 구체적으로 알려 주세요."
        ],
        "Implementation Details": [
            "첨부한 파일에 Implementation Details 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 실험 설정 및 하이퍼파라미터(예: Learning Rate, Batch Size, Optimizer, Epoch 수 등)와 구현 방법 등 Implementation Detail 전반을 빠짐없이 알려 주세요.\n>2. 훈련 시 사용한 GPU의 수와 종류, 훈련 시간 등도 가능한 한 자세히 밝혀 주세요.\n>3. 보충자료에서 재현성(Reproducibility)을 위해 코드나 모델을 공개했다면, 어떤 내용을 안내하고 있는지 알려 주세요."
        ],
        "Quantitative Results": [
            "첨부한 파일에 Quantitative Results 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 정량 결과를 확인하기 위해 참조해야 할 표(Table)들은 무엇이며, 각각 어떤 비교 결과를 보여줍니까?\n>2. 이 표들에 대해 저자들은 어떤 해석을 내렸으며, 수치가 의미하는 바와 이 방법의 장단점은 무엇인지 자세히 기술해 주세요."
        ],
        "Qualitative Results": [
            "첨부한 파일에 Qualitative Results 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 정성적 평가(Qualitative Result)에 대한 내용을 보려면 어떤 그림(이미지, 그래프, 예시 출력 등)들을 확인해야 하며, 각각 무엇을 나타내나요? 빠짐없이 공유해 주세요.\n>2. 이 그림들에 대해 저자들은 어떻게 해석하고 있나요? 어떤 결론과 의미를 이끌어내며, 방법의 장단점은 무엇인가요? 자세히 알려 주세요.\n>3. 보충자료에서 실패 사례(Failure Case)나 극단적 예시(Edge Case)를 보여주고 있다면, 어떤 예시이며 그 원인은 무엇인가요?"
        ],
        "Ablation Study": [
            "첨부한 파일에 Ablation Study 에 대한 추가적인 설명이 있다면 아래의 질문에 답하십시오. 그렇지 않다면 '해당 정보는 제공된 파일에 없습니다.' 라고 명확히 답하십시오.\n>\n>1. 저자들이 수행한 Ablation Study 목록을 모두 말해 주세요. 모듈, 손실 항, 알고리즘 등 어떤 요소를 제거 혹은 변경해 실험했는지, 그 목적은 무엇인지 설명해 주세요.\n>2. 각 Ablation Study 결과는 어느 표나 그래프, 그림을 보면 되나요? 그 위치(번호)를 알려 주십시오.\n>3. 각 Ablation Study에 대하여, 저자들의 해석에 따르면 어떤 구성 요소가 성능 개선에 중요한 역할을 하나요? 제거 혹은 추가 시 어떤 변화가 있는지, 그 구체적 해석을 알려 주세요."
        ],
    }
}

# -----------------------------------------
# Main paper prompt structure
# -----------------------------------------
main_prompts_en = {
    "Main Paper": {
        "Paper Summary": [
            "After thoroughly reviewing and carefully examining the entire attached file, please provide a detailed, comprehensive summary of the paper’s full content.\n>\n>1. Organize your summary according to the paper’s section titles and explain each section in an easy-to-understand manner. But don't make the explanation too long-winded.\n>2. For each section, please include all relevant **key equations** and **key references** that support your explanations.\n>3. When citing references in the 'ANSWER' section, only use the reference titles (not authors’ names or numbers). Include the most important 10 core cited references.\n>4. Make your answer as long as necessary.\n>5. Finally, list core cited references of this paper in the 'SOURCES' section in an unnumbered IEEE format, including each paper’s title, authors, publication year, and publication venue."
        ],
        "Introduction": [
            "After carefully reviewing the “Introduction” and “Reference” sections of the attached file, please provide a thorough and detailed response by addressing the following points:\n>\n>For each section of the 'ANSWER', please include any relevant **key equations** and any pertinent **references** using only the paper titles (not authors’ names or numbers).\n>\n>1. Define the core task (including its definition, inputs/outputs, goals, and significance).\n>2. Describe the main challenges or problems related to this task and the limitations of previous methods. Please specify the paper **titles** of these previous methods in the 'ANSWER.'\n>3. Explain the overall solution and key ideas proposed by the authors to address the problems.\n>4. Detail the main contributions of the paper (e.g., theoretical or practical achievements, performance improvements, new model architectures, or problem definitions).\n>\n>Make your answer as lengthy as needed. Finally, list all references relevant to the 'ANSWER' in the 'SOURCES' section in an unnumbered IEEE format, including each reference’s title, authors, publication year, and publication venue."
        ],
        "Related Works": [
            "After thoroughly reviewing the “Related Works” and “Reference” sections of the attached file, please provide a comprehensive and detailed answer by addressing the following:\n>\n>1. According to the authors’ classification criteria, please list and categorize **all** related studies mentioned. Summarize the core idea of each study and note any limitations highlighted by the authors. When citing references, use only **the paper titles** (not authors’ names or numbers)\n>2. Discuss how the new approach in this paper differs from and improves upon these existing studies.\n>\n>Make your answer as lengthy as needed. Finally, please list **all** cited references in a 'SOURCES' section **without omissions** using an unnumbered IEEE format, including the title, authors, publication year, and publication venue for each reference."
        ],
    },
    "Methodology": {
        "Preliminaries": [
            "After reviewing the “Methodology” and “Reference” sections of the attached file, please provide a detailed answer addressing each of the following questions:\n>\n>1. What are the key terms and symbols necessary to understand the method proposed in this paper? Explain each term and symbol clearly.\n>2. Provide a step-by-step explanation of the mathematical background (equations and theories) required for understanding this method.\n>3. Clearly explain and organize the essential prior work referenced in the paper, making sure to include relevant references for that section.\n>4. Show how these concepts connect to the subsequent model description.\n>\n>In your 'ANSWER' section for each question, please include the relevant key equations and cite the references by **title** only (not by authors’ names or numbers). Make your explanation as long as necessary. Finally, in the 'SOURCES' section, list all references relevant to the 'ANSWER' used in unnumbered IEEE format with the titles, authors, publication years, and publication venues."
        ],
        "Framework": [
            "After reviewing the “Methodology” and “Reference” sections of the attached file, please provide a detailed answer focusing on the following:\n>\n>1. Structure of the Framework:\n>    - Describe how the entire system or framework is structured, covering each **module** or **component** without omission. Modules and components are not data. It's part of the model.\n>    - Specify the accurate baseline or backbone for each module or component (with the corresponding references by **title**) and the reasons for choosing them.\n>    - Describe any structural improvements proposed by the authors for each component, including the final network architecture (layer composition, main modules, inputs/outputs, etc.) and the role or function of each component.\n>\n>2. Data Flow:\n>    - Explain, step by step, how input data moves through the framework to produce the final output. Include any relevant equations.\n>\n>3. Framework Figure:\n>    - If there are figures (e.g., Figure X) illustrating the framework or model architecture, specify their figure numbers and describe the important parts in detail.\n>\n>In your 'ANSWER' section, please include relevant key equations where applicable, citing references by **title** only (not by authors’ names or numbers) for any baselines or backbones. Make your answer as long as necessary. Finally, provide all references relevant the 'ANSWER' to the 'SOURCES' section using unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Training": [
            "After reviewing the “Methodology” and “Reference” sections of the attached file, please provide a detailed, step-by-step answer to the following questions:\n>\n>1. Training Process:\n>    - Is training done in phases or is it not broken down into phases? What are the training phases (e.g., pre-training, fine-tuning, stepwise loss combination, curriculum learning)?\n>    - What is the role or significance of each phase?\n>    - If the authors were inspired by other references for the training process, cite those references by **title**.\n>\n>2. Loss Function:\n>    - Which loss functions (including any custom ones) are used in each phase, and what outputs do they apply to?\n>    - Include relevant equations and references by **title**, explaining their purpose, function and training target(object being trained or optimized) in detail.\n>\n>3. Optimization:\n>    - Which optimization methods and hyperparameters (e.g., learning rate schedule) are used, and how are they set?\n>\n>4. Special Training Techniques:\n>    - If any special training techniques (e.g., curriculum learning, multi-task learning, self-supervised learning, semi-supervised learning, regularization) are applied, explain the objective and method for each. If not, state so.\n>\n>Please write your 'ANSWER' as long as necessary, including all relevant formulas and references in each section. In the 'SOURCES' section, list all references relevant to the 'ANSWER' used in unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Inference and Application": [
            "After reviewing the “Methodology” and, if applicable, the “Inference” or “Application” sections of the attached file, please provide a detailed answer to the following:\n>\n>1. Inference Process:\n>    - How is the inference stage of the entire system or framework carried out?\n>    - Describe the inputs, outputs, and data flow step by step, including any relevant equations or figures.\n>\n>2. Use Case:\n>    - If the paper proposes real-world application scenarios (use cases), describe all such examples in full detail.\n>\n>3. Practical Advantages:\n>    - Summarize the practical advantages (e.g., real-time processing, memory efficiency, scalability) emphasized by the authors.\n>\n>Make your 'ANSWER' section as lengthy and detailed as needed. If relevant equations or figures exist, incorporate them into your explanation. Cite references by **title** only if needed, and list them in the 'SOURCES' section in unnumbered IEEE format (with titles, authors, publication years, and publication venues)."
        ],
        "Method Summary": [
            "Based on all the information presented so far, please provide a comprehensive summary of the entire methodology, covering:\n>\n>1. Framework Structure\n>2. Data Flow\n>3. Training Strategies\n>4. Loss Functions\n>5. Inference Procedures\n>6. Potential Applications (Usability)\n>\n>If there is any additional methodology-related content not previously explained, please include it as well. In your 'ANSWER' section, explain the relevant key formulas and cite references by **title** only (not by authors’ names or numbers). Make your summary as long and systematic as necessary. Finally, list all references relevant to your summary in the 'SOURCES' section using unnumbered IEEE format that includes the titles, authors, publication years, and publication venues."
        ],
    },
    "Experiments": {
        "Datasets": [
            "After carefully reviewing the “Experiments” (or “Results”) and “Reference” sections of the attached file, please provide a detailed answer addressing the following:\n>\n>1. Dataset Description:\n>    - Describe **all** the datasets used in the experiments (type or labels, size, and any notable characteristics). Please provide the relevant reference about dataset by **title** not by author name or number.\n>    - If the paper mentions how the datasets are split (train/validation/test), please include details.\n>\n>2. Data Collection Process:\n>    - If the authors collected any data themselves, explain the collection process and the dataset’s properties (type, labels, size, etc.).\n>\n>3. Role of Datasets:\n>    - Clarify how each dataset is utilized within the experimental setup (e.g., for training, evaluation, or application).\n>\n>Please include references by **title** only (not by author name or number) in each relevant section of your 'ANSWER.' Make your explanation as long as necessary, and list all references used in the 'SOURCES' section using unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Implementation Details": [
            "After carefully reviewing the “Experiments” (or “Results”) sections of the attached file, please provide a detailed answer addressing the following:\n>\n>1. Implementation Details:\n>    - Describe all experimental settings and hyperparameters (learning rate, batch size, optimizer, number of epochs, etc.) in depth.\n>\n>2. GPU Information:\n>    - Specify the type and number of GPUs used for training.\n>    - If available, include the approximate training time.\n>\n>3. Reproducibility:\n>    - If the paper provides guidance for reproducibility (open-sourced code or models, etc.), summarize the instructions or details given.\n>\n>Please make your 'ANSWER' as long and detailed as necessary."
        ],
        "Quantitative Results": [
            "After carefully reviewing the “Experiments” (or “Results”) and “Reference” sections of the attached file, please provide a detailed answer to the following:\n>\n>1. Evaluation Metrics:\n>    - Which metrics (e.g., Accuracy, Precision, Recall, F1-score, IoU) are used? Please provide the explanation of all the metrics used.\n>    - Please provide all the relevant equations and the **references** about each metric using paper **title**, and explain how each metric is interpreted.\n>\n>2. Reference Tables:\n>    - Which tables should we look at to see the quantitative results?\n>    - What comparisons do these tables show? Don't show the numbers in the table, explain what the table is comparing.\n>\n>3. Interpretation:\n>    - How do the authors interpret these tables?\n>    - What do the numbers signify, and what are the strengths and weaknesses of the proposed method according to these results?\n>\n>Please cite references in your 'ANSWER' by **title** only (not by author name or number). Make your response as long as needed, and list all relevant references in the 'SOURCES' section in unnumbered IEEE format (including titles, authors, publication years, and publication venues)."
        ],
        "Qualitative Results": [
            "After carefully reviewing the “Experiments” (or “Results”) sections of the attached file, please provide a detailed answer focusing on the following:\n>\n>1. Qualitative Results:\n>    - Which figures (images, graphs, example outputs, etc.) illustrate the qualitative results? (not for ablation study)\n>    - List them all and explain what each figure represents in detail.\n>\n>2. Interpretation:\n>    - How do the authors interpret these figures?\n>    - What conclusions and insights do they draw, and what strengths or weaknesses of the proposed method are highlighted?\n>\n>3. Failure Case:\n>    - If the paper includes any failure or edge cases, describe those examples and discuss potential causes.\n>\n>Make your 'ANSWER' as lengthy as needed to be comprehensive."
        ],
        "Ablation Study": [
            "After carefully reviewing the “Experiments” (or “Results”) sections of the attached file, please provide a detailed answer addressing the following:\n>\n>1. Ablation Study List:\n>    - List all ablation studies performed (e.g., removing or altering modules, loss terms, or algorithms).\n>    - Explain the purpose of each study.\n>\n>2. Reference Tables or Figures:\n>    - Which tables or figures present the results of each ablation study?\n>    - Please specify their table/figure numbers (e.g., Table X, Figure Y).\n>\n>3. Interpretation:\n>    - According to the authors, which components are most crucial for performance?\n>    - How does adding or removing each component affect the results?\n>\n>Make your 'ANSWER' as long as necessary to be thorough."
        ],
        "Results Summary": [
            "Please provide a comprehensive and systematic summary of the quantitative/qualitative results and the ablation studies discussed so far, addressing the following:\n>\n>1. Summary of Quantitative and Qualitative Results\n>2. Ablation Study Overview\n>3. Strengths and Weaknesses of the Methodology\n>\n>If there are any additional experimental details or results not yet covered, please include them as well. In your 'ANSWER,' incorporate relevant key formulas and cite references by **title** only (not by authors or numbers). Make your summary as lengthy as needed. Finally, list all references used in the 'SOURCES' section in unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
    },
    "Conclusion": {
        "Limitations ans Future works": [
            "After carefully reviewing the “Conclusion” (or “Limitation”/“Future Works”) sections of the attached file, please provide a detailed answer focusing on the following:\n>\n>1. Limitations:\n>    - What unresolved problems or generalization issues does the paper mention?\n>    - Are there any constraints or shortcomings explicitly stated?\n>\n>2. Future Works:\n>    - Which research directions or potential extensions do the authors propose (e.g., applying to larger datasets, expanding the model architecture)?\n>    - If the paper does not explicitly mention certain points, please note that they are not stated.\n>\n>Please make your 'ANSWER' as lengthy as needed to cover all details."
        ],
        "Conclusion": [
            "After carefully reviewing the “Conclusion” section of the attached file, please provide a thorough summary that addresses:\n>\n>1. Main Claims:\n>    - What are the key arguments or findings the authors emphasize?\n>    - How do they position their contribution in the field?\n>\n>2. Supporting Evidence:\n>    - Which methodological strengths or experimental results do the authors cite to justify their claims?\n>\n>Please make your 'ANSWER' as long as necessary to be comprehensive."
        ],
    },
}

# -----------------------------------------
# Supplemental material prompt structure
# -----------------------------------------
supple_prompts_en = {
    "Supplemental Material": {
        "Supplemental Summary": [
            "After thoroughly reviewing and carefully examining the entire attached file, please provide a detailed, comprehensive summary of the supplemental material’s full content.\n>\n>1. Organize your summary according to the supplemental material’s section titles and explain each section in an easy-to-understand manner.\n>2. For each section, please include all relevant **key equations** and **key references** that support your explanations.\n>3. When citing references in the 'ANSWER' section, only use the reference titles (not authors’ names or numbers). Include the most important 10 core cited references.\n>4. Make your answer as long as necessary.\n>5. Finally, list core cited references of this paper in the 'SOURCES' section in an unnumbered IEEE format, including each paper’s title, authors, publication year, and publication venue."
        ],
        "Proofs": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional formulas(equations), lemmas, and their derivation or proofs in the attached file, please provide a detailed, step-by-step answer to the following questions. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. List all proofs (or derivations) for the lemmas or formulas(equations) included in the provided file without omission.\n>2. Explain the proof or derivation process of each lemma or formula step by step.\n>\n>Please write your 'ANSWER' as long as necessary, including all relevant formulas and references in each section. In the 'SOURCES' section, list all references relevant to the 'ANSWER' used in unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Framework": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about the system, framework or model in the attached file, please provide a detailed answer focusing on the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Structure of the Framework:\n>    - Describe the system or framework, covering each **module** or **component** without omission. Modules and components are not data. It's part of the model.\n>    - Specify the accurate baseline or backbone for each module or component (with the corresponding references by **title**) and the reasons for choosing them.\n>    - Describe any structural improvements proposed by the authors for each component, including the final network architecture (layer composition, main modules, inputs/outputs, etc.) and the role or function of each component.\n>\n>2. Framework Figure:\n>    - If there are figures (e.g., Figure X) illustrating the framework or model architecture, specify their figure numbers and describe the important parts in detail.\n>\n>In your 'ANSWER' section, please include relevant key equations where applicable, citing references by **title** only (not by authors’ names or numbers) for any baselines or backbones. Make your answer as long as necessary. Finally, provide all references relevant the 'ANSWER' to the 'SOURCES' section using unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Training and Loss function": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about training process or loss function in the attached file, please provide a detailed, step-by-step answer to the following questions. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Training Process:\n>    - Is training done in phases or is it not broken down into phases? What are the training phases?\n>    - What is the role or significance of each phase?\n>\n>2. Loss Function:\n>    - Which loss functions (including any custom ones) are used in each phase, and what outputs do they apply to?\n>    - Include relevant equations and references by **title**, explaining their purpose, function and training target(object being trained or optimized) in detail.\n>\n>3. Optimization:\n>    - Which optimization methods and hyperparameters (e.g., learning rate schedule) are used, and how are they set?\n>\n>Please write your 'ANSWER' as long as necessary, including all relevant formulas and references in each section. In the 'SOURCES' section, list all references relevant to the 'ANSWER' used in unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Inference and Application": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about inference process or application of the framework in the attached file, please provide a detailed answer to the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Inference Process:\n>    - How is the inference stage of the entire system or framework carried out?\n>    - Describe the inputs, outputs, and data flow step by step, including any relevant equations or figures.\n>\n>2. Use Case:\n>    - If the supplemental material proposes real-world application scenarios (use cases), describe all such examples in full detail.\n>\n>3. Practical Advantages:\n>    - Summarize the practical advantages (e.g., real-time processing, memory efficiency, scalability) emphasized by the authors.\n>\n>Make your 'ANSWER' section as lengthy and detailed as needed. If relevant equations or figures exist, incorporate them into your explanation. Cite references by **title** only if needed, and list them in the 'SOURCES' section in unnumbered IEEE format (with titles, authors, publication years, and publication venues)."
        ],
        "Datasets": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about datasets in the attached file, please provide a detailed answer addressing the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Dataset Description:\n>    - Describe **all** the datasets used in the experiments (type or labels, size, and any notable characteristics). Please provide the relevant reference about dataset by **title** not by author name or number.\n>    - If the paper mentions how the datasets are split (train/validation/test), please include details.\n>\n>2. Data Collection Process:\n>    - If the authors collected any data themselves, explain the collection process and the dataset’s properties (type, labels, size, etc.).\n>\n>3. Role of Datasets:\n>    - Clarify how each dataset is utilized within the experimental setup (e.g., for training, evaluation, or application).\n>\n>Please include references by **title** only (not by author name or number) in each relevant section of your 'ANSWER.' Make your explanation as long as necessary, and list all references used in the 'SOURCES' section using unnumbered IEEE format (titles, authors, publication years, and publication venues)."
        ],
        "Implementation Details": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about implementation details in the attached file, please provide a detailed answer addressing the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Implementation Details:\n>    - Describe all experimental settings and hyperparameters (learning rate, batch size, optimizer, number of epochs, etc.) in depth.\n>\n>2. GPU Information:\n>    - Specify the type and number of GPUs used for training.\n>    - If available, include the approximate training time.\n>\n>3. Reproducibility:\n>    - If the paper provides guidance for reproducibility (open-sourced code or models, etc.), summarize the instructions or details given.\n>\n>Please make your 'ANSWER' as long and detailed as necessary."
        ],
        "Quantitative Results": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about quantitative results in the attached file, please provide a detailed answer to the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Reference Tables:\n>    - Which tables should we look at to see the quantitative results?\n>    - What comparisons do these tables show? Don't show the numbers in the table, explain what the table is comparing.\n>\n>2. Interpretation:\n>    - How do the authors interpret these tables?\n>    - What do the numbers signify, and what are the strengths and weaknesses of the proposed method according to these results?\n>\n>Please cite references in your 'ANSWER' by **title** only (not by author name or number). Make your response as long as needed, and list all relevant references in the 'SOURCES' section in unnumbered IEEE format (including titles, authors, publication years, and publication venues)."
        ],
        "Qualitative Results": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about qualtitative results in the attached file, please provide a detailed answer focusing on the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Qualitative Results:\n>    - Which figures (images, graphs, example outputs, etc.) illustrate the qualitative results? (not for ablation study)\n>    - List them all and explain what each figure represents in detail.\n>\n>2. Interpretation:\n>    - How do the authors interpret these figures?\n>    - What conclusions and insights do they draw, and what strengths or weaknesses of the proposed method are highlighted?\n>\n>3. Failure Case:\n>    - If the paper includes any failure or edge cases, describe those examples and discuss potential causes.\n>\n>Make your 'ANSWER' as lengthy as needed to be comprehensive."
        ],
        "Ablation Study": [
            "After thoroughly reviewing and carefully examining the entire attached file, if there are additional explanation about ablation studies in the attached file, please provide a detailed answer addressing the following. If not, clearly state \"the information does not exist in the provided file.\"\n>\n>1. Ablation Study List:\n>    - List all ablation studies performed (e.g., removing or altering modules, loss terms, or algorithms).\n>    - Explain the purpose of each study.\n>\n>2. Reference Tables or Figures:\n>    - Which tables or figures present the results of each ablation study?\n>    - Please specify their table/figure numbers (e.g., Table X, Figure Y).\n>\n>3. Interpretation:\n>    - According to the authors, which components are most crucial for performance?\n>    - How does adding or removing each component affect the results?\n>\n>Make your 'ANSWER' as long as necessary to be thorough."
        ],
    }
}
