---
Title: RetouchFormer- Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention
Publication Date: 2403
Publication Place: AAAI
File Name: files/thzjnmpzumdm
Tags:
  - Generated_Document
  - Paper
Author Institution: 
Created On: 2025-12-30 (Tue) AM 09:14
Understanding Complete: 
Summary Complete: 
---
# Files
![[2403_AAAI_RetouchFormer- Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention.pdf]]
# Main Paper
## Paper Summary
>[!question]
>전체 내용을 빠짐없이 논문의 목차에 따라 구조화해 요약하고, 핵심 수식을 포함해 주세요.

>[!answer]


### 1. Abstract and Introduction

Face retouching은 original image content와 facial structures를 보존하면서 imperfections를 제거하여 얼굴 이미지를 아름답게 만드는 과정입니다. 이 논문은 이 분야의 두 가지 주요 과제를 식별합니다. 즉, 다양한 face imperfections를 자동으로 인식하는 것의 어려움과, 특히 여드름과 같은 심각한 사례에서 해당 영역에 realistic details를 합성하는 것의 어려움입니다.

저자들은 face retouching을 face image inpainting의 일반화된 "soft" 버전으로 취급하는 transformer-based approach인 **RetouchFormer**를 제안합니다. global transformations를 적용하는 일반적인 image enhancement 방법들과 달리, RetouchFormer는 imperfections를 식별하는 동시에 해당 특정 영역에 realistic content를 합성합니다. **Figure 1: The main idea behind RetouchFormer**에서 보여주듯이, 이 모델은 selective self-attention mechanism을 사용하여 imperfections의 features를 normal skin의 features로 대체합니다.

주요 기여는 다음과 같습니다:
- 대규모 paired datasets에 대한 의존도를 줄이는 semi-supervised framework.
- joint imperfection prediction과 local content synthesis를 통해 구현된 "soft inpainting" formulation.
- imperfection queries와 normal skin key-value pairs 간의 상호작용을 강화하기 위해 clean face priors를 기반으로 한 selective self-attention mechanism.

### 2. Related Work

이 논문은 RetouchFormer를 세 가지 주요 연구 분야 내에 위치시킵니다:
- **CNN-based Image Translation**: 여기에는 **"Generative adversarial nets"** 및 **"Image-to-image translation with conditional adversarial networks"** (Pix2Pix)와 같은 기초 연구뿐만 아니라, **"GPEN"** 및 **"MPRNet"**과 같은 high-quality restoration models가 포함됩니다.
- **Vision Transformer**: 이 architecture는 **"Attention is all you need"** 및 **"An image is worth 16x16 words: transformers for image recognition at scale"** (ViT)에서 영감을 얻었습니다. 또한 multi-scale representations를 처리하기 위해 **"Swin transformer: hierarchical vision transformer using shifted windows"**를 참조합니다.
- **Face Retouching**: 이 연구는 **"AutoRetouch: automatic professional face retouching"** 및 **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe)와 가장 밀접하게 관련되어 있습니다. RetouchFormer는 더 단순한 구조와 selective attention을 통한 더 적응적인 content synthesis를 사용하여 이들을 개선합니다.

### 3. Proposed Approach

이 architecture는 **Figure 2: The workflow of RetouchFormer**에 설명된 대로 encoder-transformer-decoder design을 따릅니다.

#### 3.1 Clean Face Priors
"clean" face가 어떤 모습이어야 하는지를 포착하기 위해, 저자들은 **"Neural discrete representation learning"**에서 설명된 대로 Vector Quantization (VQ)를 사용하여 feature dictionary $\mathcal{P}$를 학습합니다. encoder feature map $E(y_-)$의 임의의 위치 $(i, j)$에서의 feature vector는 dictionary에서 가장 가까운 요소로 근사화됩니다:
$$\color{orange}
\rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2
$$
여기서 $y_-$는 retouching image를 나타냅니다. 이 dictionary는 original image와 clean priors로부터 재구성된 이미지를 비교함으로써 imperfections를 식별하는 데 도움을 줍니다.

#### 3.2 Soft Mask Generation
imperfection regions를 나타내는 mask를 생성하기 위해 two-stream U-Net 구조($H$ 및 $\bar{H}$로 표시)가 사용됩니다. 모델은 imperfections가 있는 raw images가 clean face dictionary에 의해 완벽하게 재구성될 수 없다고 가정합니다. paired data의 경우, loss는 다음과 같이 정의됩니다:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)}\|\bar{H}(x_a, x_a) - \phi(x_a, y_a)\|_2^2
$$
여기서 $\phi$는 raw image $x_a$와 ground truth $y_a$ 사이의 차이를 측정합니다. unpaired data를 처리하기 위해, 모델은 **"Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"**에서 영감을 얻은 Exponential Moving Average (EMA) update strategy를 사용합니다.

#### 3.3 Selective Self-attention
핵심적인 혁신은 selective self-attention mechanism입니다. 이는 imperfection regions의 정보를 억제하고 이를 normal skin의 features로 대체합니다. queries ($Q$), keys ($K$), values ($V$)는 다음과 같이 공식화됩니다:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q,
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k,
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
여기서 $f_H$는 predicted mask에서 파생된 weighting map입니다. 이를 통해 "queries" (imperfections)가 빈틈을 채우기 위해 "keys/values" (normal skin)를 찾도록 보장합니다.

#### 3.4 Model Training
학습에는 high-quality reconstruction 및 retouching을 보장하기 위해 여러 loss functions가 포함됩니다:
- **Prior Loss**: encoder와 dictionary를 최적화합니다.
$$\color{orange}
L_{prior} = \mathbb{E}_{y_-} \left[ \sum_{(i,j)} \|E(y_-)|_{(i,j)} - \rho_*|_{(i,j)}\|_2^2 \right]
$$
- **Consistency Loss**: 합성된 이미지가 pixel-wise 및 perceptual similarity 측면에서 ground truth와 일치하도록 보장합니다 (**"The unreasonable effectiveness of deep features as a perceptual metric"**에 따라 VGG-19 features를 사용).
- **Adversarial Loss**: 결과가 실제 clean faces의 통계와 일치하도록 discriminator $D$가 사용됩니다.

전체 최적화는 multi-objective task입니다:
$$\color{orange}
\min_{E,T,G} L_{prior} + L_{con}^{rec} + L_{con}^{x2y} + L_{adv}^{syn}
$$

### 4. Experiments

#### 4.1 Datasets and Metrics
모델은 **FFHQR** dataset (70k pairs)과 **FR-wild** dataset (1,000 in-the-wild images)에서 평가되었습니다. 사용된 metrics는 PSNR, SSIM, LPIPS를 포함합니다.

#### 4.2 Quantitative and Qualitative Results
**Table 1: Quantitative comparison with competing methods on FFHQR**에서 보여주듯이, RetouchFormer는 모든 metrics에서 BPFRe 및 ABPN과 같은 state-of-the-art methods보다 우수한 성능을 보입니다. 예를 들어, $\gamma=0.1$ (10% paired data)에서 RetouchFormer는 45.13 dB의 PSNR을 달성하며, 이는 경쟁 모델들보다 상당히 높습니다.

정성적으로, **Figure 4: Visual comparison with competing methods on FR-wild images**는 RetouchFormer가 다른 방법들이 놓치는 대규모 blemishes와 다크서클을 효과적으로 제거함을 보여줍니다. **Figure 5**는 여드름을 지우고 high fidelity로 피부를 매끄럽게 만드는 능력을 추가로 보여줍니다.

### 5. Analysis of Design Elements

Ablation studies를 통해 각 구성 요소의 중요성을 확인했습니다:
- **Clean Face Prior**: prior $\mathcal{P}$를 비활성화하면 imperfection prediction에 대한 Soft-IoU가 17.9% 하락했습니다 (**Table 2**).
- **Multi-scale Tokens**: multi-scale features 대신 single patch size (SPS)를 사용하면 PSNR이 4.93 dB 하락했습니다 (**Table 3**).
- **Selective Self-attention**: 이를 Global Self-Attention (GSA)으로 대체하면 PSNR이 1.06 dB 감소하고 특정 곰보 자국(pockmarks)을 지우는 데 실패했습니다 (**Figure 7**).
- **Unpaired Data**: unpaired data ($X_u$) 없이 학습하면 성능이 크게 저하되어 semi-supervised approach의 가치를 입증했습니다.

### 6. Conclusion

RetouchFormer는 face retouching을 위한 강력한 transformer-based framework를 도입합니다. clean face priors와 selective self-attention mechanism을 결합함으로써, facial imperfections를 효과적으로 국소화하고 복구합니다. semi-supervised training strategy를 통해 모델은 제한된 paired data로도 우수한 결과를 얻을 수 있어 실제 응용 분야에서 매우 실용적입니다.

**SOURCES**

- A. v. d. Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in Proc. Neural Information Processing Systems, 2017.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Proc. Neural Information Processing Systems, 2017.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Proc. Neural Information Processing Systems, 2014.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," in Proc. International Conference on Computer Vision, 2021.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Advances in Neural Information Processing Systems, 2017.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.


## Introduction
>[!question]
>1. 이 논문에서 다루는 핵심 task(정의, 입력/출력, 목표/중요성)를 명확히 제시해 주세요.
>2. 저자들의 연구 동기가 되는 핵심 task의 challenge 또는 problem이 무엇인지, 기존 접근법의 한계점은 무엇인지, 설명해 주세요.
>3. 이 문제를 해결하기 위해 저자들은 어떤 접근법을 제시했나요? 논문에서 언급된 전반적 해결책과 핵심 아이디어를 간략히 설명해 주세요.
>4. 최종적으로 저자들이 밝힌 주요 기여점(새로운 이론적/실용적 성과, 성능 향상, 새로운 모델 구조, 문제 정의 등)은 무엇인지 자세히 알려주세요.

>[!answer]


### 1. Core Task: Face Retouching

**Face retouching**은 원본 이미지의 내용과 필수적인 **facial structures**를 최대한 보존하면서 다양한 유형의 **imperfections**를 제거하여 얼굴 이미지를 아름답게 만드는 과정으로 정의됩니다. **"Introduction"**에서 언급된 바와 같이, **input**은 일반적으로 여드름, 잡티, 주름과 같은 수많은 유형의 **imperfections**로 인해 손상된 얼굴 이미지입니다. **output**은 결점 없는 **skin tone**을 가진 깨끗하고 **high-fidelity**인 얼굴 이미지입니다.

이 과제의 주요 **goal**은 다양한 **face imperfections**를 자동으로 인식하고 해당 영역을 채울 수 있는 사실적인 **skin details**를 합성하는 것입니다. 이 과제의 **significance**는 현재 전문 리터처들이 몇 시간 동안 수작업으로 편집해야 하는 과정을 자동화할 수 있는 잠재력에 있습니다. 자동화된 솔루션을 제공함으로써 일반 사용자들도 고품질의 **face beautification**을 쉽고 효율적으로 이용할 수 있게 됩니다. 이러한 개념은 **Figure 1: The main idea behind RetouchFormer**에 시각적으로 소개되어 있으며, **imperfections**가 있는 **source image**에서 깨끗한 리터칭 결과로 전환되는 과정을 보여줍니다.

### 2. Main Challenges and Limitations of Previous Methods

**"Introduction"**에 따르면, **face retouching** 분야는 다음과 같은 몇 가지 중요한 **challenges**에 직면해 있습니다:
- **Imperfection Recognition**: 실제 환경에서 **face imperfections**는 매우 다양하기 때문에 모델이 이를 자동으로 정확하게 식별하기 어렵습니다.
- **Realistic Detail Synthesis**: 필수적인 **facial features**의 기저 구조를 보존하는 동시에, 특히 중간에서 심각한 수준의 여드름 영역에 사실적인 디테일을 채워 넣는 것은 매우 어렵습니다.

**Limitations of Previous Methods**:
- **Traditional Methods**: 초기 접근 방식은 결점 제거를 위해 **local smoothing operators**에 의존했으며, 이는 종종 자연스러운 **skin texture**의 손실을 초래했습니다.
- **Generic Image-to-Image Translation**: **"Image-to-image translation with conditional adversarial networks"** (Pix2Pix)에서 설명된 방식이나 **"MPRNet: Multi-stage progressive image restoration"**과 같은 일반적인 **image enhancement** 모델들은 특정 **imperfection localization** 기능이 부족한 경우가 많습니다. 이는 특히 큰 규모의 결점을 불완전하게 제거하는 결과로 이어집니다.
- **Specific Face Retouching Models**: **"AutoRetouch: automatic professional face retouching"** 및 **"ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"**와 같은 최근 모델들은 주로 **global convolution filters**를 사용한 **image-level transformations**를 고려합니다. 이러한 방법들은 특정 **imperfection** 영역을 적응적으로 완성하지 못하며, 훈련을 위해 수집 비용이 많이 드는 대량의 **pairwise retouching data** (원본 vs 전문적으로 리터칭된 데이터)를 필요로 합니다.
- **Two-Stage Approaches**: **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe)는 점진적인 복원을 시도했지만, 본 논문에서 제안하는 간소화된 방식에 비해 더 복잡한 **two-stage** 방법을 사용합니다.

### 3. Overall Solution and Key Ideas

저자들은 **face retouching**을 **face image inpainting**의 일반화된 "soft" 버전으로 취급하는 **transformer-based** 접근 방식인 **RetouchFormer**를 제안합니다. 이 모델은 **Figure 2: The workflow of RetouchFormer**에 표시된 것처럼 **encoder-transformer-decoder architecture**를 사용하여 **imperfections**를 식별하는 동시에 해당 영역에 사실적인 콘텐츠를 합성하도록 설계되었습니다.

**Key Ideas**:
- **Clean Face Priors**: 모델은 **"Neural discrete representation learning"**에서 상세히 설명된 기술인 **vector quantization**을 사용하여 **clean face priors**를 캡처하기 위한 **latent dictionary** $\mathcal{P}$를 학습합니다. 이 딕셔너리를 통해 모델은 깨끗한 특징들을 근사화할 수 있습니다:
$$\color{orange}
\rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2
$$
- **Soft Mask Generation**: 이전 방법들의 **localization** 부족 문제를 해결하기 위해, **RetouchFormer**는 **reconstruction-oriented localization module**을 통해 **imperfection** 영역을 예측합니다. 이는 **two-stream U-Net** 구조를 사용하여 **imperfections**가 위치한 곳을 나타내는 **mask** $H$를 생성합니다. **paired data**에 대한 **loss**는 다음과 같습니다:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)}\|\bar{H}(x_a, x_a) - \phi(x_a, y_a)\|_2^2
$$
**unpaired data**를 활용하기 위해 모델은 **"Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"**에서 영감을 받은 **Exponential Moving Average (EMA)** 전략을 채택합니다.
- **Selective Self-Attention**: 예측된 **mask**를 기반으로 모델은 **"selective self-attention"**을 수행합니다. 이 메커니즘은 **imperfection** 영역의 특징(**queries** 역할)을 명시적으로 억제하고, 이를 정상적인 피부 영역에서 합성된 특징(**keys** 및 **values** 역할)으로 대체합니다. **queries** $Q$, **keys** $K$, **values** $V$는 다음과 같이 공식화됩니다:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
- **Multi-scale Feature Tokens**: **transformer**는 다양한 크기와 규모의 **imperfections**를 처리하는 데 유연성을 제공하기 위해 **multi-scale feature tokens**를 사용합니다.

### 4. Main Contributions

논문은 **"Introduction"**과 **"Proposed Approach"** 섹션에서 세 가지 주요 기여를 나열합니다:
- **Semi-supervised Framework**: 저자들은 모델 학습을 위해 비용이 많이 드는 **paired retouching data**에 대한 의존도를 크게 줄이는 **semi-supervised face retouching transformer**를 제안합니다.
- **Soft Inpainting Formulation**: 이 모델은 **face retouching**을 **"soft inpainting"** 과제로 규정합니다. 이는 근사적인 **imperfection prediction**과 **local content synthesis**의 결합 프로세스를 통해 실현되며, **global filter-based** 방법보다 더 정밀하고 적응적인 보정을 가능하게 합니다.
- **Prior-Based Selective Self-Attention**: **clean face priors**에 기반한 **selective self-attention** 메커니즘을 도입했습니다. 이 메커니즘은 **imperfection** 영역의 **queries**와 정상 피부 영역의 **key-value pairs** 사이의 공간적 상호작용을 강화하여, 채워진 콘텐츠가 사실적이면서도 문맥적으로 적절하도록 보장합니다.
- **Performance and Efficiency**: **RetouchFormer**는 **fidelity**와 **realism** 측면에서 고급 **face retouching** 방법들을 능가합니다. **"Abstract"**에서 언급된 바와 같이, 이러한 설계 요소들은 더 큰 효과와 효율성을 가져다주며, 모델이 더 빨리 수렴하고 더 높은 품질의 결과를 달성할 수 있게 합니다 (예: **Figure 3**에 표시된 것처럼 베이스 모델 대비 1.3배의 **PSNR** 향상 및 6배의 효율성 달성).

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2021.
- B. Lei, X. Guo, Y. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in Proceedings of the International Conference on Neural Information Processing Systems (NIPS), 2017.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proceedings of the International Conference on Neural Information Processing Systems (NIPS), 2017.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M. H. Yang, and L. Shao, "MPRNet: Multi-stage progressive image restoration," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
- K. Arakawa, "Nonlinear digital filters for beautifying facial images in multimedia systems," in Proceedings of the IEEE International Symposium on Circuits and Systems (ISCAS), 2004.
- N. Batool and R. Chellappa, "Detection and inpainting of facial wrinkles using texture orientation fields and Markov random field modeling," IEEE Transactions on Image Processing, vol. 23, no. 9, 2014.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.


## Related Works
>[!question]
>1. 저자들이 사용한 분류 기준에 따라 이 논문에서 언급된 유사 또는 관련 연구들의 제목을 모두 나열하고 각 연구의 핵심 아이디어와 한계점을 정리해주세요.
>2. 기존 연구들과 비교했을 때 본 논문의 새로운 접근법과 차별화 포인트를 구체적으로 설명해 주세요.

>[!answer]


제공된 논문의 "Related Work" 및 "Introduction" 섹션을 바탕으로, 저자들은 기존 연구를 CNN-based Image Translation, Vision Transformer, 그리고 Face Retouching의 세 가지 주요 분야로 분류합니다.

### 1. 관련 연구의 분류 (Categorization of Related Studies)

#### **CNN-based Image Translation**
이 카테고리는 이미지를 한 domain에서 다른 domain으로 mapping하기 위해 Convolutional Neural Networks (CNNs)를 사용하는 것을 포함합니다.

*   **Foundational Generative Models**:
    *   **"Generative adversarial nets"**: adversarial framework를 사용하여 high-fidelity image synthesis 기능을 도입했습니다.
    *   **"Image-to-image translation with conditional adversarial networks"** (Pix2Pix): source domain과 target domain 사이의 distribution discrepancy 및 pixel-wise discrepancy를 최소화하는 mapping을 학습했습니다. **한계점**: 수집 비용이 많이 들고 시간이 오래 걸리는 pairwise training data가 필요합니다.
*   **Unpaired Image Translation**:
    *   **"Unsupervised image-to-image translation networks"**, **"Unpaired image-to-image translation using cycle-consistent adversarial networks"**, 그리고 **"Learning to discover cross-domain relations with generative adversarial networks"**: 이 방법들은 data distribution alignment를 수행하거나 cycle consistency regularization을 부여하기 위해 two-way mapping을 사용하여 paired data의 부족 문제를 해결했습니다.
*   **High-Quality Restoration and Multi-stage Models**:
    *   **"GAN prior embedded network for blind face restoration in the wild"** (GPEN): high-quality image generation을 보장하기 위해 U-shaped CNN과 GAN을 결합했습니다.
    *   **"MPRNet: Multi-stage progressive image restoration"**: multi-stage image translation을 위해 설계되었으며, high-level contextual information과 spatial details의 이점을 활용합니다.
*   **Semantic Editing and GAN Inversion**:
    *   **"GANalyze: toward visual definitions of cognitive image properties"**, **"On the “steerability” of generative adversarial networks"**, **"Editing in style: uncovering the local semantics of GANs"**, **"Closed-form factorization of latent semantics in GANs"**, 및 **"StyleSpace analysis: disentangled controls for StyleGAN image generation"**: 이 연구들은 pre-trained GANs의 latent space가 semantic organization을 가지고 있어 semantic editing이 가능하다는 점을 관찰했습니다.
    *   **"Image processing using multi-code GAN prior"**, **"Encoding in style: a StyleGan encoder for image-to-image translation"**, 및 **"Designing an encoder for StyleGan image manipulation"**: 이 방법들은 조작을 위해 이미지를 다시 latent space로 투영하는 high-fidelity GAN inversion을 활용했습니다.
    *   **"StyleFlow: attribute-conditioned exploration of StyleGAN Generated images using conditional continuous normalizing flows"**: latent space에서 non-linear transformations를 학습하기 위해 continuous normalizing flow model을 채택했습니다.
    *   **"StyleCLIP: text-driven manipulation of StyleGAN imagery"**: 이미지 편집을 위해 language-vision pre-training을 활용했습니다.
    *   **일반적인 한계점**: 이러한 방법들 대부분은 입력 이미지의 구체적인 변형을 고려하지 않고 semantic image translation을 위한 global latent transformations를 수행합니다.

#### **Vision Transformer**
이 카테고리는 원래 자연어 처리 분야의 Transformer architecture를 computer vision task에 적응시키는 데 중점을 둡니다.

*   **Foundational Architectures**:
    *   **"Attention is all you need"** 및 **"BERT: Pre-training of deep bidirectional transformers for language understanding"**: 핵심적인 Transformer 및 self-attention mechanisms를 확립했습니다.
*   **Image Recognition and Processing**:
    *   **"An image is worth 16x16 words: transformers for image recognition at scale"** (ViT) 및 **"CvR: Introducing convolutions to vision transformers"**: representation learning에서 우수한 성능을 입증했습니다.
    *   **"Swin transformer: hierarchical vision transformer using shifted windows"**: 높은 image resolutions와 scale variations를 해결하기 위해 shifted windows를 가진 hierarchical representation을 학습했습니다.
    *   **"Pre-trained image processing transformer"** (IPT): denoising 및 super-resolution과 같은 low-level tasks를 위해 대규모 pre-trained model을 채택했습니다.
*   **Restoration and Editing**:
    *   **"RestoreFormer: High-quality blind face restoration from undegraded key-value pairs"**: blind face restoration을 개선하기 위해 high-quality key-value pairs 형태의 priors와 cross-attention을 수행했습니다.
    *   **"Style transformer for image inversion and editing"**, **"End-to-end object detection with transformers"**, 및 **"Deformable DETR: Deformable transformers for end-to-end object detection"**: Transformer 기반 방법들을 이미지 편집 및 object detection에 적용했습니다.

#### **Face Retouching**
이 카테고리는 결점을 제거하여 얼굴 이미지를 아름답게 만들기 위해 특별히 설계된 방법들을 포함합니다.

*   **Traditional and Smoothing Methods**:
    *   **"Nonlinear digital filters for beautifying facial images in multimedia systems"**, **"Detection and inpainting of facial wrinkles using texture orientation fields and Markov random field modeling"**, **"Automatic freckles detection and retouching"**, 및 **"Data-driven enhancement of facial attractiveness"**: 이 방법들은 일반적으로 결점 제거를 위해 local smoothing operators 또는 texture orientation fields를 적용했습니다.
    *   **"FabSoften: face beautification via dynamic skin smoothing, guided feathering and texture restoration"**: 미세한 디테일을 복원하기 위해 wavelet-based skin texture restoration 방법을 채택했습니다.
*   **Deep Learning-based Retouching**:
    *   **"AutoRetouch: automatic professional face retouching"**: retouching task를 위해 deep image translation을 수행하려는 시도입니다.
    *   **"ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"**: 빠른 local retouching을 위한 adaptive blend pyramid network를 제안했습니다.
    *   **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe): 깨끗한 얼굴을 점진적으로 복원하기 위해 two-stage retouching strategy를 채택했습니다.
    *   **일반적인 한계점**: 이러한 많은 방법들(AutoRetouch 및 ABPN 등)은 global convolution filters를 통한 image-level transformations만 고려하므로, 결함이 있는 영역을 적응적으로 완성(adaptively completed)할 수 없습니다. 또한, 종종 광범위한 pairwise data를 필요로 합니다.

### 2. RetouchFormer의 차별점 및 개선 사항 (Differences and Improvements of RetouchFormer)

RetouchFormer는 위에 언급된 기존 연구들에 비해 몇 가지 핵심적인 개선 사항을 도입했습니다:

*   **Adaptive Local Completion vs. Global Transformation**: global filters를 적용하는 일반적인 image translation 방법(**"MPRNet: Multi-stage progressive image restoration"**)이나 초기 retouching 모델(**"AutoRetouch: automatic professional face retouching"**)과 달리, RetouchFormer는 retouching을 "soft inpainting" task로 취급합니다. 이는 결함 영역을 명시적으로 식별하고 해당 영역만을 위해 콘텐츠를 적응적으로 합성합니다.
*   **Selective Self-Attention Mechanism**: **"RestoreFormer: High-quality blind face restoration from undegraded key-value pairs"**와 같은 모델이 복원을 위해 priors를 사용하는 반면, RetouchFormer는 "selective self-attention" mechanism을 도입합니다. 이 mechanism은 clean face priors를 사용하여 결함 영역 내의 상호작용을 억제하고, 결함 "queries"와 정상 피부 "key-value" pairs 사이의 상호작용을 강화합니다. 이를 통해 local content가 건강한 피부 feature로부터 합성되도록 보장합니다.
*   **Semi-supervised Learning**: 고가의 paired data가 필요한 한계(**"Image-to-image translation with conditional adversarial networks"**에서 보여짐)를 극복하기 위해, RetouchFormer는 semi-supervised settings를 위해 설계되었습니다. 이는 reconstruction-oriented localization module과 adversarial training에 의해 안정화된 paired 및 unpaired data를 모두 활용합니다.
*   **Multi-scale Flexibility**: transformer에서 multi-scale encoding scheme을 채택함으로써, RetouchFormer는 콘텐츠를 계층적으로 표현할 수 있습니다. 이를 통해 fixed-scale tokens나 single-scale filters를 사용하는 방법보다 다양한 scale의 결함을 더 효과적으로 처리할 수 있습니다.
*   **Structural Simplicity and Efficiency**: **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe)와 같은 two-stage 방법들과 비교하여, RetouchFormer는 더 단순한 encoder-transformer-decoder architecture를 채택하여 더 나은 성능과 높은 효율성을 달성합니다 (**Figure 3: Convergence properties of RetouchFormer**에서 보여지듯, 최상의 결과와 일치하면서도 최대 6배 더 빠름).

**SOURCES**

- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Proc. Neural Information Processing Systems, 2014.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- M.-Y. Liu, T. Breuel, and J. Kautz, "Unsupervised image-to-image translation networks," in Proc. Neural Information Processing Systems, 2017.
- J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, "Unpaired image-to-image translation using cycle-consistent adversarial networks," in Proc. International Conference on Computer Vision, 2017.
- T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim, "Learning to discover cross-domain relations with generative adversarial networks," in Proc. International Conference on Machine Learning, 2017.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M. H. Yang, and L. Shao, "MPRNet: Multi-stage progressive image restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- L. Goetschalckx, A. Andonian, A. Oliva, and P. Isola, "GANalyze: toward visual definitions of cognitive image properties," in Proc. International Conference on Computer Vision, 2019.
- A. Jahanian, L. Chai, and P. Isola, "On the “steerability” of generative adversarial networks," in arXiv:1907.07171, 2019.
- E. Collins, R. Bala, B. Price, and S. Susstrunk, "Editing in style: uncovering the local semantics of GANs," in Proc. IEEE conference on Computer Vision and Pattern Recognition, 2020.
- Y. Shen and B. Zhou, "Closed-form factorization of latent semantics in GANs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- Z. Wu, D. Lischinski, and E. Shechtman, "StyleSpace analysis: disentangled controls for StyleGAN image generation," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- J. Gu, Y. Shen, and B. Zhou, "Image processing using multi-code GAN prior," in Proceedings of IEEE conference on computer vision and pattern recognition, 2020.
- E. Richardson, Y. Alaluf, Y. Patashnik, Y. Nitzan, Y. Azar, S. Shapiro, and D. Cohen-Or, "Encoding in style: a StyleGan encoder for image-to-image translation," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- O. Tov, Y. Alaluf, Y. Nitzan, Y. Patashnik, and D. Cohen-Or, "Designing an encoder for StyleGan image manipulation," ACM Transactions on Graphics, 2021.
- R. Abdal, P. Zhu, N. J. Mitra, and P. Wonka, "StyleFlow: attribute-conditioned exploration of StyleGAN Generated images using conditional continuous normalizing flows," ACM Transactions on Graphics, 2021.
- Y. Patashnik, Z. Wu, E. Shechtman, D. Cohen-Or, and D. Lischinski, "StyleCLIP: text-driven manipulation of StyleGAN imagery," in Proc. International Conference on Computer Vision, 2021.
- J. Devlin, M.-W. Chang, K. Lee, and C. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in arXiv:1801.04805, 2018.
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Proc. Neural Information Processing Systems, 2017.
- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: transformers for image recognition at scale," in International Conference on Learning Representations, 2021.
- H. Wu, B. Xiao, N. Codella, M. Liu, X. Dai, L. Yuan, and L. Zhang, "CvR: Introducing convolutions to vision transformers," arXiv:2103.15808, 2021.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," in Proc. International Conference on Computer Vision, 2021.
- H. Chen, Y. Wang, T. Guo, C. Xu, Y. Deng, Z. Liu, M. Ma, C. Xu, C. Xu, and W. Gao, "Pre-trained image processing transformer," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- X. Hu, Q. Huang, Z. Shi, S. Li, C. Gao, L. Sun, and Q. Li, "Style transformer for image inversion and editing," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in Proc. European Conference on Computer Vision, 2020.
- X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, "Deformable DETR: Deformable transformers for end-to-end object detection," in International Conference on Learning Representations, 2021.
- K. Arakawa, "Nonlinear digital filters for beautifying facial images in multimedia systems," in Proc. IEEE International Symposium on Circuits and Systems, 2004.
- N. Batool and R. Chellappa, "Detection and inpainting of facial wrinkles using texture orientation fields and Markov random field modeling," IEEE Transactions on Image Processing, 2014.
- U. Lipowezky and S. Cahen, "Automatic freckles detection and retouching," in Proc. IEEE Convention of Electrical and Electronics Engineers in Israel, 2008.
- T. Leyvand, D. Cohen-Or, G. Dror, and D. Lischinski, "Data-driven enhancement of facial attractiveness," ACM Transactions on Graphics, 2008.
- S. Velusamy, R. Parihar, R. Kini, and A. Rege, "FabSoften: face beautification via dynamic skin smoothing, guided feathering and texture restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition Workshop, 2020.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- B. Lei, X. Guo, Y. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.

# Methodology
## Preliminaries
>[!question]
>1. 이 논문에서 제시하는 방법을 이해하기 위해 필요한 주요 용어·기호는 무엇인지, 각각의 의미와 함께 설명해주십시오.
>2. 이 논문에서 제시하는 방법을 이해하기 위한 수학적 배경(수식·이론)을 차근차근 풀어서 설명해 주세요.
>3. 이 논문에서 제시하는 방법을 이해하기 위한 필수적인 사전 연구(선행 논문)를 알기 쉽게 설명하고 정리해 주세요.
>4. 이 개념들이 이후 모델 설명과 어떻게 연결되는지 밝혀주십시오.

>[!answer]


### 1. Key Terms and Symbols

RetouchFormer의 methodology를 이해하기 위해서는 논문 전반에 걸쳐 사용되는 다음의 terms와 symbols를 정의하는 것이 필수적입니다:

- **$X$**: 전체 training data 세트로, 소량의 paired data $X_a$와 대량의 unpaired data $X_u$로 구성됩니다 (즉, $X = X_a \cup X_u$). 이는 "Proposed Approach"의 "Overview" section에 자세히 설명되어 있습니다.
- **$X_a = \{(x_a, y_a)\}$**: paired raw-retouching images의 subset으로, 여기서 $x_a$는 raw image이고 $y_a$는 전문가가 리터칭한 ground truth입니다.
- **$X_u = \{x_u, y_u\}$**: unpaired images의 subset으로, 여기서 $x_u$와 $y_u$는 서로 다른 identity에 대응할 수 있으며 직접적인 mapping이 존재하지 않습니다.
- **$\gamma$**: paired training data의 비율로, $\gamma = \|X_a\| / \|X\|$로 정의됩니다. 이는 "Semi-supervised Settings" section에서 언급된 바와 같이, 다양한 수준의 supervision 하에서 모델의 성능을 평가하는 데 사용됩니다.
- **$P = \{\rho_1, \rho_2, \dots, \rho_K\}$**: "clean face priors"를 포착하기 위해 학습된 feature dictionary입니다. 이는 encoder space에서 features를 재구성하는 데 사용되는 $K$개의 elements로 구성됩니다.
- **$E, T, G$**: architecture의 핵심 구성 요소인 Encoder ($E$), Transformer ($T$), 그리고 Decoder ($G$)입니다.
- **$H$ 및 $\bar{H}$**: "Soft Mask Generation"에 사용되는 U-Net structure의 두 가지 streams입니다. $\bar{H}$는 selective self-attention에 사용되는 최종 imperfection mask를 생성합니다.
- **$f_H(x_-)$**: mask $\bar{H}$에서 유도된 weighting map으로, 결점(imperfections)의 위치와 강도를 나타냅니다.
- **$Q, K, V$**: self-attention mechanism에서 사용되는 Query, Key, Value matrices입니다.
- **$\otimes$**: Hadamard product (element-wise multiplication)로, imperfection mask를 기반으로 features에 가중치를 부여하는 데 사용됩니다.
- **$\phi(\cdot, \cdot)$**: mask evaluation에서 사용되는 Euclidean distance와 같이 각 pixel 위치에서의 image difference를 측정하는 함수입니다.

이러한 용어들은 semi-supervised framework와 imperfection localization 및 synthesis를 위한 구체적인 메커니즘을 설명하기 위해 "Proposed Approach" 및 "RetouchFormer" section에서 도입되었습니다.

---

### 2. Mathematical Background and Step-by-Step Explanation

RetouchFormer 방법론은 prior learning, mask generation, 그리고 selective attention에 주로 초점을 맞춘 몇 가지 수학적 토대 위에 구축되었습니다.

#### Step 1: Learning Clean Face Priors via Vector Quantization
모델은 먼저 깨끗한 피부 features를 표현하기 위해 dictionary $P$를 학습합니다. encoded image $E(y_-)$의 $(i, j)$ 위치에 있는 주어진 feature vector에 대해, dictionary에서 가장 가까운 element를 다음을 사용하여 찾습니다:
$$\color{orange}
\rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(y_-)|(i,j) - \rho_k\|_2^2
$$
"Clean Face Priors"에서 설명된 이 과정을 통해 모델은 $\tilde{x}_- = R(E_\rho(x_-))$로 표시되는 얼굴의 "clean" 버전을 재구성할 수 있으며, 여기서 $R$은 reconstruction decoder입니다.

#### Step 2: Soft Mask Generation
결점을 식별하기 위해 raw image $x_-$와 재구성된 이미지 $\tilde{x}_-$를 concatenation하여 two-stream U-Net ($H$ 및 $\bar{H}$)에 통과시킵니다. paired data의 경우, mask는 raw image와 retouched image 사이의 실제 차이에 대해 평가됩니다:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)}\|\bar{H}(x_a, x_a) - \phi(x_a, y_a)\|_2^2
$$
unpaired data의 경우, 예측의 신뢰성을 보장하기 위해 confidence-based loss가 사용됩니다:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)} [-I_{\{\phi(x_a,y_a)>0\}} \log \sigma(H(x_a, x_a))] + \lambda \mathbb{E}_{x_u} [-\sigma(H(x_u, x_u)) \log \sigma(H(x_u, x_u))]
$$
"Soft Mask Generation"에서 설명된 바와 같이, $\bar{H}$의 parameters는 $H$의 Exponential Moving Average (EMA)를 사용하여 업데이트됩니다:
$$\color{orange}
\theta_{\bar{H}} \leftarrow \mu\theta_{\bar{H}} + (1 - \mu)\theta_H
$$

#### Step 3: Selective Self-Attention
핵심적인 혁신은 selective self-attention mechanism입니다. 표준 attention과 달리, weighting map $f_H(x_-)$를 사용하여 Queries는 결점에, Keys와 Values는 정상 피부에 집중하도록 합니다:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
attended feature map $\Delta f_E$는 다음과 같이 계산됩니다:
$$\color{orange}
\Delta f_E = \text{softmax}(QK^T / \sqrt{\Lambda})V
$$
마지막으로, 결점 영역의 features를 합성된 "normal" features로 교체하여 features를 업데이트합니다:
$$\color{orange}
\hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
$$
"Selective Self-attention"에서 볼 수 있는 이 공식은 결점 영역의 content가 건강한 피부 영역의 context로부터 합성되도록 보장합니다.

---

### 3. Essential Prior Work

이 논문은 computer vision의 여러 핵심 연구 분야를 기반으로 합니다:

- **CNN-based Image Translation**: 이 논문은 source domains를 target domains로 매핑하는 기초적인 방법으로 "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix)를 참조합니다. 또한 고품질 image synthesis를 위해 "Multi-stage progressive image restoration" (MPRNet) 및 "GAN prior embedded network for blind face restoration in the wild" (GPEN)을 언급합니다.
- **Vision Transformers**: architecture는 "An image is worth 16x16 words: transformers for image recognition at scale" (ViT) 및 "Swin transformer: hierarchical vision transformer using shifted windows" (SwinTransformer)에서 영감을 받았습니다. 특히, face restoration을 위해 dictionary 기반 접근 방식을 사용하는 "RestoreFormer: High-quality blind face restoration from degraded key-value pairs"와 관련이 있습니다.
- **Face Retouching**: 이전의 전문화된 방법에는 "AutoRetouch: automatic professional face retouching" 및 "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"가 포함됩니다. 또한 이 논문은 two-stage restoration 전략을 사용하는 "Blemish-Aware and Progressive Face Retouching With Limited Paired Data" (BPFRe)와 스스로를 비교합니다.
- **Vector Quantization (VQ)**: clean face priors를 학습하는 방법은 discrete latent representations를 학습하기 위한 VQ-VAE framework를 도입한 "Neural discrete representation learning"에 기반합니다.
- **Semi-supervised Learning**: 안정적인 mask generation을 위한 EMA의 사용은 "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"에서 영감을 받았습니다.

---

### 4. Connection to the Model Description

이러한 concepts는 다음과 같이 RetouchFormer architecture에 체계적으로 통합됩니다:

1.  **Prior Learning to Localization**: "Neural discrete representation learning"의 **Vector Quantization** 개념이 feature dictionary $P$를 생성하는 데 사용됩니다. dictionary에는 "clean"한 얼굴 features만 포함되어 있기 때문에, input과 dictionary 기반 reconstruction 사이의 reconstruction error는 **Soft Mask Generation** module이 결점을 식별할 수 있는 "clue"를 제공합니다.
2.  **Localization to Synthesis**: 예측된 mask $f_H(x_-)$는 **Selective Self-Attention** mechanism을 직접 조절합니다. mask를 사용하여 Queries(결점)를 정의하고 mask의 역(inverse)을 사용하여 Keys 및 Values(정상 피부)를 정의함으로써, 모델은 Transformer가 건강한 피부 영역의 정보를 사용하여 잡티를 "채우도록(fill in)" 강제합니다. 이는 "Introduction"에서 설명된 inpainting task의 "soft" 버전입니다.
3.  **Multi-scale Representation**: **SwinTransformer**에서 착안하여, 모델은 "Selective Self-attention"에서 논의된 바와 같이 다양한 크기의 결점(예: 작은 점 vs 큰 여드름 부위)을 처리하기 위해 multi-scale feature tokens ($f_E$)를 사용합니다.
4.  **Semi-supervised Stability**: "Mean teachers are better role models"의 **EMA** update rule은 제한된 paired data ($X_a$)로 학습할 때에도 imperfection prediction이 안정적으로 유지되도록 보장하여, 모델이 대량의 unpaired data ($X_u$)를 효과적으로 활용할 수 있게 합니다.
5.  **Adversarial Refinement**: 마지막으로, "Generative adversarial nets"의 **GAN** framework가 예측된 mask를 수신하는 discriminator $D$를 통해 채택되어, generator가 실제 "clean"한 피부와 구별할 수 없는 사실적인 피부 질감을 생성하도록 강제합니다.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from degraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," Advances in Neural Information Processing Systems, 2017.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Neural Information Processing Systems, 2014.
- S. W. Zamir, A. Arora, L. Khan, S. Hayat, M. Khan, F. S. Yang, and L. Shao, "Multi-stage progressive image restoration," IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: transformers for image recognition at scale," International Conference on Learning Representations, 2021.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," International Conference on Computer Vision, 2021.


## Framework
>[!question]
>1. 전체 시스템 또는 프레임워크가 어떤 모듈/블록으로 구성되어 있는지 다음의 항목들을 포함하여 자세히 설명해주십시오.
>    - 전체 시스템 또는 프레임워크의 모든 구성요소의 Baseline 또는 Backbone 과 그에 대응하는 refereces 및 선택 이유.
>    - 각 구성요소에서 저자들이 주장하는 구조적 개선점과 최종 네트워크 구조(레이어 구성, 주요 모듈, 입력과 출력 등) 및 역할과 기능
>2. 전체 시스템 또는 프레임워크의 입력과 출력, 각 구성요소 간 연결 과정(데이터 흐름)을 단계별로 설명해주십시오.
>3. 프레임워크나 모델 아키텍처를 나타내는 그림(Figure)이 있다면, 그 위치(번호)와 함께 그림에서 중요한 부분들을 구체적으로 소개해 주세요.

>[!answer]


RetouchFormer는 고품질 face retouching을 위해 설계된 transformer-based framework입니다. 이 시스템은 semi-supervised approach를 사용하여 facial imperfections를 식별하고 해당 영역에 사실적인 skin textures를 합성하는 방식으로 작동합니다. 이 시스템은 encoder-transformer-decoder architecture를 기반으로 구축되었으며, prior-based selective self-attention mechanism으로 강화되었습니다.

### 1. Structure of the Framework

RetouchFormer framework는 retouching 과정에서 각자 특정한 역할을 수행하는 여러 상호 연결된 modules로 구성됩니다. "Figure 2: The workflow of RetouchFormer"에서 묘사된 바와 같이, 시스템은 다음과 같은 components를 포함합니다:

#### A. Encoder ($E$) 및 Multi-scale Feature Extraction
encoder $E$는 input image $x_-$로부터 features를 추출하는 역할을 합니다. 고정된 scale을 사용하는 표준 transformers와 달리, RetouchFormer는 **multi-scale encoding scheme**을 채택합니다.
- **Baseline/Backbone**: 이 multi-scale approach는 "Swin transformer: hierarchical vision transformer using shifted windows"에서 볼 수 있는 hierarchical representations에서 영감을 받았습니다.
- **Reason for Choice**: 이를 통해 model이 local regions의 콘텐츠를 계층적으로 표현할 수 있는데, 이는 acne나 pockmarks와 같은 facial imperfections의 scale이 상당히 다양하기 때문에 매우 중요합니다.
- **Structural Improvement**: encoder는 서로 다른 scale의 imperfections를 효과적으로 처리하기 위해 여러 intermediate layers에서 spatial patches를 추출합니다.

#### B. Feature Dictionary ($P$) 및 Reconstruction Decoder ($R$)
이 module은 imperfections를 식별하는 데 도움이 되는 "clean face priors"를 캡처합니다.
- **Baseline/Backbone**: 이 component는 "Neural discrete representation learning"에서 설명된 **Vector Quantization (VQ)**를 기반으로 합니다.
- **Reason for Choice**: VQ는 model이 고품질의 깨끗한 skin textures를 나타내는 discrete latent space (dictionary)를 학습할 수 있게 해줍니다.
- **Role**: dictionary $P$는 깨끗한 retouching images의 features를 재구성하도록 최적화됩니다. 결점이 있는 raw image가 통과할 때, dictionary-based reconstruction인 $\tilde{x}_- = R(E_\rho(x_-))$에는 해당 결점들이 결여되어 나타나므로, localization을 위한 단서를 제공합니다.

#### C. Soft Mask Generation Network ($\{H, \bar{H}\}$)
이 module은 imperfections의 위치를 예측합니다.
- **Baseline/Backbone**: "Image-to-Image Translation with Conditional Adversarial Networks"에서 확인되는 image-to-image translation tasks의 일반적인 선택인 **U-Net** 구조를 활용합니다.
- **Structural Improvement**: 저자들은 **two-stream structure** ($H$ 및 $\bar{H}$)를 제안합니다. stream $\bar{H}$는 "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"의 기술인 $H$의 **Exponential Moving Average (EMA)**를 사용하여 업데이트됩니다.
- **Role**: raw image $x_-$와 reconstructed image $\tilde{x}_-$를 input으로 받아 retouching이 필요한 영역을 나타내는 soft mask $\bar{H}(x_-)$를 생성합니다.

#### D. Transformer ($T$)와 Selective Self-Attention Blocks (SSAB)
transformer $T$는 imperfection regions의 features를 normal skin features로 변환하는 핵심 처리 unit입니다.
- **Baseline/Backbone**: transformer architecture는 "Attention is all you need" 및 "An image is worth 16x16 words: transformers for image recognition at scale"을 기반으로 합니다.
- **Structural Improvement**: 저자들은 **Selective Self-Attention (SSA)** 메커니즘을 도입합니다. SSA는 global attention 대신 predicted mask를 사용하여 spatial interactions를 제한합니다.
- **Role**: imperfection regions로부터의 정보를 억제하고 이를 normal skin regions에서 집계된 features로 대체합니다.

#### E. Synthesis Decoder ($G$) 및 Discriminator ($D$)
- **Synthesis Decoder ($G$)**: 이 component는 transformer로부터 변환된 features를 받아 최종적인 고품질 retouching image $\hat{y}^x_-$를 합성합니다.
- **Discriminator ($D$)**: "Generative adversarial nets"를 기반으로 하는 discriminator는 pixel-level real-fake identification을 수행합니다.
- **Structural Improvement**: 결과를 개선하기 위해 predicted masks $\bar{H}(x_-)$가 discriminator에 주입되어, imperfection regions에서 합성된 세부 사항에 더 많은 주의를 기울이도록 강제합니다.

---

### 2. Data Flow

RetouchFormer를 통한 data flow는 raw image를 retouched image로 변환하기 위해 체계적인 경로를 따릅니다:

1.  **Feature Extraction**: raw input image $x_-$가 encoder $E$를 통과하여 multi-scale features $f_E(x_-)$를 생성합니다.
2.  **Prior-based Reconstruction**: encoder features $E(x_-)$는 dictionary $P$를 사용하여 quantized됩니다. 가장 가까운 dictionary element $\rho_*$는 다음을 통해 찾습니다:
    $$\color{orange}
    \rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(y_-)|(i,j) - \rho_k\|_2^2
    $$
    그 후 reconstruction decoder $R$이 "clean" reconstructed image $\tilde{x}_- = R(E_\rho(x_-))$를 생성합니다.
3.  **Imperfection Localization**: raw image $x_-$와 reconstructed image $\tilde{x}_-$가 concatenation되어 mask generation network $\{H, \bar{H}\}$에 입력되며, soft mask $\bar{H}(x_-)$와 그에 해당하는 weighting map $f_H(x_-)$를 생성합니다.
4.  **Selective Transformation**: Transformer $T$ 내부에서 features $f_E(x_-)$와 weighting map $f_H(x_-)$를 사용하여 Queries ($Q$), Keys ($K$), 그리고 Values ($V$)를 계산합니다:
    $$\color{orange}
    Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
    $$
    $$\color{orange}
    K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
    $$
    $$\color{orange}
    V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
    $$
    그 후 attention mechanism은 normal skin의 context를 사용하여 imperfection regions를 위한 새로운 features $\Delta f_E$를 합성합니다. 최종적으로 변환된 features $\hat{f}_E(x_-)$는 다음과 같이 형성됩니다:
    $$\color{orange}
    \hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
    $$
5.  **Final Synthesis**: 변환된 features가 decoder $G$를 통과하여 최종 output image $\hat{y}^x_- = G(\hat{f}_E(x_-))$를 생성합니다.

---

### 3. Framework Figure

architecture의 주요 그림은 "Figure 2: The workflow of RetouchFormer"입니다. 이 그림은 다음의 세부 사항을 보여줍니다:

-   **Left Side**: **Encoder $E$**로 들어가는 input image $x_-$를 보여줍니다. $\tilde{x}_-$를 생성하기 위해 **Dictionary $P$**와 **Reconstruction Decoder $R$**로 이어지는 경로를 묘사합니다.
-   **Center-Left**: source $x$와 reconstruction $\tilde{x}$를 받아 mask $\bar{H}(x)$를 생성하는 **Soft mask generation network**를 보여줍니다.
-   **Center**: 여러 개의 **Selective Self-Attention Blocks (SSAB)**로 구성된 **Transformer $T$**를 표시합니다. multi-scale features $f_E(x)$와 weights $f_H(x)$가 어떻게 처리되는지 보여줍니다.
-   **Right Side**: output $\hat{y}^x$를 생성하는 **Decoder $G$**를 보여줍니다. 또한 output image, mask, original image를 받아 real-fake prediction을 수행하는 **Discriminator $D$**를 보여줍니다.
-   **Inset (Top Right)**: **Selective Self-Attention-based Block (SSAB)**의 상세 뷰를 제공하며, imperfection features (Queries)를 normal skin features (Keys 및 Values)와 분리하는 Hadamard product ($\otimes$) 연산을 강조합니다.

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," International Conference on Computer Vision, 2021.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," Advances in Neural Information Processing Systems, 2017.
- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, x. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: transformers for image recognition at scale," International Conference on Learning Representations, 2021.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Neural Information Processing Systems, 2014.
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," Neural Information Processing Systems, 2017.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.


## Training
>[!question]
>1. Training은 어떤 단계로 이뤄지며(예: 사전 학습 후 파인튜닝, 단계별 Loss 결합, 커리큘럼 러닝 등), 각 단계의 역할과 의미는 무엇인지 알려주세요.
>2. 커스텀 손실 함수를 포함하여 각 단계에서 사용되는 손실 함수는 무엇이고, 어떤 출력에 적용되고 Training 대상은 무엇인지, 목적과 기능은 무엇인지 수식을 들어 설명해 주세요. 
>3. 어떤 최적화(Optimization) 기법과 하이퍼파라미터(러닝 레이트 스케줄 등)를 사용했는지, 어떻게 설정했는지 자세히 적어 주세요.
>4. 커리큘럼 러닝, 멀티태스크 러닝, 자기지도 학습, 반지도 학습, 정규화(Regularization)와 같은 특별 학습 기법이 적용되었다면, 각각의 목적과 방식을 자세히 밝혀 주세요.

>[!answer]


RetouchFormer의 training process는 prior learning, imperfection localization, 그리고 high-fidelity image synthesis를 통합하는 정교한 semi-supervised framework입니다. 저자들은 서로 다른 constituent networks에 걸쳐 여러 loss terms의 균형을 맞춤으로써 시스템을 최적화합니다.

### 1. Training Process

RetouchFormer의 training은 (pre-training 후 fine-tuning과 같은) 순차적인 시간적 단계로 명시적으로 나뉘지 않고, 대신 prior-based reconstruction, imperfection prediction, 그리고 retouching evaluation이라는 세 가지 주요 측면의 **joint optimization**으로 구조화됩니다. "Model Training" 섹션에 명시된 바와 같이, constituent networks는 서로 다른 loss terms로 최적화되며, 오직 encoder $E$만이 task 간에 공유되고 다른 network들은 개별 task에 전용됩니다.

training process의 세 가지 핵심 측면은 다음과 같습니다:
- **Prior-based Reconstruction**: 이 측면은 feature dictionary를 사용하여 "clean face prior"를 학습하는 데 중점을 둡니다. encoder $E$와 dictionary $P$는 retouching images $y_-$의 features를 reconstruct하도록 최적화됩니다. 이 과정은 "Neural discrete representation learning"의 **Vector Quantization (VQ)** algorithm에서 영감을 받았습니다.
- **Imperfection Prediction**: 이는 결점을 식별하기 위해 mask generation networks $H$와 $\bar{H}$를 학습시키는 것을 포함합니다. 저자들은 paired data가 직접적인 supervision을 제공하고, unpaired data는 confidence-based loss를 통해 처리되는 semi-supervised approach를 사용합니다.
- **Retouching Evaluation**: 이 측면은 최종 synthesized image $\hat{y}^x_-$가 (paired data의 경우) ground truth와 일치하고 (adversarial training을 통해) 높은 realism을 유지하도록 보장합니다.

전체 optimization formulation은 "Model Training" 섹션에 다음과 같이 표현됩니다:
$$\color{orange}
\min_{P,R} L_{prior} + L_{con}^{rec},
$$
$$\color{orange}
\min_{H} L_{\bar{H}} + L_H,
$$
$$\color{orange}
\min_{E,T,G} L_{prior} + L_{con}^{rec} + L_{con}^{x2y} + L_{adv}^{syn},
$$
$$\color{orange}
\max_{D} L_{adv}^{disc}.
$$

### 2. Loss Functions

RetouchFormer는 framework의 다양한 components를 최적화하기 위해 다양한 loss functions를 활용합니다.

#### A. Prior-based Reconstruction Losses
고품질의 clean face priors를 포착하기 위해, encoder $E$와 dictionary $P$는 다음을 사용하여 최적화됩니다:
- **Prior Loss ($L_{prior}$)**: 이 loss는 dictionary가 clean images $y_- \in \{ya, yu\}$의 features를 정확하게 표현할 수 있도록 보장합니다.
$$\color{orange}
L_{prior} = \mathbb{E}_{y_-} \left[ \sum_{(i,j)} \|E(y_-)|_{(i,j)} - \rho_*|_{(i,j)}\|_2^2 \right]
$$
- **Reconstruction Consistency Loss ($L_{con}^{rec}$)**: 고품질 reconstruction을 보장하기 위해, $L_1$ loss와 perceptual loss (pre-trained VGG-19 network 사용)의 조합이 적용됩니다:
$$\color{orange}
L_{con}^{rec} = \mathbb{E}_{y_-} [\eta\|y_- - \tilde{y}_-\|_1 + \|V(y_-) - V(\tilde{y}_-)\|_2^2]
$$
여기서 $\tilde{y}_- = R(E_\rho(y_-))$ 이고 $V(\cdot)$는 VGG-19 features를 나타냅니다. 이는 "Model Training" 섹션에 자세히 설명되어 있습니다.

#### B. Imperfection Prediction Losses
mask generation network $H$는 결점을 식별하도록 학습됩니다:
- **Paired Mask Loss ($L_{\bar{H}}$)**: paired data의 경우, $\bar{H}$의 output을 raw image와 retouched image 사이의 실제 차이와 비교합니다:
$$\color{orange}
L_{\bar{H}} = \mathbb{E}_{(x_a,ya)} \|\bar{H}(x_a, x_a) - \phi(x_a, ya)\|_2^2
$$
- **Confidence-based Loss ($L_H$)**: paired 및 unpaired data 모두에 대해, 이 loss는 확신 있는 predictions를 강제합니다:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,ya)} [-I_{\{\phi(x_a,ya)>0\}} \log \sigma(H(x_a, x_a))] + \lambda \mathbb{E}_{x_u} [-\sigma(\bar{H}(x_u, x_u)) \log \sigma(H(x_u, x_u))]
$$
이 접근 방식은 "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"의 **Exponential Moving Average (EMA)** 기법에서 영감을 받았습니다.

#### C. Retouching Evaluation Losses
- **Paired Consistency Loss ($L_{con}^{x2y}$)**: synthesized image $\hat{y}^x_a$와 ground truth $y_a$ 사이의 일관성 정도를 측정합니다:
$$\color{orange}
L_{con}^{x2y} = \mathbb{E}_{(x_a,ya)} [\zeta\|y_a - \hat{y}^x_a\|_1 + \|V(y_a) - V(\hat{y}^x_a)\|_2^2]
$$
- **Adversarial Losses ($L_{adv}^{syn}$ 및 $L_{adv}^{disc}$)**: "Generative adversarial nets"에 기반하여, 이 loss들은 realism을 향상시킵니다. discriminator $D$는 pixel-level real-fake identification을 수행하며, retouched regions에 주의를 집중시키기 위해 imperfection mask $\bar{H}$가 주입됩니다:
$$\color{orange}
L_{adv}^{syn} = \mathbb{E}_{x_-} [\log(1 - D(\hat{y}^x_-, \bar{H}(x_-)))]
$$
$$\color{orange}
L_{adv}^{disc} = \mathbb{E}_{y_-} [\log D(y_-, \bar{H}(y_-))] + \mathbb{E}_{x_-} [\log(1 - D(\hat{y}^x_-, \bar{H}(x_-))) + \log(1 - D(x_-, \bar{H}(x_-)))]
$$

### 3. Optimization

최적화 세부 사항은 "Implementation Details" 섹션에 제공됩니다:
- **Optimizer**: parameters는 "Adam: a method for stochastic optimization"에서 제안된 **Adam optimizer**를 사용하여 업데이트됩니다.
- **Learning Rate**: 초기 learning rate는 $2e^{-4}$로 설정됩니다.
- **Schedule**: learning rate는 **cosine decay schedule**을 사용하여 수정됩니다.
- **Iterations and Batch Size**: 모델은 총 **300k iterations** 동안 학습되며, 각 batch는 **single image**를 포함합니다.
- **Hyperparameters**: weighting factors는 $\lambda = 0.5$, $\mu = 0.99$, $\eta = 10$, 그리고 $\zeta = 10$으로 설정됩니다. dictionary $P$는 $K = 1024$ elements를 포함합니다.

### 4. Special Training Techniques

RetouchFormer는 성능을 향상시키기 위해 몇 가지 특별한 기법을 채용합니다:

- **Semi-supervised Learning**: 모델은 비용이 많이 드는 paired data에 대한 의존도를 줄이도록 설계되었습니다. 저자들은 비율 $\gamma = \|X_a\| / \|X\|$를 정의하고, 매우 낮은 $\gamma$ (예: 0.01)에서도 모델이 안정적으로 유지됨을 입증합니다.
- **Exponential Moving Average (EMA)**: unpaired images에 대해 더 신뢰할 수 있는 imperfection masks를 생성하기 위해, stream $\bar{H}$의 parameters는 $H$의 moving average로 업데이트됩니다:
$$\color{orange}
\theta_{\bar{H}} \leftarrow \mu\theta_{\bar{H}} + (1 - \mu)\theta_H
$$
이 기법은 "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"에 기반합니다.
- **Vector Quantization (VQ)**: clean face prior dictionary $P$를 학습하는 데 사용되며, 모델이 리터칭을 시도하기 전에 건강한 피부가 어떻게 보이는지 "알 수 있게" 합니다. 이는 "Neural discrete representation learning"에 기반합니다.
- **Multi-task Learning**: framework는 imperfection prediction과 local content synthesis를 공동으로 수행하며, 여기서 두 task는 selective self-attention mechanism을 통해 서로 정보를 주고받습니다.

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," Advances in Neural Information Processing Systems, 2017.
- D. P. Kingma and J. L. Ba, "Adam: a method for stochastic optimization," International Conference on Learning Representations, 2015.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Neural Information Processing Systems, 2014.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.


## Inference and Application
>[!question]
>1. 전체 시스템 혹은 프레임워크의 추론(Inference)은 어떻게 이뤄지나요? Inference 단계의 입력과 출력, 데이터 흐름을 구체적으로 서술해 주세요.
>2. 논문에서 제시하는 실제 응용 사례(Use Case)가 있다면, 모든 예시를 빠짐없이 자세히 설명해 주세요.
>3. 또한, 저자들이 강조하는 접근법의 실용적 장점(실시간 처리, 메모리 효율, 확장성 등)을 정리해 주십시오.

>[!answer]


RetouchFormer의 inference process는 결함이 있는 face image를 고품질의 retouched version으로 변환하는 간소화된 end-to-end operation으로 설계되었습니다. 학습된 clean face priors와 selective self-attention mechanism을 활용하여, model은 blemish regions를 적응적으로 식별하고 realistic skin textures로 채울 수 있습니다.

### 1. Inference Process

Inference 단계에서 model은 encoder-transformer-decoder architecture를 통한 feed-forward path를 따릅니다. 이 과정은 다음과 같은 단계로 나눌 수 있습니다:

**Step 1: Multi-scale Feature Extraction**
시스템의 input은 여드름, 반점 또는 다크서클과 같은 다양한 결함을 포함할 수 있는 raw face image $x_-$입니다. 이 이미지는 encoder $E$를 통과하며 multi-scale features $f_E(x_-)$를 추출합니다. "Introduction"에서 언급했듯이, 이러한 multi-scale tokens는 다양한 크기의 결함을 처리하는 데 필요한 유연성을 제공합니다.

**Step 2: Imperfection Localization via Prior-based Reconstruction**
결함의 위치를 식별하기 위해 model은 학습된 feature dictionary $P$를 활용합니다. Encoder features는 dictionary에서 가장 가까운 "clean" elements를 찾기 위해 quantized됩니다:
$$\color{orange}
\rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(x_-)|(i,j) - \rho_k\|_2^2
$$
그 후 reconstruction decoder $R$은 이러한 quantized features를 사용하여 reconstructed "clean" image $\tilde{x}_-$를 생성합니다. Raw image $x_-$와 reconstruction $\tilde{x}_-$는 concatenation되어 soft mask generation network $\bar{H}$로 전달되며, 결함의 위치와 강도를 나타내는 weighting map $f_H(x_-)$를 생성합니다. 이 workflow는 "Figure 2: The workflow of RetouchFormer"에 시각적으로 표현되어 있습니다.

**Step 3: Selective Feature Transformation**
Multi-scale features $f_E(x_-)$와 weighting map $f_H(x_-)$는 Transformer $T$로 입력됩니다. Selective Self-Attention Blocks (SSAB) 내부에서 features가 변환됩니다. Queries $Q$는 결함 영역에 집중되는 반면, keys $K$와 values $V$는 정상적인 피부 영역에서 파생됩니다:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
Attention mechanism은 blemish areas에 대해 합성된 "normal" skin 정보를 포함하는 residual feature map $\Delta f_E$를 계산합니다. 이후 features는 다음과 같이 업데이트됩니다:
$$\color{orange}
\hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
$$
"Selective Self-attention" 섹션에 있는 이 식은 model이 결함 정보를 억제하고 이를 contextually relevant skin textures로 대체하는 방법을 보여줍니다.

**Step 4: Final Image Synthesis**
변환된 features $\hat{f}_E(x_-)$는 synthesis decoder $G$를 통과하여 최종 retouched output image $\hat{y}^x_-$를 생성합니다. Decoder는 합성된 세부 사항이 보존된 facial structures와 매끄럽게 통합되도록 보장합니다.

### 2. Use Case

논문은 주로 자동화된 professional-grade portrait editing에 초점을 맞춘 RetouchFormer의 여러 실제 응용 시나리오를 식별합니다:

- **Blemish Removal**: 주요 use case는 "Introduction" 및 "Qualitative Results on FR-wild"에서 언급된 "moderate-to-severe acne regions", "pockmarks", "spots with small scales"를 포함한 다양한 face imperfections를 제거하는 것입니다.
- **Skin Smoothing and Beautification**: 단순히 반점을 제거하는 것을 넘어, model은 "erasing dark circles and smoothing skin"을 통해 "flawless skin tone"을 만드는 데 사용되며, 이는 일반적으로 전문 리터처가 수동으로 작업하는 데 수 시간이 걸리는 일입니다.
- **Handling Diverse Real-World Data**: Model은 통제되지 않은 환경에서 촬영된 사진에 대한 유용성을 입증하기 위해 "different types of facial blemishes"를 가진 "1,000 in-the-wild face images"를 포함하는 "FR-wild" dataset에서 테스트되었습니다.
- **High-Resolution Retouching**: 이 framework는 "clean face images with high fidelity"를 합성하도록 설계되어 고품질 사진 응용 분야에 적합합니다.

### 3. Practical Advantages

저자들은 기존 state-of-the-art methods 대비 RetouchFormer의 몇 가지 실질적인 장점을 강조합니다:

- **Efficiency in Training and Convergence**: "Figure 3: Convergence properties of RetouchFormer"에서 보여주듯이, model은 base model의 최고 결과와 일치하는 수준에 최대 **6배 더 빠르게** 도달하며 더 높은 전체 PSNR을 달성합니다.
- **Reduced Data Requirements**: 과업을 semi-supervised "soft inpainting" 문제로 공식화함으로써, RetouchFormer는 획득 비용이 높고 시간이 많이 걸리는 paired images에 대한 "training data collection" 요구 사항을 크게 줄입니다. 이는 "Proposed Approach"의 "Overview"에서 강조되었습니다.
- **Structural Simplicity**: "Blemish-Aware and Progressive Face Retouching With Limited Paired Data" (BPFRe)와 같은 two-stage methods와 비교하여, RetouchFormer는 "Vision Transformer" 섹션에서 언급된 바와 같이 "adopts a simpler structure and achieves better retouching performance"를 보여줍니다.
- **Scale Flexibility**: Multi-scale feature tokens의 사용은 model이 "increased flexibility in dealing with the imperfections at various scales"를 가질 수 있게 하여, 아주 작은 주근깨부터 큰 여드름 패치까지 모두 효과적으로 처리할 수 있도록 합니다.
- **High Fidelity and Realism**: Model은 "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo" 및 "RestoreFormer: High-quality blind face restoration from degraded key-value pairs"와 같은 고급 방법들보다 PSNR, SSIM, LPIPS 측면에서 우수한 성능을 보이며, 특히 필수적인 facial structures를 보존하면서 결함을 제거하는 데 탁월합니다.

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from degraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," International Conference on Computer Vision, 2021.


## Method Summary
>[!question]
>지금까지 제시된 내용을 바탕으로, 프레임워크 구성, 데이터 흐름, 학습 전략, 손실 함수, 추론 및 활용 가능성을 모두 아우르는 풍부한 요약을 작성해 주세요.

>[!answer]


RetouchFormer는 고품질 face retouching을 위해 설계된 transformer-based framework로, 필수적인 facial structures를 보존하면서 다양한 facial imperfections를 제거하는 것을 목표로 합니다. 이 방법론은 face retouching을 face image inpainting의 일반화된 'soft' 버전으로 취급하며, 결점을 식별한 후 주변의 정상적인 피부에서 합성된 realistic skin textures로 대체합니다.

### 1. Framework Structure

RetouchFormer의 architecture는 "Figure 2: The workflow of RetouchFormer"에서 설명된 것처럼 encoder-transformer-decoder 설계를 따릅니다. 시스템은 여러 specialized modules로 구성됩니다:

- **Multi-scale Encoder ($E$)**: 이 module은 여러 intermediate layers에서 input image의 features를 추출합니다. "Swin transformer: hierarchical vision transformer using shifted windows"의 hierarchical representation에서 영감을 받은 이 multi-scale approach는 모델이 작은 점부터 큰 acne regions까지 다양한 크기의 결점을 처리할 수 있게 합니다.
- **Feature Dictionary ($P$) 및 Reconstruction Decoder ($R$)**: "clean face priors"를 캡처하기 위해, 모델은 "Neural discrete representation learning"에서 상세히 설명된 **Vector Quantization (VQ)** 기술을 사용하여 latent dictionary를 학습합니다. 이 dictionary는 고품질 skin textures를 나타냅니다. Reconstruction decoder $R$은 이러한 priors를 사용하여 blemishes를 식별하기 위한 reference 역할을 하는 clean version의 face를 생성합니다.
- **Soft Mask Generation Network ($\{H, \bar{H}\}$)**: 이 component는 imperfection regions를 식별합니다. 이는 two-stream U-Net structure를 사용합니다. Semi-supervised setting에서 안정적인 예측을 보장하기 위해, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"의 전략을 따라 stream $\bar{H}$의 parameters는 $H$의 **Exponential Moving Average (EMA)**를 사용하여 업데이트됩니다.
- **Transformer ($T$) with Selective Self-Attention Blocks (SSAB)**: feature space에서 실제 retouching을 수행하는 모델의 핵심입니다. 이는 blemish features를 억제하고 이를 normal skin features로 대체하기 위해 새로운 selective self-attention mechanism을 사용합니다.
- **Synthesis Decoder ($G$) 및 Discriminator ($D$)**: Decoder $G$는 변환된 features를 다시 high-fidelity image로 변환합니다. "Generative adversarial nets"에 기반한 Discriminator $D$는 pixel-level real-fake identification을 수행하여 output의 realism을 보장하며, predicted mask가 주입되어 discriminator의 attention을 retouched areas에 집중시킵니다.

### 2. Data Flow

RetouchFormer를 통한 data flow는 localization과 synthesis의 체계적인 과정입니다:

1.  **Feature Extraction**: Raw input image $x_-$는 encoder $E$에 의해 처리되어 multi-scale features $f_E(x_-)$를 생성합니다.
2.  **Prior-based Reconstruction**: Features는 dictionary $P$를 사용하여 quantized됩니다. 각 feature vector에 대해 가장 가까운 dictionary element $\rho_*$가 선택됩니다:
    $$\color{orange}
    \rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(y_-)|(i,j) - \rho_k\|_2^2
    $$
    Decoder $R$은 이러한 quantized features로부터 "clean" image $\tilde{x}_-$를 재구성합니다.
3.  **Mask Generation**: Raw image $x_-$와 reconstruction $\tilde{x}_-$가 결합되어 mask network에 입력되고, 결점의 위치를 나타내는 weighting map $f_H(x_-)$를 생성합니다.
4.  **Selective Self-Attention**: Transformer에서 features $f_E(x_-)$와 mask $f_H(x_-)$는 Queries ($Q$), Keys ($K$), Values ($V$)를 계산하는 데 사용됩니다:
    $$\color{orange}
    Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
    $$
    $$\color{orange}
    K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
    $$
    $$\color{orange}
    V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
    $$
    이는 queries가 imperfections를 나타내고 keys/values가 normal skin을 나타내도록 보장합니다. Attended features $\Delta f_E$는 feature map을 업데이트하는 데 사용됩니다:
    $$\color{orange}
    \hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
    $$
5.  **Synthesis**: 변환된 features $\hat{f}_E(x_-)$는 decoder $G$를 통과하여 최종 retouched image $\hat{y}^x_-$를 생성합니다.

### 3. Training Strategies

RetouchFormer는 고가의 paired data에 대한 의존도를 줄이기 위해 **semi-supervised approach**를 사용하여 학습됩니다. Training set $X$는 소규모의 paired images $X_a$ subset과 대규모의 unpaired images $X_u$ subset으로 구성됩니다.

전략은 다음을 포함합니다:
- **Joint Optimization**: Priors를 위한 dictionary, localization을 위한 U-Net, synthesis를 위한 transformer-decoder를 동시에 최적화합니다.
- **EMA for Stability**: Mask generation stream $\bar{H}$를 업데이트하기 위해 EMA를 사용하면 unpaired data $X_u$에 대해 더 신뢰할 수 있는 targets를 제공합니다.
- **Adversarial Training**: 합성된 skin textures가 실제 clean skin의 statistical distribution과 일치하도록 discriminator를 활용합니다.

### 4. Loss Functions

모델은 여러 loss functions의 조합을 사용하여 최적화됩니다:

- **Prior Loss ($L_{prior}$)**: Clean skin features를 나타내도록 dictionary $P$를 최적화합니다.
- **Reconstruction Consistency Loss ($L_{con}^{rec}$)**: Reconstruction decoder $R$이 $L_1$ 및 perceptual losses (VGG-19)를 사용하여 고품질 이미지를 생성할 수 있도록 보장합니다.
- **Mask Losses ($L_{\bar{H}}$ 및 $L_H$)**: $L_{\bar{H}}$는 paired data에 대해 직접적인 supervision을 제공하며, $L_H$는 paired 및 unpaired data 모두에 대한 confidence-based loss입니다.
- **Retouching Consistency Loss ($L_{con}^{x2y}$)**: Paired data에 대해 합성된 output과 ground truth 사이의 유사성을 측정합니다.
- **Adversarial Losses ($L_{adv}^{syn}$ 및 $L_{adv}^{disc}$)**: 합성된 영역의 realism을 강제합니다.

### 5. Inference Procedures

Inference 동안 과정은 간단한 feed-forward pass입니다:
1.  Input image는 multi-scale features로 encoding됩니다.
2.  Clean face prior는 reconstruction을 생성하는 데 사용되며, 이는 mask network가 imperfections를 찾는 데 도움을 줍니다.
3.  Transformer의 selective self-attention mechanism은 식별된 imperfections의 features를 주변 normal skin에서 집계된 features로 대체합니다.
4.  Decoder는 최종 high-fidelity, retouched image를 합성합니다.

### 6. Potential Applications (Usability)

RetouchFormer는 전문적 및 소비자 수준의 portrait editing에서 활용도가 높습니다:
- **Automated Blemish Removal**: 수동으로 식별하기 어렵고 다양한 acne, pockmarks, spots를 효율적으로 제거합니다.
- **Skin Smoothing**: 기본적인 facial structure를 보존하면서 다크서클을 지우고 skin texture를 매끄럽게 합니다.
- **High-Fidelity Portrait Editing**: "MPRNet: Multi-stage progressive image restoration"에서 논의된 일반적인 image enhancement 방법보다 뛰어난 성능으로 중등도에서 중증의 피부 상태에 대해 realistic details를 합성합니다.
- **Resource-Efficient Retouching**: Semi-supervised 특성과 빠른 convergence("Figure 3" 참조) 덕분에 대규모 paired datasets를 사용할 수 없는 시나리오에서 실용적인 해결책이 됩니다.

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," International Conference on Computer Vision, 2021.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," Advances in Neural Information Processing Systems, 2017.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Neural Information Processing Systems, 2014.
- S. W. Zamir, A. Arora, L. Khan, S. Hayat, M. Khan, F. S. Yang, and L. Shao, "Multi-stage progressive image restoration," IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: transformers for image recognition at scale," International Conference on Learning Representations, 2021.
- D. P. Kingma and J. L. Ba, "Adam: a method for stochastic optimization," International Conference on Learning Representations, 2015.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.

# Experiments
## Datasets
>[!question]
>1. 실험에 사용된 모든 데이터셋에 대하여 각 데이터셋의 종류나 라벨, 데이터 양 등 주요 특징을 설명해 주세요.
>    - 데이터셋들의 분할(훈련, 검증, 테스트 등) 방법이 논문에 명시되어 있다면 설명해주십시오.
>2. 만약 논문에서 직접 데이터를 수집했다면, 그 수집 과정과 데이터셋의 특성(종류, 라벨, 분량 등)을 설명해 주세요.
>3. 실험의 시나리오에서 각 데이터셋이 실험에서 어떤 역할(훈련, 평가, 응용 등)을 맡았는지 구체적으로 알려 주세요.

>[!answer]


### Dataset Description

제공된 논문에서 수행된 실험들은 **RetouchFormer** model을 학습하고 평가하기 위해 몇 가지 주요 datasets를 활용합니다. 이 datasets는 특성, 크기, 그리고 supervision 수준에 따라 다양합니다.

1.  **FFHQR Dataset**  
    face retouching 실험에 사용된 주요 dataset은 "AutoRetouch: automatic professional face retouching"에서 소개된 **FFHQR**입니다. 이 dataset은 70,000쌍의 raw image와 professionally retouched face image로 구성되어 있습니다. 이 dataset의 raw data는 "A style-based generator architecture for generative adversarial networks"에서 처음 제안된 **Flickr-Face-HQ (FFHQ)** dataset에서 가져온 것입니다.
    - **Size and Splits**: "AutoRetouch: automatic professional face retouching"에서 설정된 환경을 따라, dataset은 다음과 같이 나뉩니다:
        - **Training set**: 56,000 image pairs.
        - **Validation set**: 7,000 image pairs.
        - **Test set**: 7,000 image pairs.
    - **Characteristics**: 고품질의 pairwise data를 제공하여, ground truth에 대한 retouching performance 평가를 가능하게 합니다.

2.  **FR-wild Dataset**  
    real-world data에 대한 model의 **generalization capability**를 평가하기 위해, 저자들은 **FR-wild** dataset을 사용합니다.
    - **Size**: 1,000개의 "in-the-wild" face images를 포함하고 있습니다.
    - **Characteristics**: 이 이미지들은 실제 환경에서 마주치는 다양한 유형의 facial blemishes를 특징으로 합니다. 특히, 이 dataset은 retouching ground truth가 없기 때문에 **quantitative benchmarking**보다는 **qualitative evaluation**에 적합합니다.

### Data Collection and Preparation Process

저자들은 raw images를 직접 수집하지 않았으며, 대신 **semi-supervised learning environment**를 조성하기 위해 특정 sampling 및 preparation process를 수행했습니다.

#### Semi-supervised Data Preparation
비싼 비용이 드는 paired retouching data에 대한 의존도를 낮추기 위해, 저자들은 다음과 같이 training data 세트 $\color{orange}X = X_a \cup X_u$ 를 정의합니다:
- $\color{orange}X_a = \{(x_a, y_a)\}$ 는 paired raw-retouching images의 작은 subset을 나타냅니다.
- $\color{orange}X_u = \{x_u, y_u\}$ 는 unpaired images의 더 큰 subset을 나타내며, 여기서 raw image $\color{orange}x_u$ 와 retouching image $\color{orange}y_u$ 는 서로 다른 identity에 해당할 수 있습니다.

paired training data의 비율은 factor $\color{orange}\gamma$ 에 의해 정의됩니다:
$$\color{orange}
\gamma = \frac{\|X_a\|}{\|X\|}
$$
실험에서 저자들은 **FFHQR** dataset에서 paired images를 무작위로 sampling하여 $\color{orange}\gamma$  값이 $\color{orange}\{0.01, 0.1, 0.2, 1\}$  범위에 해당하는 subsets를 생성합니다. 이 과정을 통해 paired data가 극도로 제한적인 상황(예: 전체 데이터의 1%만 있는 경우)에서도 **RetouchFormer**의 안정성과 효과를 입증할 수 있습니다.

### Role of Datasets

각 dataset은 논문의 실험 framework 내에서 별도의 목적을 수행합니다:

- **FFHQR (Training and Quantitative Evaluation)**:  
  이 dataset은 **encoder-transformer-decoder architecture**와 **discriminator**를 학습시키는 데 사용됩니다. ground truth retouched images를 포함하고 있기 때문에, "The unreasonable effectiveness of deep features as a perceptual metric"에서 설명된 Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS)와 같은 metrics를 사용한 **quantitative comparisons**의 기초가 됩니다.

- **FR-wild (Qualitative Evaluation and Generalization)**:  
  이 dataset은 ground truth를 사용할 수 없으므로, 제약이 없는 real-world environments에서 중간에서 심각한 수준의 acne, pockmarks, dark circles를 처리하는 model의 우수성을 시각적으로 보여주는 데 사용됩니다. 이는 **FFHQR**로 학습되었음에도 불구하고, model이 보지 못한 실제 환경의 blemishes에 대해 잘 일반화(generalizes)된다는 것을 증명하는 역할을 합니다.

- **FFHQ (Source Material)**:  
  별도의 실험 세트로 직접 사용되지는 않지만, **FFHQR** 쌍의 "raw" 측면을 구성하는 기초적인 raw facial data를 제공합니다.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and R. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.


## Implementation Details
>[!question]
>1. 실험 설정 및 하이퍼파라미터(예: Learning Rate, Batch Size, Optimizer, Epoch 수 등)와 구현 방법 등 Implementation Detail 전반을 빠짐없이 알려 주세요.
>2. 훈련 시 사용한 GPU의 수와 종류, 훈련 시간 등도 가능한 한 자세히 밝혀 주세요.
>3. 논문에서 재현성(Reproducibility)을 위해 코드나 모델을 공개했다면, 어떤 내용을 안내하고 있는지 알려 주세요.

>[!answer]
 thoughtful
Korean expert in AI, fluent in both Korean and English.
Translate English descriptions of AI research papers into Korean naturally.

        *   No additions or omissions.
        *   Incorporate English terms for domain-specific, academic, technical, and specialized terminology (e.g., "AutoEncoder" instead of "자동부호기").
        *   Do not translate proper nouns (paper titles, people's names).
        *   Do not translate mathematical formulas.
        *   Do not translate the "SOURCE" section.
        *   Formal tone.
        *   Keep quoted sentences, diagrams, and pictures in English.

    *   *Section 1: Implementation Details*
        *   RetouchFormer, image processing, network architecture, optimization process, face retouching.
        *   Image Resolution: $512 \times 512$ pixels.
        *   Feature Dictionary: $\mathcal{P}$, clean face priors, $K = 1024$ elements.
        *   Token Representation: Selective Self-attention, spatial patches, encoder $E$, 512-dimensional vectors, tokens, transformer.
        *   Hyperparameters and Weighting Factors: multi-task loss function, $\lambda = 0.5$, imperfection prediction loss ($L_H$), $\mu = 0.99$, momentum coefficient, Exponential Moving Average (EMA), mask generation network, consistency loss ($L_{con}^{rec}$), $L_1$ loss, VGG-19 based perceptual features, $\zeta = 10$, consistency loss on paired data ($L_{con}^{x2y}$), synthesized retouching result, ground truth.
        *   Optimization Strategy: Adam optimizer, stochastic optimization, Learning Rate ($2e^{-4}$), cosine decay schedule, Training Duration (300,000 iterations), Batch Size (single image).

    *   *Section 2: GPU Information*
        *   GPU Type and Number: Not mentioned.
        *   Training Time: Efficient, 6 times faster, PSNR, Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR ($\gamma=0.1$).

    *   *Section 3: Reproducibility*
        *   Code and Models: No information on open-source code, pre-trained models, or GitHub.

    *   *SOURCES Section* (Keep as is).

    *   *Header:* ### 1. Implementation Details -> ### 1. Implementation Details
    *   *Intro:* **RetouchFormer**의 구현은 고품질의 **face retouching**을 보장하기 위해 **image processing**, **network architecture**, 그리고 **optimization process**에 대한 구체적인 설정을 포함합니다. 5페이지의 "Implementation Details" 섹션에 따르면, 다음과 같은 설정이 사용되었습니다.

    *   *Sub-header:* **Image Pre-processing and Architecture Settings**
    *   *Bullet 1:* **Image Resolution**: 다른 방법들과의 공정한 비교를 위해, 별도의 언급이 없는 한 모든 training 및 testing images는 $512 \times 512$ pixels 해상도로 조정됩니다.
    *   *Bullet 2:* **Feature Dictionary**: **clean face priors**를 캡처하는 데 사용되는 **feature dictionary** $\mathcal{P}$는 $K = 1024$개의 elements로 구성됩니다.
    *   *Bullet 3:* **Token Representation**: 4페이지의 "Selective Self-attention" 섹션에서 설명된 바와 같이, **encoder** $E$의 여러 intermediate layers에서 **spatial patches**가 추출되며, **transformer**를 위한 **tokens** 역할을 하도록 512-dimensional vectors로 변환됩니다.

    *   *Sub-header:* **Hyperparameters and Weighting Factors**
    *   *Intro:* 이 모델은 **multi-task loss function**의 다양한 구성 요소들 사이의 균형을 맞추기 위해 여러 **weighting factors**를 활용합니다. "Implementation Details" 및 "Model Training" 섹션에 명시된 바와 같습니다:
    *   *Bullet 1:* **$\lambda = 0.5$**: Equation (3)에서 사용되는 이 factor는 **imperfection prediction loss** ($L_H$) 내에서 **unpaired training data**의 영향을 제어합니다.
    *   *Bullet 2:* **$\mu = 0.99$**: 이는 "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"에서 참조된 **mask generation network** parameters의 **Exponential Moving Average (EMA)** 업데이트에 사용되는 **momentum coefficient**입니다.
    *   *Bullet 3:* **$\eta = 10$**: 이 **weighting factor**는 **consistency loss** ($L_{con}^{rec}$)를 위한 Equation (9)에 적용되며, $L_1$ loss와 **VGG-19 based perceptual features**를 결합하여 고품질의 **reconstruction**을 보장합니다.
    *   *Bullet 4:* **$\zeta = 10$**: 이 factor는 **paired data**에 대한 **consistency loss** ($L_{con}^{x2y}$)를 위한 Equation (10)에서 사용되며, **synthesized retouching result**와 **ground truth** 사이의 일관성 정도를 측정합니다.

    *   *Sub-header:* **Optimization Strategy**
    *   *Bullet 1:* **Optimizer**: 모델의 parameters는 "Adam: a method for stochastic optimization"에서 상세히 다루는 **deep learning**의 **stochastic optimization**을 위한 대중적인 선택인 **Adam optimizer**를 사용하여 업데이트됩니다.
    *   *Bullet 2:* **Learning Rate**: 초기 **learning rate**는 $2e^{-4}$로 설정됩니다.
    *   *Bullet 3:* **Learning Rate Schedule**: **learning rate**는 **cosine decay schedule**을 사용하여 학습 과정 전반에 걸쳐 조정됩니다.
    *   *Bullet 4:* **Training Duration**: 모델은 총 300,000번의 **iterations** 동안 학습됩니다.
    *   *Bullet 5:* **Batch Size**: 각 **training batch**는 단일 이미지를 포함합니다.

    *   *Section 2: GPU Information*
    *   *Bullet 1:* **GPU Type and Number**: 특정 유형(예: NVIDIA RTX 3090, V100) 및 학습에 사용된 GPU의 수에 관한 정보는 **제공된 파일에 존재하지 않습니다.**
    *   *Bullet 2:* **Training Time**: 정확한 총 학습 시간은 명시적으로 언급되지 않았으나, 5페이지의 "Comparison with State-of-the-arts" 섹션에서 **RetouchFormer**가 매우 효율적임을 언급하고 있습니다. 구체적으로, "Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR ($\gamma=0.1$)"에서 설명된 바와 같이, **base model**의 최고 결과와 일치하는 수준까지 최대 **6 times faster**하게 도달하며 더 높은 **PSNR**로 수렴합니다.

    *   *Section 3: Reproducibility*
    *   *Bullet 1:* **Code and Models**: 제공된 파일에는 **open-source code**, **pre-trained models**의 배포 또는 공용 저장소(GitHub 등)로의 링크에 관한 **정보가 포함되어 있지 않습니다.** "Conclusion" 및 "Acknowledgments" 섹션에서도 재현성을 위한 외부 링크는 언급되지 않았습니다.

    *   *SOURCES Section:* (Keep original).

    *   Check against rules:
        *   Natural Korean? Yes.
        *   English terms for technical words? Yes (e.g., encoder, transformer, tokens, loss function, etc.).
        *   No translation of paper titles/names? Yes.
        *   No translation of formulas? Yes.
        *   No translation of SOURCES? Yes.
        *   Formal tone? Yes.

### 1. Implementation Details

**RetouchFormer**의 구현은 고품질의 **face retouching**을 보장하기 위해 **image processing**, **network architecture**, 그리고 **optimization process**에 대한 구체적인 설정을 포함합니다. 5페이지의 "Implementation Details" 섹션에 따르면, 다음과 같은 설정이 사용되었습니다.

**Image Pre-processing and Architecture Settings**
- **Image Resolution**: 다른 방법들과의 공정한 비교를 위해, 별도의 언급이 없는 한 모든 **training** 및 **testing images**는 $512 \times 512$ pixels 해상도로 조정됩니다.
- **Feature Dictionary**: **clean face priors**를 캡처하는 데 사용되는 **feature dictionary** $\mathcal{P}$는 $K = 1024$개의 **elements**로 구성됩니다.
- **Token Representation**: 4페이지의 "Selective Self-attention" 섹션에서 설명된 바와 같이, **encoder** $E$의 여러 **intermediate layers**에서 **spatial patches**가 추출되며, **transformer**를 위한 **tokens** 역할을 하도록 512-dimensional vectors로 변환됩니다.

**Hyperparameters and Weighting Factors**
이 모델은 **multi-task loss function**의 다양한 구성 요소들 사이의 균형을 맞추기 위해 여러 **weighting factors**를 활용합니다. "Implementation Details" 및 "Model Training" 섹션에 명시된 바와 같습니다:
- **$\lambda = 0.5$**: Equation (3)에서 사용되는 이 **factor**는 **imperfection prediction loss** ($L_H$) 내에서 **unpaired training data**의 영향을 제어합니다.
- **$\mu = 0.99$**: 이는 "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"에서 참조된 **mask generation network parameters**의 **Exponential Moving Average (EMA)** 업데이트에 사용되는 **momentum coefficient**입니다.
- **$\eta = 10$**: 이 **weighting factor**는 **consistency loss** ($L_{con}^{rec}$)를 위한 Equation (9)에 적용되며, $L_1$ loss와 **VGG-19 based perceptual features**를 결합하여 고품질의 **reconstruction**을 보장합니다.
- **$\zeta = 10$**: 이 **factor**는 **paired data**에 대한 **consistency loss** ($L_{con}^{x2y}$)를 위한 Equation (10)에서 사용되며, **synthesized retouching result**와 **ground truth** 사이의 일관성 정도를 측정합니다.

**Optimization Strategy**
- **Optimizer**: 모델의 **parameters**는 "Adam: a method for stochastic optimization"에서 상세히 다루는 **deep learning**의 **stochastic optimization**을 위한 대중적인 선택인 **Adam optimizer**를 사용하여 업데이트됩니다.
- **Learning Rate**: 초기 **learning rate**는 $2e^{-4}$로 설정됩니다.
- **Learning Rate Schedule**: **learning rate**는 **cosine decay schedule**을 사용하여 학습 과정 전반에 걸쳐 조정됩니다.
- **Training Duration**: 모델은 총 300,000번의 **iterations** 동안 학습됩니다.
- **Batch Size**: 각 **training batch**는 단일 이미지를 포함합니다.

### 2. GPU Information

실험에 사용된 하드웨어와 관련하여:
- **GPU Type and Number**: 특정 유형(예: NVIDIA RTX 3090, V100) 및 학습에 사용된 GPU의 수에 관한 정보는 **제공된 파일에 존재하지 않습니다.**
- **Training Time**: 정확한 총 학습 시간은 명시적으로 언급되지 않았으나, 5페이지의 "Comparison with State-of-the-arts" 섹션에서 **RetouchFormer**가 매우 효율적임을 언급하고 있습니다. 구체적으로, "Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR ($\gamma=0.1$)"에서 설명된 바와 같이, **base model**의 최고 결과와 일치하는 수준까지 최대 **6 times faster**하게 도달하며 더 높은 **PSNR**로 수렴합니다.

### 3. Reproducibility

결과 재현을 위한 리소스 가용성과 관련하여:
- **Code and Models**: 제공된 파일에는 **open-source code**, **pre-trained models**의 배포 또는 공용 저장소(GitHub 등)로의 링크에 관한 **정보가 포함되어 있지 않습니다.** "Conclusion" 및 "Acknowledgments" 섹션에서도 재현성을 위한 외부 링크는 언급되지 않았습니다.

**SOURCES**

- D. P. Kingma and J. L. Ba, "Adam: a method for stochastic optimization," in Proc. International Conference on Learning Representation, 2015.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Advances in Neural Information Processing Systems, 2017.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.


## Quantitative Results
>[!question]
>1. 정량적 평가(Quantitative Evaluation)를 위해 어떤 지표(Accuracy, Precision, Recall, F1-score, IoU 등)를 사용했는지 모두 나열하고 해당 지표들의 수식과 해석 방법을 설명해 주세요. (수식이 없다면 관련 참고문헌을 제시하셔도 됩니다.)
>2. 정량 결과를 확인하기 위해 참조해야 할 표(Table)들은 무엇이며, 각각 어떤 비교 결과를 보여줍니까?
>3. 이 표들에 대해 저자들은 어떤 해석을 내렸으며, 수치가 의미하는 바와 이 방법의 장단점은 무엇인지 자세히 기술해 주세요.

>[!answer]


### 1. Evaluation Metrics

**RetouchFormer**의 성능을 평가하기 위해 저자들은 네 가지 주요 **metrics**를 활용합니다. 이 **metrics**는 수학적 및 지각적 관점 모두에서 **synthesized images**의 품질을 평가합니다.

*   **Peak Signal-to-Noise Ratio (PSNR)**: 이는 **image processing**에서 **reconstruction**의 품질을 측정하는 데 사용되는 표준 **metric**입니다. 이는 신호의 최대 가능 전력과 표현의 충실도에 영향을 미치는 **corrupting noise**의 전력 사이의 비율을 계산합니다. 이 논문의 맥락에서, 높은 **PSNR**은 **retouched image**가 수학적으로 **ground truth**에 더 가깝다는 것을 나타냅니다.
*   **Structural Similarity Index Measure (SSIM)**: 이 **metric**은 두 이미지 사이의 유사성을 측정하는 데 사용됩니다. 절대적인 오류에 집중하는 **PSNR**과 달리, **SSIM**은 인간의 시각적 인지와 더 일치하는 **structural information**, **luminance**, 그리고 **contrast**의 변화를 고려합니다. 더 높은 **SSIM** 값(1에 가까울수록)은 이미지 구조의 보존이 더 잘 이루어졌음을 의미합니다.
*   **Learned Perceptual Image Patch Similarity (LPIPS)**: **"The unreasonable effectiveness of deep features as a perceptual metric"**에서 소개된 바와 같이, **LPIPS**는 **pre-trained network**의 **deep features**를 사용하여 두 이미지 사이의 **perceptual distance**를 측정합니다. 이는 전통적인 **metrics**보다 인간의 이미지 유사성 판단을 더 정확하게 모방하도록 설계되었습니다. 낮은 **LPIPS** 점수는 **synthesized image**가 지각적으로 **ground truth**와 더 유사함을 나타냅니다.
*   **Soft-IoU (Intersection over Union)**: 이 **metric**은 특히 **"Analysis of Design Elements"** 섹션(Table 2)에서 **imperfection prediction (mask generation)**의 정확도를 평가하기 위해 사용됩니다. 이는 예측된 **imperfection regions**와 실제 영역 사이의 겹침을 측정합니다. 높은 **Soft-IoU**는 **blemishes**의 더 정밀한 **localization**을 나타냅니다.

**Equations and References**:
**PSNR**, **SSIM**, **LPIPS**, 그리고 **Soft-IoU**에 대한 수학적 방정식은 **제공된 파일에 존재하지 않습니다.** 논문은 **LPIPS metric**을 위해 **"The unreasonable effectiveness of deep features as a perceptual metric"**을 참조하고 있지만, 이러한 평가 **metrics** 중 어느 것에 대해서도 구체적인 공식은 제공하지 않습니다.

### 2. Reference Tables

정량적 결과와 비교는 세 개의 주요 표에 제시되어 있습니다:

*   **Table 1: Quantitative comparison with competing methods on FFHQR**: 이 표는 **RetouchFormer**와 **Pix2PixHD** (**"Image-to-image translation with conditional adversarial networks"**로 참조됨)와 같은 일반적인 **image translation models**, **GPEN** (**"GAN prior embedded network for blind face restoration in the wild"**)과 같은 **restoration models**, 그리고 **BPFRe** (**"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"**)와 같은 전문적인 **retouching models**를 포함한 여러 **state-of-the-art** 방법들 간의 포괄적인 비교를 제공합니다. 비교는 네 가지 다른 수준의 **supervision** ($\gamma = 0.01, 0.1, 0.2, 1$)에 대해 수행되어, **paired training data**의 양이 증가함에 따라 각 모델이 어떻게 수행되는지 보여줍니다.
*   **Table 2: Quantitative results of RetouchFormer with and without the priors $\mathcal{P}$ in soft mask generation**: 이 표는 **"Clean Face Prior"**에 초점을 맞춘 **ablation study**입니다. 이는 전체 **RetouchFormer**와 **mask generation**을 위해 **feature dictionary** $\mathcal{P}$를 사용하지 않는 변형 모델의 성능을 비교합니다. **PSNR**, **SSIM**, 그리고 **Soft-IoU**를 사용하여 **prior**가 이미지 품질과 **blemish localization** 모두에 미치는 영향을 보여줍니다.
*   **Table 3: Results of RetouchFormer and ablative models on FFHQR ($\gamma=0.1$)**: 이 표는 특정 **design elements**의 중요성을 평가합니다. 전체 모델을 다음 세 가지 변형 모델과 비교합니다:
    1.  **w/ SPS**: **multi-scale tokens** 대신 단일 **patch size**를 사용합니다.
    2.  **w/ GSA**: 제안된 **Selective Self-Attention** 대신 **Global Self-Attention**을 사용합니다.
    3.  **w/o $X_u$**: **unpaired training data**를 사용하지 않고 학습되었습니다.

### 3. Interpretation of Results

저자들은 **RetouchFormer**의 강점과 핵심 구성 요소의 필요성을 강조하기 위해 정량적 결과를 해석합니다.

**Significance of the Numbers**:
*   **Superiority over SOTA**: Table 1에서 **RetouchFormer**는 모든 **supervision levels**에서 일관되게 가장 높은 **PSNR** 및 **SSIM**과 가장 낮은 **LPIPS**를 달성합니다. 예를 들어, $\gamma=0.1$에서 두 번째로 우수한 방법인 **BPFRe**보다 **PSNR**에서 0.56 dB 앞섭니다. **LPIPS**의 개선은 특히 주목할 만하며, **BPFRe**보다 12 **percentage point** 개선되어 훨씬 높은 **perceptual realism**을 나타냅니다.
*   **Efficiency**: **"Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR"**에서 보여지듯이, 이 모델은 **base model**의 최고 결과와 최대 6배 더 빠르게 일치하며, 이는 높은 **computational efficiency**를 의미합니다.
*   **Impact of Design Elements**: Table 3은 **multi-scale tokens (SPS)**를 제거하면 성능이 크게 저하됨(**PSNR**에서 4.93 dB 하락)을 보여주며, 이는 다양한 규모의 **blemishes**를 처리하는 데 있어 그들의 역할을 입증합니다. **Selective Self-Attention**을 **Global Self-Attention (GSA)**으로 대체하면 **PSNR**이 1.06 dB 하락하며, 이는 **imperfections**를 정상적인 피부 특징으로 효과적으로 대체하는 데 **selective mechanism**이 중요하다는 것을 확인시켜 줍니다.

**Strengths and Weaknesses**:
*   **Strengths**:
    *   **High Fidelity and Realism**: 이 모델은 수학적으로 정확하면서도 지각적으로 즐거운 깨끗한 얼굴 이미지를 합성하는 데 탁월합니다.
    *   **Robustness to Limited Data**: **semi-supervised framework**와 **unpaired data ($X_u$)**의 사용 덕분에, 모델은 매우 적은 **paired data ($\gamma=0.01$)**로도 높은 성능을 유지합니다.
    *   **Precise Localization**: **clean face priors**의 사용은 Table 2에 나타난 **Soft-IoU**의 17.9% 향상에서 입증되듯이 **blemish detection**을 크게 개선합니다.
*   **Weaknesses**:
    제공된 파일에는 제안된 **RetouchFormer** 방법의 구체적인 **weaknesses**가 명시적으로 나열되어 있지 않습니다. 그러나 저자들은 **selective self-attention mechanism**이 없으면 모델이 "**pockmarks**를 중화하고 **dark circles**를 지우는 데 실패한다"고 언급하며, 이는 작업의 복잡성과 성공을 위한 해당 특정 구성 요소에 대한 모델의 의존성을 강조합니다.

**SOURCES**

- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and R. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," in Proc. International Conference on Computer Vision, 2021.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Multi-stage progressive image restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.


## Qualitative Results
>[!question]
>1. 정성적 평가(Qualitative Result)에 대한 내용을 보려면 어떤 그림(이미지, 그래프, 예시 출력 등)들을 확인해야 하며, 각각 무엇을 나타내나요? 빠짐없이 공유해 주세요.
>2. 이 그림들에 대해 저자들은 어떻게 해석하고 있나요? 어떤 결론과 의미를 이끌어내며, 방법의 장단점은 무엇인가요? 자세히 알려 주세요.
>3. 논문에서 실패 사례(Failure Case)나 극단적 예시(Edge Case)를 보여주고 있다면, 어떤 예시이며 그 원인은 무엇인가요?

>[!answer]


### 1. Qualitative Results

**RetouchFormer**의 **qualitative performance**는 이미지의 높은 **fidelity**를 유지하면서 얼굴의 결점을 식별하고 제거하는 모델의 능력을 보여주는 여러 그림을 통해 설명됩니다. 다음 그림들은 모델의 효과에 대한 시각적 증거를 제공합니다:

*   **Figure 1: The main idea behind RetouchFormer** (1페이지)  
    이 그림은 제안된 접근 방식에 대한 입문용 시각적 요약 역할을 합니다. 여드름이 눈에 띄는 얼굴의 **Source image**와 결점의 위치를 나타내는 **Imperfection prediction** map(결점의 위치를 나타내는 **heatmap**)을 보여줍니다. 그런 다음 **imperfection regions**(**queries**)의 **features**가 **normal skin regions**(**keys** 및 **values**)의 **features**로 대체되는 **Selective self-attention** mechanism을 설명합니다. 마지막으로 여드름이 제거되고 사실적인 **skin texture**로 대체된 **Retouching result**를 보여줍니다.

*   **Figure 4: Visual comparison with competing methods on FR-wild images** (6페이지)  
    이 그림은 **RetouchFormer**와 GPEN (**"GAN prior embedded network for blind face restoration in the wild"**), AutoRetouch (**"AutoRetouch: automatic professional face retouching"**), MPRNet (**"Multi-stage progressive image restoration"**), RestoreFormer (**"RestoreFormer: High-quality blind face restoration from undegraded key-value pairs"**), ABPN (**"ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"**), 그리고 BPFRe (**"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"**)를 포함한 여러 **state-of-the-art methods** 간의 **side-by-side qualitative comparison**을 제공합니다. 이미지들은 중간에서 심각한 수준의 여드름, **pockmarks**, 다크서클이 있는 실제 얼굴을 포함하는 **FR-wild** dataset에서 가져온 것입니다. **Red boxes**는 다른 방법들이 결점을 완전히 제거하는 데 어려움을 겪는 특정 영역을 강조하는 데 사용됩니다.

*   **Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild** (6페이지)  
    이 그림은 세 가지 세트의 **high-resolution before-and-after examples**를 제시합니다. 특히 다양한 유형의 얼굴 고민을 처리하는 모델의 다재다능함을 입증합니다. 예시들은 기저의 **facial structure**와 필수적인 **features**를 보존하면서 **removing acne**, 눈 밑의 **erasing dark circles**, 그리고 **smoothing skin**에서의 모델의 성공을 강조합니다.

### 2. Interpretation of Qualitative Results

저자들은 이러한 시각적 결과를 해석하여 기존 기술에 대한 **RetouchFormer**의 우수성을 강조합니다.

*   **Handling Large-Scale Blemishes**: **Figure 4**에 대한 논의에서 저자들은 BPFRe 및 RestoreFormer와 같은 고급 방법들이 **generic models**보다 더 나은 결과를 생성하지만, 여전히 "fail to remove the blemishes at large scales"라는 점을 관찰했습니다. 그러나 **RetouchFormer**는 "significantly pleasant" 결과를 생성하며, 다른 모델들이 남겨두는 심각한 결점까지 성공적으로 제거하는 것으로 설명됩니다.
*   **Realism and Fidelity**: 저자들은 **RetouchFormer**에 의해 합성된 리터칭 이미지가 "high realism and fidelity"를 달성한다고 결론짓습니다. 이는 **clean face priors**를 활용하여 "approximately predict imperfections"하고 "simultaneously synthesize realistic details"하는 모델의 능력 덕분입니다.
*   **Generalization**: 이러한 그림들에서 도출된 핵심 통찰은 모델의 강력한 **generalization** 능력입니다. 모델은 **FFHQR** dataset으로 학습되었지만, **FR-wild** dataset(학습 중에 모델이 보지 못한 데이터)에 대한 **qualitative results**는 모델이 다양한 실제 얼굴 결점을 효과적으로 처리할 수 있음을 보여줍니다.
*   **Strengths Highlighted**: 강조된 주요 강점은 결점의 정밀한 **localization**과 해당 영역을 채우기 위해 정상 피부의 **contextual information**을 효과적으로 사용하는 것입니다. 저자들은 모델이 "utilize the spatial information of imperfections and synthesize the content from the contextual features of normal skin"할 수 있다고 언급합니다.

### 3. Failure Case

제안된 **RetouchFormer** 모델에 대한 특정 **failure cases** 또는 **edge cases**에 관한 정보는 **제공된 파일에 존재하지 않습니다.** 저자들은 모델의 성공과 경쟁 방법 대비 개선 사항에 집중합니다. 특정 **ablated** 버전의 모델(예: Global Self-Attention을 사용하는 버전)이 **pockmarks**를 중화하는 데 실패한다고 언급하지만, 전체 **RetouchFormer** 모델이 특정 입력에서 실패하는 예시는 제공하지 않습니다.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Multi-stage progressive image restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.


## Ablation Study
>[!question]
>1. 저자들이 수행한 Ablation Study 목록을 모두 말해 주세요. 모듈, 손실 항, 알고리즘 등 어떤 요소를 제거 혹은 변경해 실험했는지, 그 목적은 무엇인지 설명해 주세요.
>2. 각 Ablation Study 결과는 어느 표나 그래프, 그림을 보면 되나요? 그 위치(번호)를 알려 주십시오.
>3. 각 Ablation Study에 대하여, 저자들의 해석에 따르면 어떤 구성 요소가 성능 개선에 중요한 역할을 하나요? 제거 혹은 추가 시 어떤 변화가 있는지, 그 구체적 해석을 알려 주세요.

>[!answer]


이 논문의 저자들은 **RetouchFormer**의 개별 **design elements**의 효과를 검증하기 위해 일련의 **ablation studies**를 수행했습니다. 이 연구들은 주로 7페이지의 **"Analysis of Design Elements"** 섹션에 자세히 설명되어 있습니다.

### 1. Ablation Study List

저자들은 **clean face prior**, **multi-scale tokens**, **selective self-attention**, 그리고 **unpaired training data**의 기여도를 평가하기 위해 네 가지 별개의 **ablation studies**를 수행했습니다.

*   **Clean Face Prior ($\mathcal{P}$)**:
    *   **Purpose**: **retouching images** $\{y_a, y_u\}$에서 학습된 **feature dictionary** $\mathcal{P}$가 실제로 **face imperfection prediction**에 도움이 되는지 확인하기 위함입니다.
    *   **Method**: 저자들은 '**w/o $\mathcal{P}$**'라는 **variant**를 만들었으며, 여기서 **mask generation**을 담당하는 **network** $H$는 **dictionary**로부터 재구성된 **features**의 **guidance** 없이 **raw image**에서 직접 **imperfection mask**를 생성합니다.

*   **Multi-scale Tokens**:
    *   **Purpose**: 다양한 크기의 **imperfections**를 처리하기 위해 **encoder**의 여러 **intermediate layers**에서 **features**를 추출하는 것(**multi-scale**)이 필요한지 결정하기 위함입니다.
    *   **Method**: **multi-scale patch-based representation** 대신 **encoder**의 최종 **output**만을 사용하여 **tokens**를 형성하는 '**w/ SPS**' (**Single Patch Size**) **variant**를 개발했습니다.

*   **Selective Self-Attention**:
    *   **Purpose**: **imperfection regions**와 **normal skin** 사이의 **interactions**를 억제하는 제안된 **selective self-attention mechanism**이 **standard attention mechanisms**보다 우수한지 평가하기 위함입니다.
    *   **Method**: **selective mechanism**을 **standard global self-attention**으로 대체하여 '**w/ GSA**' (**Global Self-Attention**) **variant**를 구축했습니다.

*   **Unpaired Training Data ($X_u$)**:
    *   **Purpose**: **semi-supervised setting**에서 대량의 **unpaired data**를 사용하는 것의 영향을 분석하기 위함입니다.
    *   **Method**: 제한된 **paired data** $X_a$만을 사용하여 **model**을 학습시키는 '**w/o $X_u$**' **variant**를 테스트했습니다.

### 2. Reference Tables and Figures

이러한 **ablation studies**의 결과는 다음의 **tables**와 **figures**에 제시되어 있습니다.

*   **Clean Face Prior**: 결과는 **"Table 2: Quantitative results of RetouchFormer with and without the priors $\mathcal{P}$ in soft mask generation"**에 제시되어 있으며, **"Figure 6: The soft masks generated by RetouchFormer with and without the prior $\mathcal{P}$."**에서 시각화되었습니다.
*   **Multi-scale Tokens, Selective Self-Attention, and Unpaired Data**: 이 세 가지 연구에 대한 **quantitative results**는 **"Table 3: Results of RetouchFormer and ablative models on FFHQR ($\gamma=0.1$)."**에 정리되어 있습니다.
*   **Visual Comparisons**: **SPS**, **GSA**, 그리고 $X_u$ **variants**에 대한 **qualitative differences**는 **"Figure 7: Representative retouching images synthesized by RetouchFormer and ablative models on FR-wild images."**에 설명되어 있습니다.

### 3. Interpretation of Components

저자들의 분석에 따르면, 각 **component**는 중요한 역할을 하지만 일부는 특정 **performance metrics**에 더 급격한 영향을 미칩니다.

#### Most Crucial Components
*   **Multi-scale Tokens**: 이는 높은 **PSNR**을 유지하는 데 가장 중요한 **component**로 보입니다. 이를 제거했을 때('**w/ SPS**') **PSNR**에서 **4.93 dB**라는 가장 큰 **performance drop**이 발생했습니다. 저자들은 **"Analysis of Design Elements"**에서 **multi-scale tokens**가 "**imperfections with different scales**"를 처리하는 데 필수적이라고 명시합니다.
*   **Unpaired Training Data ($X_u$)**: 이는 **perceptual quality**와 **model stability**에 결정적입니다. '**w/o $X_u$**' **variant**는 **LPIPS metric** 측면에서 **full model**보다 약 **74%** 낮은 성능을 보였습니다. 저자들은 $X_u$가 "**stabilizing the performance of RetouchFormer**"에 중요하다고 결론지었습니다.
*   **Selective Self-Attention**: 이는 특정 **blemishes**를 실제로 제거하는 데 중요합니다. '**w/ GSA**' **variant**는 **PSNR**에서 **1.06 dB** 하락을 보였으며, **"Figure 7: Representative retouching images synthesized by RetouchFormer and ablative models on FR-wild images"**에서 볼 수 있듯이 "**fails to neutralize pockmarks and erase dark circles under the eyes**"와 같은 결과를 보입니다.

#### Effects of Adding/Removing Components
*   **Adding the Clean Face Prior ($\mathcal{P}$)**: **"Table 2"**에서 보여주듯이, **prior**를 포함하면 **imperfection prediction**을 위한 **Soft-IoU**에서 **17.9% gain**을 얻습니다. 시각적으로 **"Figure 6"**은 **prior**가 **model**이 '**w/o $\mathcal{P}$**' 버전에 비해 "**more imperfections**"를 발견하도록 돕는다는 것을 입증합니다.
*   **Removing Selective Self-Attention**: 명시적인 **imperfection prediction** 및 **suppression** 없이는 **model**이 **blemish features**를 **normal skin features**로 효과적으로 대체할 수 없습니다. 이는 최종 **retouched image**의 **fidelity**를 크게 떨어뜨립니다.
*   **Removing Multi-scale Tokens**: **model**은 유연성을 잃게 됩니다. 작은 점과 같은 **fine-grained details**와 심한 여드름과 같은 **larger-scale blemishes**를 동시에 처리하는 능력이 저하됩니다.
*   **Removing Unpaired Data**: **model**이 더 큰 **unpaired dataset**에 의해 제공되는 **regularization**이 부족해지기 때문에 **face retouching** 작업이 "**highly challenging**"해지며, **clean faces**에 대한 **perceptual similarity**가 크게 손실됩니다.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and R. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Advances in Neural Information Processing Systems, 2017.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in Proc. Neural Information Processing Systems, 2017.
- T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019.


## Results Summary
>[!question]
>지금까지 논의된 정량/정성적 결과와 Ablation Study를 종합하여 풍부하고 체계적인 요약을 부탁드립니다. 또한, 이 방법론의 강점과 약점도 동시에 짚어 주세요.

>[!answer]


**RetouchFormer**는 고품질 face retouching을 위해 설계된 transformer-based framework로, 이 task를 face image inpainting의 일반화된 "soft" 버전으로 간주합니다. 이 모델은 imperfections를 식별하는 동시에 사실적인 skin content를 합성합니다. 아래는 제공된 연구를 바탕으로 한 정량적 및 정성적 결과, ablation studies, 그리고 방법론의 강점과 약점에 대한 종합적인 요약입니다.

### 1. Summary of Quantitative and Qualitative Results

**정량적 성능 (Quantitative Performance)**
**RetouchFormer**의 정량적 평가는 주로 "AutoRetouch: automatic professional face retouching"에서 설명된 **FFHQR** dataset에서 수행되었습니다. 모델은 paired training data의 비율을 나타내는 다양한 supervision levels ($\gamma$)에 대해 여러 state-of-the-art (SOTA) 방법들과 비교되었습니다.

*   **지표상의 우위 (Superiority in Metrics)**: **"Table 1: Quantitative comparison with competing methods on FFHQR"**에 나타난 바와 같이, **RetouchFormer**는 BPFRe ("Blemish-Aware and Progressive Face Retouching With Limited Paired Data"), ABPN ("ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"), 그리고 RestoreFormer ("RestoreFormer: High-quality blind face restoration from undegraded key-value pairs")를 포함한 모든 경쟁 방법들을 일관되게 능가합니다.
*   **주요 성과 (Key Gains)**: supervision level이 $\gamma=0.1$일 때, **RetouchFormer**는 45.13 dB의 PSNR을 달성하였으며, 이는 두 번째로 우수한 방법인 BPFRe보다 0.56 dB 높은 수치입니다. 더 중요한 것은, LPIPS 점수에서 0.0093을 기록하여 BPFRe 대비 12% 포인트의 개선을 보였으며, 이는 훨씬 더 높은 perceptual fidelity를 의미합니다.
*   **효율성 (Efficiency)**: 이 모델은 매우 효율적입니다. **"Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR"**에 따르면, **RetouchFormer**는 더 높은 PSNR로 수렴하며, base model의 최상위 결과에 도달하는 속도가 최대 6배 더 빠릅니다.

**정성적 성능 (Qualitative Performance)**
정성적 결과는 다양하고 심각한 real-world blemishes를 처리하는 모델의 능력을 입증합니다.
*   **실제 환경에서의 일반화 (Real-World Generalization)**: **FR-wild** dataset을 사용한 **"Figure 4: Visual comparison with competing methods on FR-wild images"**는 다른 방법들이 종종 대규모 acne나 pockmarks의 흔적을 남기는 반면, **RetouchFormer**는 이를 성공적으로 제거하고 해당 영역을 사실적인 skin texture로 대체함을 보여줍니다.
*   **다재다능함 (Versatility)**: **"Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild"**는 세 가지 뚜렷한 영역에서의 성공을 보여줍니다: inflammatory acne 제거, 눈 밑의 dark circles 지우기, 그리고 주요 facial features를 보존하면서 전반적인 skin texture를 매끄럽게 다듬기입니다.

### 2. Ablation Study Overview

저자들은 **RetouchFormer**의 세 가지 핵심 설계 요소를 검증하기 위해 체계적인 ablation studies를 수행했습니다.

**A. Clean Face Prior ($\mathcal{P}$)**
모델은 "Neural discrete representation learning"에서 제안된 Vector Quantization (VQ)을 통해 clean face priors를 캡처하기 위한 feature dictionary $\mathcal{P}$를 학습합니다. dictionary에서 가장 가까운 element $\rho_*$는 다음과 같이 찾습니다:
$$\color{orange}
\rho_*|(i,j) = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|(i,j) - \rho_k\|_2^2
$$
**"Table 2"**에 나타난 것처럼, 이 prior를 사용하면 imperfection prediction의 Soft-IoU에서 **17.9%의 이득**을 얻을 수 있습니다. **"Figure 6"**은 이 prior가 모델이 prior가 없는 모델보다 훨씬 더 많은 blemish regions를 식별하도록 돕는다는 것을 확인시켜 줍니다.

**B. Multi-scale Tokens**
고정된 scale 대신, 모델은 encoder $E$의 여러 중간 layer에서 patches를 추출합니다. **"Table 3"**은 Single Patch Size (SPS)를 사용할 경우 PSNR이 **4.93 dB**이나 대폭 하락함을 보여주며, 이는 다양한 크기의 imperfections를 해결하는 데 multi-scale features가 필수적임을 증명합니다.

**C. Selective Self-Attention**
이것은 가장 혁신적인 구성 요소로, 모델이 imperfections와 normal skin 사이의 상호작용을 억제합니다. queries ($Q$), keys ($K$), values ($V$)는 예측된 imperfection mask $f_H(x_-)$를 사용하여 공식화됩니다:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
최종 feature update는 다음과 같습니다:
$$\color{orange}
f_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
$$
**"Table 3"**에 따르면, 이를 Global Self-Attention (GSA)으로 교체하면 **1.06 dB**의 PSNR 손실이 발생하고 pockmarks와 dark circles를 지우는 데 실패하게 됩니다 (**"Figure 7"** 참조).

### 3. Strengths and Weaknesses of the Methodology

**강점 (Strengths)**
*   **Semi-supervised Stability**: 이 방법론은 unpaired data ($X_u$)를 효과적으로 활용하여 학습을 안정화합니다. $X_u$를 제거하면 LPIPS 성능이 74% 저하됩니다 (**"Table 3"**).
*   **High Fidelity**: retouching을 "soft inpainting"으로 공식화함으로써, 모델은 blemish regions만 수정되도록 보장하고 얼굴의 나머지 부분은 그대로 유지하여 identity와 필수적인 세부 사항을 보존합니다.
*   **Computational Efficiency**: selective attention을 적용한 transformer-based design은 global convolution-based methods보다 빠르고 효과적입니다.

**약점 (Weaknesses)**
제공된 파일에는 **RetouchFormer**의 **약점이 명시적으로 나열되어 있지 않습니다.** 그러나 방법론에 기반하여 다음과 같이 추론할 수 있습니다:
*   **Mask Accuracy에 대한 의존성**: selective self-attention의 성공은 imperfection prediction mask $f_H$의 정확도에 크게 의존합니다. 만약 mask가 blemish를 식별하지 못하면, 해당 영역에 대해 selective attention이 트리거되지 않습니다.
*   **Two-Stream Mask Generation의 복잡성**: two-stream structure와 Exponential Moving Average (EMA) 업데이트를 갖춘 U-Net의 사용은 단순한 end-to-end 모델에 비해 architectural complexity를 가중시킵니다.

### 4. Additional Experimental Details

**Loss Functions**
모델은 네 가지 loss terms의 조합을 사용하여 최적화됩니다:
1.  **Prior Loss ($L_{prior}$)**: clean face features를 재구성하도록 encoder와 dictionary를 최적화합니다.
2.  **Imperfection Prediction Loss ($L_H$ 및 $L_H$)**: paired 및 unpaired data를 모두 사용하여 정확한 mask generation을 강제합니다.
3.  **Consistency Loss ($L_{con}^{rec}$ 및 $L_{con}^{x2y}$)**: 합성된 이미지가 ground truth와 일치하고 VGG-19 features를 사용하여 고품질 세부 사항을 유지하도록 보장합니다.
4.  **Adversarial Loss ($L_{adv}^{syn}$ 및 $L_{adv}^{disc}$)**: 합성된 이미지가 실제 clean faces의 통계와 일치하도록 discriminator $D$를 사용합니다.

**Training Setup**
모델은 **Adam optimizer**를 사용하여 **300,000 iterations** 동안 학습되었으며, 초기 learning rate는 **$2e^{-4}$**이고 **cosine decay schedule**을 적용했습니다. 각 batch는 $512 \times 512$ resolution의 단일 이미지를 포함합니다.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and R. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in Proc. Neural Information Processing Systems, 2017.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Advances in Neural Information Processing Systems, 2017.
- D. P. Kingma and J. L. Ba, "Adam: a method for stochastic optimization," in Proc. International Conference on Learning Representation, 2015.
- T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- S. W. Zamir, 핵심 A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Multi-stage progressive image restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.

# Conclusion
## Limitations ans Future works
>[!question]
>1. 이 논문에서 제안한 방법의 단점이나 아직 해결되지 않은 문제, 일반화에 대한 이슈 등 한계점을 상세히 설명해 주세요.
>2. 논문에서 제시된 후속 연구 아이디어나 확장 가능 방향(더 큰 규모의 데이터셋 적용, 모델 구조 확대 등)에 대해 구체적으로 풀어 써 주세요.

>[!answer]


### RetouchFormer 개요

RetouchFormer는 고품질 face retouching을 위해 설계된 transformer-based framework입니다. 이는 imperfection localization의 부재나 대규모 pairwise training data에 대한 과도한 의존성과 같은 이전 방법론들의 한계를 해결합니다. 이 모델은 face retouching을 "soft inpainting" task로 취급하며, 결함을 식별하는 동시에 해당 영역에 realistic skin textures를 합성합니다. 이는 clean face priors, reconstruction-oriented localization module, 그리고 selective self-attention mechanism의 결합을 통해 달성됩니다.

"RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention"에서 설명된 바와 같이, architecture는 encoder-transformer-decoder 설계를 따릅니다. encoder는 multi-scale features를 추출하고, transformer는 imperfection features를 normal skin features로 대체하기 위해 selective self-attention을 수행하며, decoder는 최종 retouched image를 합성합니다.

---

### 1. Limitations

제공된 파일에 따르면, 저자들은 별도의 "Limitations" 섹션을 포함하지 않았으며, 주로 기존 state-of-the-art methods 대비 모델의 우수한 성능에 초점을 맞춥니다. 그러나 텍스트로부터 제약 사항 및 과제에 관한 다음과 같은 점들을 유추할 수 있습니다.

*   **Unresolved Problems and Generalization Issues**: RetouchFormer 모델 자체의 구체적인 미해결 문제나 generalization issues에 관한 정보는 **제공된 파일에 존재하지 않습니다**. 반대로, 저자들은 "Figure 4: Visual comparison with competing methods on FR-wild images" 및 "Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild"에서 보여주듯, FR-wild dataset과 같은 real-world data에서의 강력한 generalization capabilities를 강조합니다.
*   **Explicit Constraints**:
    *   **Input Resolution**: 5페이지의 "Implementation Details" 섹션에서 저자들은 공정한 비교를 위해 별도의 언급이 없는 한 training 및 testing images 모두 $\color{orange}512 \times 512$  pixels로 resize되었다고 명시합니다. 이는 모델의 표준 동작을 위한 고정된 spatial constraint를 나타냅니다.
    *   **Data Dependency**: 모델이 semi-supervised 방식이며 paired data의 필요성을 줄여주지만, 저자들은 "Analysis of Design Elements" (7페이지)에서 "paired data의 양이 감소함에 따라 face retouching task는 매우 도전적이 된다"고 언급합니다. 그들은 unpaired data ($\color{orange}X_u$ )가 성능 안정화에 필수적임을 강조하며, 이는 모델이 high fidelity를 유지하기 위해 여전히 균형 잡힌 dataset을 필요로 함을 시사합니다.
*   **Discriminator Sensitivity**: 저자들은 "Model Training" 섹션(5페이지)에서 "imperfection regions가 매우 작을 수 있으며, 일반적인 image-level discriminators는 이 경우 속기 쉽다"고 언급합니다. 이를 완화하기 위해, 그들은 여러 layers에 mask features를 주입하여 pixel-level real-fake identification을 수행하는 특정 discriminator를 설계해야 했습니다.

---

### 2. Future Works

잠재적인 확장 또는 향후 연구 방향에 관하여:

*   **Proposed Research Directions**: 구체적인 향후 연구 방향이나 잠재적 확장(예: 모델을 더 큰 datasets에 적용하거나 architecture를 확장하는 것 등)에 관한 정보는 **제공된 파일에 존재하지 않습니다**.
*   **Conclusion Summary**: 7페이지의 "Conclusion" 섹션은 retouching을 soft inpainting으로 정식화하고, clean face priors를 학습하며, selective self-attention을 활용하는 등 본 연구의 현재 성과를 요약하고 있지만, 저자들이 다음에 무엇을 조사할 계획인지는 명시적으로 밝히지 않았습니다.

---

### RetouchFormer Mechanism의 기술적 세부 사항

저자들이 왜 한계점에 집중하지 않았는지 이해하기 위해, 그들이 이전 문제들을 해결했다고 주장하는 핵심 혁신 기술들을 검토하는 것이 도움이 됩니다:

#### Clean Face Priors
모델은 Vector Quantization (VQ)을 사용하여 clean face priors를 포착하기 위한 feature dictionary $\color{orange}\mathcal{P}$ 를 학습합니다. 위치 $\color{orange}(i, j)$ 에서의 feature vector는 dictionary에서 가장 가까운 요소로 근사화됩니다:
$$\color{orange}
\rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2
$$
여기서 $\color{orange}E(y_-)$ 는 retouching image의 encoder feature입니다. 이 prior는 네트워크가 "깨끗한" 피부가 어떻게 보여야 하는지 식별하는 데 도움을 주어, 결함의 localization을 보조합니다 ("RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention"에서 논의됨).

#### Selective Self-Attention
selective self-attention mechanism은 "soft inpainting" 접근 방식의 핵심입니다. 이는 imperfection regions 간의 상호작용을 억제하고, 대신 모델이 normal skin의 features를 사용하여 해당 영역을 채우도록 유도합니다. queries ($\color{orange}Q$ ), keys ($\color{orange}K$ ), 그리고 values ($\color{orange}V$ )는 다음과 같이 공식화됩니다:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q,
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k,
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v,
$$
여기서 $\color{orange}f_H(x_-)$ 는 예측된 imperfection mask를 나타냅니다. 이는 결함과 관련된 tokens가 queries 역할을 하도록 보장하는 반면, keys와 values는 normal skin regions에서 추출되도록 합니다 (4페이지의 "Selective Self-attention" 섹션에 상세히 설명됨).

**SOURCES**

- L. Xie, W. Xue, W. Xu, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Proceedings of Neural Information Processing Systems, 2017.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.


## Conclusion
>[!question]
>이 논문의 결론을 정리해 주세요. 연구진이 주장하는 바와 이를 뒷받침하는 핵심 근거(Method 강점 및 실험 결과)는 무엇인가요?

>[!answer]


"RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention" 논문의 "Conclusion" 섹션은 face retouching의 새로운 정식화(formulation)와 모델이 기존 방법들을 능가할 수 있게 하는 architectural innovations를 강조하며 연구를 요약합니다.

### 1. Main Claims

저자들은 image processing 및 computer vision 분야에서 RetouchFormer의 기여를 정의하는 몇 가지 핵심 주장과 발견을 강조합니다:

*   **'Soft Inpainting'으로서의 정식화**: 주요 주장은 face retouching을 face image inpainting의 일반화된 "soft" 버전으로 취급하는 것입니다. "Proposed Approach" 섹션에서 언급되고 "Conclusion"에서 재차 강조되었듯이, global transformations를 적용하는 기존 방식과 달리 RetouchFormer는 "결함 영역을 공동으로 탐지하고 완성(jointly detect and complete imperfection regions)"하도록 설계되었습니다.
*   **Localization을 위한 Clean Face Priors**: 저자들은 결함을 정확히 예측하기 위해 feature dictionary 형태의 "clean face priors"를 학습하는 것이 필수적이라고 주장합니다. original features와 dictionary로부터 재구성된 features 사이의 차이를 측정함으로써, 모델은 결함이 위치한 곳을 대략적으로 예측할 수 있습니다. 이는 dictionary $\color{orange}\mathcal{P}$ 에서의 nearest-neighbor search로 수학적으로 표현됩니다:
    $$\color{orange}
    \rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2,
    $$
    여기서 $\color{orange}E(y_-)$ 는 입력 이미지의 encoder features를 나타냅니다 ("RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention" 참조).
*   **Selective Self-Attention Mechanism**: 주요 주장 중 하나는 "prior-based selective self-attention" mechanism의 효과성입니다. 이 mechanism은 spatial interactions를 제한함으로써 "얼굴 결함을 억제(suppress face imperfections)"하도록 설계되었습니다. global attention 대신, 결함 영역의 features가 "normal skin" 영역의 정보를 사용하여 합성되도록 보장합니다.
*   **Semi-supervised Stability**: 저자들은 자신들의 접근 방식이 "unpaired training data"를 효과적으로 활용하여 성능을 안정화한다고 주장합니다. 이는 실제 환경에서 수집하기 어렵고 비용이 많이 드는 완벽하게 쌍을 이룬(perfectly paired) raw-and-retouched datasets에 대한 과도한 의존도를 줄여줍니다.

### 2. Supporting Evidence

이러한 주장들을 정당화하기 위해 저자들은 방법론적 강점과 실험 결과의 조합을 제공합니다:

*   **FFHQR에서의 정량적 우위**: 저자들은 "Table 1: Quantitative comparison with competing methods on FFHQR"의 결과를 인용하여 RetouchFormer가 BPFRe, ABPN, RestoreFormer와 같은 선진 방법들을 능가함을 증명합니다. 예를 들어, $\color{orange}\gamma=0.1$  (10% paired data)에서 RetouchFormer는 45.13 dB의 PSNR과 0.0093이라는 가장 낮은 LPIPS score를 기록하며, 더 높은 fidelity와 perceptual quality를 나타냈습니다.
*   **실제 데이터에서의 정성적 리얼리즘**: "Conclusion"은 우수한 능력을 입증하는 "extensive comparisons"를 언급합니다. 이는 "Figure 4: Visual comparison with competing methods on FR-wild images"와 "Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild"에 의해 뒷받침되며, 모델이 얼굴 구조를 보존하면서도 중간에서 심각한 수준의 acne, pockmarks, dark circles를 성공적으로 제거함을 보여줍니다.
*   **Ablation Studies (설계 검증)**:
    *   **Priors의 효과**: "Table 2: Quantitative results of RetouchFormer with and without the priors P in soft mask generation"에서 저자들은 dictionary-based priors를 포함하는 것이 결함 localization을 위한 Soft-IoU에서 17.9%의 이득을 가져온다는 것을 보여줍니다.
    *   **Selective vs. Global Attention**: 저자들은 selective attention mechanism을 Global Self-Attention (GSA)과 비교하여 정당화합니다. "Table 3: Results of RetouchFormer and ablative models on FFHQR"에 나타난 것처럼, selective attention을 GSA로 대체하면 PSNR이 45.13 dB에서 44.07 dB로 크게 떨어지며, dark circles와 같은 특정 결점을 지우는 데 실패합니다 ("Figure 7: Representative retouching images synthesized by RetouchFormer and ablative models on FR-wild images" 참조).
    *   **Multi-scale Tokens**: multi-scale features의 사용은 Table 3의 "w/ SPS" (Single Patch Size) 변형 모델을 통해 정당화되는데, 이는 4.93 dB의 성능 하락을 초래하여 다양한 크기의 결함을 처리하기 위해 multi-scale tokens가 필요함을 입증했습니다.
*   **효율성 및 수렴성**: 저자들은 "Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR"에서 모델의 효율성을 강조하며, RetouchFormer가 더 높은 PSNR로 수렴하고 base model의 최적 결과에 도달하는 속도가 최대 6배 더 빠르다는 점을 언급합니다.

**SOURCES**

- L. Xie, W. Xue, W. Xu, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Proceedings of Neural Information Processing Systems, 2017.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M. H. Yang, and L. Shao, "Multi-stage progressive image restoration," IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," IEEE Conference on Computer Vision and Pattern Recognition, 2021.