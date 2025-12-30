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
>After thoroughly reviewing and carefully examining the entire attached file, please provide a detailed, comprehensive summary of the paper’s full content.
>
>1. Organize your summary according to the paper’s section titles and explain each section in an easy-to-understand manner. But don't make the explanation too long-winded.
>2. For each section, please include all relevant **key equations** and **key references** that support your explanations.
>3. When citing references in the 'ANSWER' section, only use the reference titles (not authors’ names or numbers). Include the most important 10 core cited references.
>4. Make your answer as long as necessary.
>5. Finally, list core cited references of this paper in the 'SOURCES' section in an unnumbered IEEE format, including each paper’s title, authors, publication year, and publication venue.

>[!answer]


### 1. Abstract and Introduction

Face retouching is the process of beautifying a face image by removing imperfections while preserving the original image content and facial structures. The paper identifies two primary challenges in this field: the difficulty of automatically recognizing diverse face imperfections and the challenge of synthesizing realistic details in those regions, especially for severe cases like acne.

The authors propose **RetouchFormer**, a transformer-based approach that treats face retouching as a generalized "soft" version of face image inpainting. Unlike generic image enhancement methods that apply global transformations, RetouchFormer simultaneously identifies imperfections and synthesizes realistic content in those specific regions. As shown in **Figure 1: The main idea behind RetouchFormer**, the model uses a selective self-attention mechanism to replace the features of imperfections with those of normal skin.

Key contributions include:
- A semi-supervised framework that reduces dependence on large paired datasets.
- A "soft inpainting" formulation realized through joint imperfection prediction and local content synthesis.
- A selective self-attention mechanism based on clean face priors to strengthen interactions between imperfection queries and normal skin key-value pairs.

### 2. Related Work

The paper situates RetouchFormer within three main research areas:
- **CNN-based Image Translation**: This includes foundational works like **"Generative adversarial nets"** and **"Image-to-image translation with conditional adversarial networks"** (Pix2Pix), as well as high-quality restoration models like **"GPEN"** and **"MPRNet"**.
- **Vision Transformer**: The architecture draws inspiration from **"Attention is all you need"** and **"An image is worth 16x16 words: transformers for image recognition at scale"** (ViT). It also references **"Swin transformer: hierarchical vision transformer using shifted windows"** for handling multi-scale representations.
- **Face Retouching**: The work is most closely related to **"AutoRetouch: automatic professional face retouching"** and **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe). RetouchFormer improves upon these by using a simpler structure and more adaptive content synthesis via selective attention.

### 3. Proposed Approach

The architecture follows an encoder-transformer-decoder design, as illustrated in **Figure 2: The workflow of RetouchFormer**.

#### 3.1 Clean Face Priors
To capture what a "clean" face should look like, the authors learn a feature dictionary $\mathcal{P}$ using Vector Quantization (VQ), as described in **"Neural discrete representation learning"**. The feature vector at any location $(i, j)$ of the encoder feature map $E(y_-)$ is approximated by the nearest element in the dictionary:
$$\color{orange}
\rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2
$$
where $y_-$ represents a retouching image. This dictionary helps in identifying imperfections by comparing the original image with its reconstruction from clean priors.

#### 3.2 Soft Mask Generation
A two-stream U-Net structure (denoted as $H$ and $\bar{H}$) is used to generate a mask indicating imperfection regions. The model assumes that raw images with imperfections cannot be perfectly reconstructed by the clean face dictionary. For paired data, the loss is defined as:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)}\|\bar{H}(x_a, x_a) - \phi(x_a, y_a)\|_2^2
$$
where $\phi$ measures the difference between the raw image $x_a$ and the ground truth $y_a$. To handle unpaired data, the model uses an Exponential Moving Average (EMA) update strategy, inspired by **"Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"**.

#### 3.3 Selective Self-attention
The core innovation is the selective self-attention mechanism. It suppresses information from imperfection regions and replaces it with features from normal skin. The queries ($Q$), keys ($K$), and values ($V$) are formulated as:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q,
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k,
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
where $f_H$ is the weighting map derived from the predicted mask. This ensures that the "queries" (imperfections) look for "keys/values" (normal skin) to fill in the gaps.

#### 3.4 Model Training
The training involves multiple loss functions to ensure high-quality reconstruction and retouching:
- **Prior Loss**: Optimizes the encoder and dictionary.
$$\color{orange}
L_{prior} = \mathbb{E}_{y_-} \left[ \sum_{(i,j)} \|E(y_-)|_{(i,j)} - \rho_*|_{(i,j)}\|_2^2 \right]
$$
- **Consistency Loss**: Ensures the synthesized image matches the ground truth in terms of pixel-wise and perceptual similarity (using VGG-19 features as per **"The unreasonable effectiveness of deep features as a perceptual metric"**).
- **Adversarial Loss**: A discriminator $D$ is used to ensure the results match the statistics of real clean faces.

The total optimization is a multi-objective task:
$$\color{orange}
\min_{E,T,G} L_{prior} + L_{con}^{rec} + L_{con}^{x2y} + L_{adv}^{syn}
$$

### 4. Experiments

#### 4.1 Datasets and Metrics
The model was evaluated on the **FFHQR** dataset (70k pairs) and the **FR-wild** dataset (1,000 in-the-wild images). Metrics used include PSNR, SSIM, and LPIPS.

#### 4.2 Quantitative and Qualitative Results
As shown in **Table 1: Quantitative comparison with competing methods on FFHQR**, RetouchFormer outperforms state-of-the-art methods like BPFRe and ABPN across all metrics. For instance, at $\gamma=0.1$ (10% paired data), RetouchFormer achieves a PSNR of 45.13 dB, which is significantly higher than its competitors.

Qualitatively, **Figure 4: Visual comparison with competing methods on FR-wild images** demonstrates that RetouchFormer effectively removes large-scale blemishes and dark circles that other methods miss. **Figure 5** further showcases its ability to erase acne and smooth skin with high fidelity.

### 5. Analysis of Design Elements

Ablation studies confirmed the importance of each component:
- **Clean Face Prior**: Disabling the prior $\mathcal{P}$ led to a 17.9% drop in Soft-IoU for imperfection prediction (**Table 2**).
- **Multi-scale Tokens**: Using a single patch size (SPS) instead of multi-scale features caused a PSNR drop of 4.93 dB (**Table 3**).
- **Selective Self-attention**: Replacing it with Global Self-Attention (GSA) reduced the PSNR by 1.06 dB and failed to erase certain pockmarks (**Figure 7**).
- **Unpaired Data**: Training without unpaired data ($X_u$) significantly degraded performance, proving the value of the semi-supervised approach.

### 6. Conclusion

RetouchFormer introduces a robust transformer-based framework for face retouching. By combining clean face priors with a selective self-attention mechanism, it effectively localizes and repairs facial imperfections. The semi-supervised training strategy allows the model to achieve superior results even with limited paired data, making it highly practical for real-world applications.

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
>After carefully reviewing the “Introduction” and “Reference” sections of the attached file, please provide a thorough and detailed response by addressing the following points:
>
>For each section of the 'ANSWER', please include any relevant **key equations** and any pertinent **references** using only the paper titles (not authors’ names or numbers).
>
>1. Define the core task (including its definition, inputs/outputs, goals, and significance).
>2. Describe the main challenges or problems related to this task and the limitations of previous methods. Please specify the paper **titles** of these previous methods in the 'ANSWER.'
>3. Explain the overall solution and key ideas proposed by the authors to address the problems.
>4. Detail the main contributions of the paper (e.g., theoretical or practical achievements, performance improvements, new model architectures, or problem definitions).
>
>Make your answer as lengthy as needed. Finally, list all references relevant to the 'ANSWER' in the 'SOURCES' section in an unnumbered IEEE format, including each reference’s title, authors, publication year, and publication venue.

>[!answer]


### 1. Core Task: Face Retouching

Face retouching is defined as the process of beautifying a face image by removing various types of imperfections while preserving the original image content and essential facial structures as much as possible. As stated in the "Introduction," the **input** is typically a face image that has suffered from numerous types of imperfections, such as acne, spots, or wrinkles. The **output** is a clean, high-fidelity face image with a flawless skin tone.

The primary **goal** of this task is to automatically recognize diverse face imperfections and synthesize realistic skin details to fill those regions. The **significance** of this task lies in its potential to automate a process that currently requires professional retouchers several hours of manual editing. By providing an automated solution, high-quality face beautification becomes accessible and efficient for general users. This concept is visually introduced in **Figure 1: The main idea behind RetouchFormer**, which shows the transition from a source image with imperfections to a clean retouching result.

### 2. Main Challenges and Limitations of Previous Methods

According to the "Introduction," the field of face retouching faces several critical **challenges**:
- **Imperfection Recognition**: Face imperfections are highly diverse in the real world, making it difficult for models to automatically and accurately identify them.
- **Realistic Detail Synthesis**: It is challenging to fill in realistic details, particularly for moderate-to-severe acne regions, while simultaneously preserving the underlying structure of essential facial features.

**Limitations of Previous Methods**:
- **Traditional Methods**: Early approaches relied on local smoothing operators for blemish removal, which often resulted in a loss of natural skin texture.
- **Generic Image-to-Image Translation**: Methods such as those described in **"Image-to-image translation with conditional adversarial networks"** (Pix2Pix) or generic image enhancement models like **"MPRNet: Multi-stage progressive image restoration"** often lack specific imperfection localization. This typically leads to the incomplete removal of blemishes, especially at larger scales.
- **Specific Face Retouching Models**: Recent models like **"AutoRetouch: automatic professional face retouching"** and **"ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"** primarily consider image-level transformations using global convolution filters. These methods fail to adaptively complete specific imperfection regions and usually require large amounts of pairwise retouching data (original vs. professionally retouched) for training, which is expensive to collect.
- **Two-Stage Approaches**: While **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe) attempted a progressive restoration, it utilizes a more complex two-stage method compared to the streamlined approach proposed in this paper.

### 3. Overall Solution and Key Ideas

The authors propose **RetouchFormer**, a transformer-based approach that treats face retouching as a generalized "soft" version of face image inpainting. The model is designed to simultaneously identify imperfections and synthesize realistic content in those regions using an encoder-transformer-decoder architecture, as shown in **Figure 2: The workflow of RetouchFormer**.

**Key Ideas**:
- **Clean Face Priors**: The model learns a latent dictionary $\mathcal{P}$ to capture clean face priors using vector quantization, a technique detailed in **"Neural discrete representation learning"**. This dictionary allows the model to approximate clean features:
$$\color{orange}
\rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2
$$
- **Soft Mask Generation**: To address the lack of localization in previous methods, RetouchFormer predicts imperfection regions via a reconstruction-oriented localization module. It uses a two-stream U-Net structure to generate a mask $H$ that indicates where imperfections are located. For paired data, the loss is:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)}\|\bar{H}(x_a, x_a) - \phi(x_a, y_a)\|_2^2
$$
To leverage unpaired data, the model employs an Exponential Moving Average (EMA) strategy, inspired by **"Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results"**.
- **Selective Self-Attention**: Based on the predicted mask, the model performs "selective self-attention." This mechanism explicitly suppresses features from imperfection regions (acting as queries) and replaces them with features synthesized from normal skin regions (acting as keys and values). The queries $Q$, keys $K$, and values $V$ are formulated as:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
- **Multi-scale Feature Tokens**: The transformer uses multi-scale feature tokens to provide flexibility in dealing with imperfections of various sizes and scales.

### 4. Main Contributions

The paper lists three primary contributions in the "Introduction" and "Proposed Approach" sections:
- **Semi-supervised Framework**: The authors propose a semi-supervised face retouching transformer that significantly reduces the dependence on expensive paired retouching data for model training.
- **Soft Inpainting Formulation**: The model characterizes face retouching as a "soft inpainting" task. This is realized through a joint process of approximate imperfection prediction and local content synthesis, allowing for more precise and adaptive corrections than global filter-based methods.
- **Prior-Based Selective Self-Attention**: The introduction of a selective self-attention mechanism based on clean face priors. This mechanism strengthens the spatial interactions between the queries of imperfection regions and the key-value pairs of normal skin regions, ensuring that the filled content is both realistic and contextually appropriate.
- **Performance and Efficiency**: RetouchFormer outperforms advanced face retouching methods in terms of fidelity and realism. As noted in the "Abstract," the design elements bring greater effectiveness and efficiency, allowing the model to converge faster and achieve higher quality results (e.g., 1.3X PSNR improvement and 6X efficiency over base models as shown in **Figure 3**).

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
>After thoroughly reviewing the “Related Works” and “Reference” sections of the attached file, please provide a comprehensive and detailed answer by addressing the following:
>
>1. According to the authors’ classification criteria, please list and categorize **all** related studies mentioned. Summarize the core idea of each study and note any limitations highlighted by the authors. When citing references, use only **the paper titles** (not authors’ names or numbers)
>2. Discuss how the new approach in this paper differs from and improves upon these existing studies.
>
>Make your answer as lengthy as needed. Finally, please list **all** cited references in a 'SOURCES' section **without omissions** using an unnumbered IEEE format, including the title, authors, publication year, and publication venue for each reference.

>[!answer]


Based on the "Related Work" and "Introduction" sections of the provided paper, the authors categorize existing research into three main areas: CNN-based Image Translation, Vision Transformer, and Face Retouching.

### 1. Categorization of Related Studies

#### **CNN-based Image Translation**
This category encompasses the use of Convolutional Neural Networks (CNNs) for mapping images from one domain to another.

*   **Foundational Generative Models**:
    *   **"Generative adversarial nets"**: Introduced the capability for high-fidelity image synthesis using an adversarial framework.
    *   **"Image-to-image translation with conditional adversarial networks"** (Pix2Pix): Learned a mapping to minimize distribution discrepancy and pixel-wise discrepancy between source and target domains. **Limitation**: It requires pairwise training data, which is expensive and time-consuming to collect.
*   **Unpaired Image Translation**:
    *   **"Unsupervised image-to-image translation networks"**, **"Unpaired image-to-image translation using cycle-consistent adversarial networks"**, and **"Learning to discover cross-domain relations with generative adversarial networks"**: These methods addressed the lack of paired data by performing data distribution alignment or using two-way mapping to impose cycle consistency regularization.
*   **High-Quality Restoration and Multi-stage Models**:
    *   **"GAN prior embedded network for blind face restoration in the wild"** (GPEN): Combined a U-shaped CNN with a GAN to ensure high-quality image generation.
    *   **"MPRNet: Multi-stage progressive image restoration"**: Designed for multi-stage image translation, benefiting from high-level contextual information and spatial details.
*   **Semantic Editing and GAN Inversion**:
    *   **"GANalyze: toward visual definitions of cognitive image properties"**, **"On the “steerability” of generative adversarial networks"**, **"Editing in style: uncovering the local semantics of GANs"**, **"Closed-form factorization of latent semantics in GANs"**, and **"StyleSpace analysis: disentangled controls for StyleGAN image generation"**: These studies observed that the latent space of pre-trained GANs possesses semantic organization, allowing for semantic editing.
    *   **"Image processing using multi-code GAN prior"**, **"Encoding in style: a StyleGan encoder for image-to-image translation"**, and **"Designing an encoder for StyleGan image manipulation"**: These methods leveraged high-fidelity GAN inversion to project images back into the latent space for manipulation.
    *   **"StyleFlow: attribute-conditioned exploration of StyleGAN Generated images using conditional continuous normalizing flows"**: Employed a continuous normalizing flow model to learn non-linear transformations in the latent space.
    *   **"StyleCLIP: text-driven manipulation of StyleGAN imagery"**: Utilized language-vision pre-training for image editing.
    *   **General Limitation**: Most of these methods perform global latent transformations for semantic image translation without considering the specific variations in input images.

#### **Vision Transformer**
This category focuses on the adaptation of the Transformer architecture, originally from natural language processing, to computer vision tasks.

*   **Foundational Architectures**:
    *   **"Attention is all you need"** and **"BERT: Pre-training of deep bidirectional transformers for language understanding"**: Established the core Transformer and self-attention mechanisms.
*   **Image Recognition and Processing**:
    *   **"An image is worth 16x16 words: transformers for image recognition at scale"** (ViT) and **"CvR: Introducing convolutions to vision transformers"**: Demonstrated superior performance in representation learning.
    *   **"Swin transformer: hierarchical vision transformer using shifted windows"**: Learnt a hierarchical representation with shifted windows to address high image resolutions and scale variations.
    *   **"Pre-trained image processing transformer"** (IPT): Adopted a large-scale pre-trained model for low-level tasks like denoising and super-resolution.
*   **Restoration and Editing**:
    *   **"RestoreFormer: High-quality blind face restoration from undegraded key-value pairs"**: Performed cross-attention with priors in terms of high-quality key-value pairs to improve blind face restoration.
    *   **"Style transformer for image inversion and editing"**, **"End-to-end object detection with transformers"**, and **"Deformable DETR: Deformable transformers for end-to-end object detection"**: Applied Transformer-based methods to image editing and object detection.

#### **Face Retouching**
This category includes methods specifically designed to beautify face images by removing blemishes.

*   **Traditional and Smoothing Methods**:
    *   **"Nonlinear digital filters for beautifying facial images in multimedia systems"**, **"Detection and inpainting of facial wrinkles using texture orientation fields and Markov random field modeling"**, **"Automatic freckles detection and retouching"**, and **"Data-driven enhancement of facial attractiveness"**: These methods typically applied local smoothing operators or texture orientation fields for blemish removal.
    *   **"FabSoften: face beautification via dynamic skin smoothing, guided feathering and texture restoration"**: Adopted a wavelet-based skin texture restoration method to restore fine-grained details.
*   **Deep Learning-based Retouching**:
    *   **"AutoRetouch: automatic professional face retouching"**: An attempt to perform deep image translation for the retouching task.
    *   **"ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"**: Proposed an adaptive blend pyramid network for fast local retouching.
    *   **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe): Adopted a two-stage retouching strategy to progressively restore a clean face.
    *   **General Limitation**: Many of these methods (like AutoRetouch and ABPN) only consider image-level transformations via global convolution filters, meaning imperfection regions cannot be adaptively completed. Additionally, they often require extensive pairwise data.

### 2. Differences and Improvements of RetouchFormer

RetouchFormer introduces several key improvements over the existing studies mentioned above:

*   **Adaptive Local Completion vs. Global Transformation**: Unlike generic image translation methods (**"MPRNet: Multi-stage progressive image restoration"**) or early retouching models (**"AutoRetouch: automatic professional face retouching"**) that apply global filters, RetouchFormer treats retouching as a "soft inpainting" task. It explicitly identifies imperfection regions and adaptively synthesizes content specifically for those areas.
*   **Selective Self-Attention Mechanism**: While models like **"RestoreFormer: High-quality blind face restoration from undegraded key-value pairs"** use priors for restoration, RetouchFormer introduces a "selective self-attention" mechanism. This mechanism uses clean face priors to suppress interactions within imperfection regions and strengthen interactions between imperfection "queries" and normal skin "key-value" pairs. This ensures that local content is synthesized from healthy skin features.
*   **Semi-supervised Learning**: To overcome the limitation of requiring expensive paired data (as seen in **"Image-to-image translation with conditional adversarial networks"**), RetouchFormer is designed for semi-supervised settings. It utilizes both paired and unpaired data, stabilized by a reconstruction-oriented localization module and adversarial training.
*   **Multi-scale Flexibility**: By adopting a multi-scale encoding scheme in the transformer, RetouchFormer can hierarchically represent content. This allows it to handle imperfections at various scales more effectively than methods using fixed-scale tokens or single-scale filters.
*   **Structural Simplicity and Efficiency**: Compared to two-stage methods like **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"** (BPFRe), RetouchFormer adopts a simpler encoder-transformer-decoder architecture that achieves better performance and higher efficiency (matching best results up to 6 times faster, as shown in **Figure 3: Convergence properties of RetouchFormer**).

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
>After reviewing the “Methodology” and “Reference” sections of the attached file, please provide a detailed answer addressing each of the following questions:
>
>1. What are the key terms and symbols necessary to understand the method proposed in this paper? Explain each term and symbol clearly.
>2. Provide a step-by-step explanation of the mathematical background (equations and theories) required for understanding this method.
>3. Clearly explain and organize the essential prior work referenced in the paper, making sure to include relevant references for that section.
>4. Show how these concepts connect to the subsequent model description.
>
>In your 'ANSWER' section for each question, please include the relevant key equations and cite the references by **title** only (not by authors’ names or numbers). Make your explanation as long as necessary. Finally, in the 'SOURCES' section, list all references relevant to the 'ANSWER' used in unnumbered IEEE format with the titles, authors, publication years, and publication venues.

>[!answer]


### 1. Key Terms and Symbols

To understand the methodology of RetouchFormer, it is essential to define the following terms and symbols used throughout the paper:

- **$X$**: The complete set of training data, which is composed of a small subset of paired data $X_a$ and a larger subset of unpaired data $X_u$ (i.e., $X = X_a \cup X_u$). This is detailed in the "Overview" section of the "Proposed Approach."
- **$X_a = \{(x_a, y_a)\}$**: The subset of paired raw-retouching images, where $x_a$ is the raw image and $y_a$ is the professionally retouched ground truth.
- **$X_u = \{x_u, y_u\}$**: The subset of unpaired images, where $x_u$ and $y_u$ may correspond to different identities and do not have a direct mapping.
- **$\gamma$**: The proportion of paired training data, defined as $\gamma = \|X_a\| / \|X\|$. This is used to evaluate the model's performance under different levels of supervision, as mentioned in the "Semi-supervised Settings" section.
- **$P = \{\rho_1, \rho_2, \dots, \rho_K\}$**: A feature dictionary learned to capture "clean face priors." It consists of $K$ elements used for reconstructing features in the encoder space.
- **$E, T, G$**: The core components of the architecture: the Encoder ($E$), the Transformer ($T$), and the Decoder ($G$).
- **$H$ and $\bar{H}$**: Two streams of a U-Net structure used for "Soft Mask Generation." $\bar{H}$ produces the final imperfection mask used for selective self-attention.
- **$f_H(x_-)$**: A weighting map derived from the mask $\bar{H}$, which indicates the location and intensity of imperfections.
- **$Q, K, V$**: The Query, Key, and Value matrices used in the self-attention mechanism.
- **$\otimes$**: The Hadamard product (element-wise multiplication), used to weight features based on the imperfection mask.
- **$\phi(\cdot, \cdot)$**: A function measuring image difference at each pixel location, such as Euclidean distance, used in mask evaluation.

These terms are introduced in the "Proposed Approach" and "RetouchFormer" sections to describe the semi-supervised framework and the specific mechanisms for imperfection localization and synthesis.

---

### 2. Mathematical Background and Step-by-Step Explanation

The RetouchFormer method is built upon several mathematical foundations, primarily focusing on prior learning, mask generation, and selective attention.

#### Step 1: Learning Clean Face Priors via Vector Quantization
The model first learns a dictionary $P$ to represent clean skin features. For a given feature vector at location $(i, j)$ of the encoded image $E(y_-)$, the nearest element in the dictionary is found using:
$$\color{orange}
\rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(y_-)|(i,j) - \rho_k\|_2^2
$$
This process, described in "Clean Face Priors," allows the model to reconstruct a "clean" version of the face, denoted as $\tilde{x}_- = R(E_\rho(x_-))$, where $R$ is a reconstruction decoder.

#### Step 2: Soft Mask Generation
To identify imperfections, the raw image $x_-$ and its reconstruction $\tilde{x}_-$ are concatenated and passed through a two-stream U-Net ($H$ and $\bar{H}$). For paired data, the mask is evaluated against the actual difference between the raw and retouched images:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)}\|\bar{H}(x_a, x_a) - \phi(x_a, y_a)\|_2^2
$$
For unpaired data, a confidence-based loss is used to ensure the prediction is reliable:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,y_a)} [-I_{\{\phi(x_a,y_a)>0\}} \log \sigma(H(x_a, x_a))] + \lambda \mathbb{E}_{x_u} [-\sigma(H(x_u, x_u)) \log \sigma(H(x_u, x_u))]
$$
As explained in "Soft Mask Generation," the parameters of $\bar{H}$ are updated using an Exponential Moving Average (EMA) of $H$:
$$\color{orange}
\theta_{\bar{H}} \leftarrow \mu\theta_{\bar{H}} + (1 - \mu)\theta_H
$$

#### Step 3: Selective Self-Attention
The core innovation is the selective self-attention mechanism. Unlike standard attention, it uses the weighting map $f_H(x_-)$ to focus queries on imperfections and keys/values on normal skin:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
The attended feature map $\Delta f_E$ is calculated as:
$$\color{orange}
\Delta f_E = \text{softmax}(QK^T / \sqrt{\Lambda})V
$$
Finally, the features are updated by replacing imperfection features with the synthesized "normal" features:
$$\color{orange}
\hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
$$
This formulation, found in "Selective Self-attention," ensures that the content in imperfection regions is synthesized from the context of healthy skin.

---

### 3. Essential Prior Work

The paper builds upon several key areas of research in computer vision:

- **CNN-based Image Translation**: The paper references "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix) as a foundational method for mapping source domains to target domains. It also mentions "Multi-stage progressive image restoration" (MPRNet) and "GAN prior embedded network for blind face restoration in the wild" (GPEN) for high-quality image synthesis.
- **Vision Transformers**: The architecture is inspired by "An image is worth 16x16 words: transformers for image recognition at scale" (ViT) and "Swin transformer: hierarchical vision transformer using shifted windows" (SwinTransformer). Specifically, it relates to "RestoreFormer: High-quality blind face restoration from degraded key-value pairs," which uses a dictionary-based approach for face restoration.
- **Face Retouching**: Previous specialized methods include "AutoRetouch: automatic professional face retouching" and "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo." The paper also compares itself to "Blemish-Aware and Progressive Face Retouching With Limited Paired Data" (BPFRe), which uses a two-stage restoration strategy.
- **Vector Quantization (VQ)**: The method for learning clean face priors is based on "Neural discrete representation learning," which introduced the VQ-VAE framework for learning discrete latent representations.
- **Semi-supervised Learning**: The use of EMA for stable mask generation is inspired by "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results."

---

### 4. Connection to the Model Description

These concepts are systematically integrated into the RetouchFormer architecture as follows:

1.  **Prior Learning to Localization**: The concept of **Vector Quantization** from "Neural discrete representation learning" is used to create the feature dictionary $P$. Because the dictionary only contains "clean" face features, the reconstruction error between the input and the dictionary-based reconstruction provides a "clue" for the **Soft Mask Generation** module to identify imperfections.
2.  **Localization to Synthesis**: The predicted mask $f_H(x_-)$ directly modulates the **Selective Self-Attention** mechanism. By using the mask to define Queries (imperfections) and the inverse of the mask to define Keys and Values (normal skin), the model forces the Transformer to "fill in" the blemishes using information from healthy skin areas. This is a "soft" version of the inpainting task described in the "Introduction."
3.  **Multi-scale Representation**: Drawing from **SwinTransformer**, the model uses multi-scale feature tokens ($f_E$) to handle imperfections of various sizes (e.g., small spots vs. large acne regions), as discussed in "Selective Self-attention."
4.  **Semi-supervised Stability**: The **EMA** update rule from "Mean teachers are better role models" ensures that the imperfection prediction remains stable even when training with limited paired data ($X_a$), allowing the model to leverage the large amount of unpaired data ($X_u$) effectively.
5.  **Adversarial Refinement**: Finally, the **GAN** framework from "Generative adversarial nets" is employed via a discriminator $D$ that receives the predicted mask, forcing the generator to produce realistic skin textures that are indistinguishable from real "clean" skin.

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
>After reviewing the “Methodology” and “Reference” sections of the attached file, please provide a detailed answer focusing on the following:
>
>1. Structure of the Framework:
>    - Describe how the entire system or framework is structured, covering each **module** or **component** without omission. Modules and components are not data. It's part of the model.
>    - Specify the accurate baseline or backbone for each module or component (with the corresponding references by **title**) and the reasons for choosing them.
>    - Describe any structural improvements proposed by the authors for each component, including the final network architecture (layer composition, main modules, inputs/outputs, etc.) and the role or function of each component.
>
>2. Data Flow:
>    - Explain, step by step, how input data moves through the framework to produce the final output. Include any relevant equations.
>
>3. Framework Figure:
>    - If there are figures (e.g., Figure X) illustrating the framework or model architecture, specify their figure numbers and describe the important parts in detail.
>
>In your 'ANSWER' section, please include relevant key equations where applicable, citing references by **title** only (not by authors’ names or numbers) for any baselines or backbones. Make your answer as long as necessary. Finally, provide all references relevant the 'ANSWER' to the 'SOURCES' section using unnumbered IEEE format (titles, authors, publication years, and publication venues).

>[!answer]


The RetouchFormer is a transformer-based framework designed for high-quality face retouching. It operates by identifying facial imperfections and synthesizing realistic skin textures in those regions using a semi-supervised approach. The system is built on an encoder-transformer-decoder architecture, enhanced with a prior-based selective self-attention mechanism.

### 1. Structure of the Framework

The RetouchFormer framework consists of several interconnected modules, each serving a specific role in the retouching process. As illustrated in "Figure 2: The workflow of RetouchFormer," the system includes the following components:

#### A. Encoder ($E$) and Multi-scale Feature Extraction
The encoder $E$ is responsible for extracting features from the input image $x_-$. Unlike standard transformers that might use a fixed scale, RetouchFormer adopts a **multi-scale encoding scheme**.
- **Baseline/Backbone**: The multi-scale approach is inspired by hierarchical representations found in "Swin transformer: hierarchical vision transformer using shifted windows."
- **Reason for Choice**: This allows the model to represent content in local regions hierarchically, which is crucial because facial imperfections (like acne or pockmarks) vary substantially in scale.
- **Structural Improvement**: The encoder extracts spatial patches from multiple intermediate layers to address imperfections of different scales effectively.

#### B. Feature Dictionary ($P$) and Reconstruction Decoder ($R$)
This module captures "clean face priors" to help identify imperfections.
- **Baseline/Backbone**: This component is based on **Vector Quantization (VQ)** as described in "Neural discrete representation learning."
- **Reason for Choice**: VQ allows the model to learn a discrete latent space (a dictionary) that represents high-quality, clean skin textures.
- **Role**: The dictionary $P$ is optimized to reconstruct features of clean retouching images. When a raw image with imperfections is passed through, the dictionary-based reconstruction $\tilde{x}_- = R(E_\rho(x_-))$ will lack those imperfections, providing a clue for localization.

#### C. Soft Mask Generation Network ($\{H, \bar{H}\}$)
This module predicts the location of imperfections.
- **Baseline/Backbone**: It utilizes a **U-Net** structure, a common choice for image-to-image translation tasks as seen in "Image-to-Image Translation with Conditional Adversarial Networks."
- **Structural Improvement**: The authors propose a **two-stream structure** ($H$ and $\bar{H}$). The stream $\bar{H}$ is updated using the **Exponential Moving Average (EMA)** of $H$, a technique from "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results."
- **Role**: It takes the raw image $x_-$ and the reconstructed image $\tilde{x}_-$ as inputs to generate a soft mask $\bar{H}(x_-)$, which indicates the regions to be retouched.

#### D. Transformer ($T$) with Selective Self-Attention Blocks (SSAB)
The transformer $T$ is the core processing unit that transforms features of imperfection regions into normal skin features.
- **Baseline/Backbone**: The transformer architecture is based on "Attention is all you need" and "An image is worth 16x16 words: transformers for image recognition at scale."
- **Structural Improvement**: The authors introduce the **Selective Self-Attention (SSA)** mechanism. Instead of global attention, SSA uses the predicted mask to limit spatial interactions.
- **Role**: It suppresses information from imperfection regions and replaces it with features aggregated from normal skin regions.

#### E. Synthesis Decoder ($G$) and Discriminator ($D$)
- **Synthesis Decoder ($G$)**: This component takes the transformed features from the transformer and synthesizes the final high-quality retouching image $\hat{y}^x_-$.
- **Discriminator ($D$)**: Based on "Generative adversarial nets," the discriminator performs pixel-level real-fake identification.
- **Structural Improvement**: To improve results, the predicted masks $\bar{H}(x_-)$ are injected into the discriminator, forcing it to pay more attention to the synthesized details in the imperfection regions.

---

### 2. Data Flow

The data flow through RetouchFormer follows a systematic path to transform a raw image into a retouched one:

1.  **Feature Extraction**: The raw input image $x_-$ is passed through the encoder $E$ to produce multi-scale features $f_E(x_-)$.
2.  **Prior-based Reconstruction**: The encoder features $E(x_-)$ are quantized using the dictionary $P$. The nearest dictionary element $\rho_*$ is found via:
    $$\color{orange}
    \rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(y_-)|(i,j) - \rho_k\|_2^2
    $$
    The reconstruction decoder $R$ then produces a "clean" reconstructed image $\tilde{x}_- = R(E_\rho(x_-))$.
3.  **Imperfection Localization**: The raw image $x_-$ and reconstructed image $\tilde{x}_-$ are concatenated and fed into the mask generation network $\{H, \bar{H}\}$ to produce the soft mask $\bar{H}(x_-)$ and its corresponding weighting map $f_H(x_-)$.
4.  **Selective Transformation**: Inside the Transformer $T$, the features $f_E(x_-)$ and the weighting map $f_H(x_-)$ are used to compute Queries ($Q$), Keys ($K$), and Values ($V$):
    $$\color{orange}
    Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
    $$
    $$\color{orange}
    K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
    $$
    $$\color{orange}
    V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
    $$
    The attention mechanism then synthesizes new features $\Delta f_E$ for the imperfection regions using the context of normal skin. The final transformed features $\hat{f}_E(x_-)$ are formed by:
    $$\color{orange}
    \hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
    $$
5.  **Final Synthesis**: The transformed features are passed through the decoder $G$ to generate the final output image $\hat{y}^x_- = G(\hat{f}_E(x_-))$.

---

### 3. Framework Figure

The primary illustration of the architecture is **Figure 2**, titled "The workflow of RetouchFormer." The figure details the following:

-   **Left Side**: Shows the input image $x_-$ entering the **Encoder $E$**. It depicts the path to the **Dictionary $P$** and the **Reconstruction Decoder $R$** to create $\tilde{x}_-$.
-   **Center-Left**: Illustrates the **Soft mask generation network**, which takes the source $x$ and the reconstruction $\tilde{x}$ to produce the mask $\bar{H}(x)$.
-   **Center**: Displays the **Transformer $T$**, composed of multiple **Selective Self-Attention Blocks (SSAB)**. It shows how multi-scale features $f_E(x)$ and weights $f_H(x)$ are processed.
-   **Right Side**: Shows the **Decoder $G$** producing the output $\hat{y}^x$. It also shows the **Discriminator $D$** receiving the output image, the mask, and the original image to perform real-fake prediction.
-   **Inset (Top Right)**: Provides a detailed view of the **Selective Self-Attention-based Block (SSAB)**, highlighting the Hadamard product ($\otimes$) operations that separate imperfection features (Queries) from normal skin features (Keys and Values).

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," International Conference on Computer Vision, 2021.
- P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," IEEE Conference on Computer Vision and Pattern Recognition, 2017.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," Advances in Neural Information Processing Systems, 2017.
- A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: transformers for image recognition at scale," International Conference on Learning Representations, 2021.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Neural Information Processing Systems, 2014.
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," Neural Information Processing Systems, 2017.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.


## Training
>[!question]
>After reviewing the “Methodology” and “Reference” sections of the attached file, please provide a detailed, step-by-step answer to the following questions:
>
>1. Training Process:
>    - Is training done in phases or is it not broken down into phases? What are the training phases (e.g., pre-training, fine-tuning, stepwise loss combination, curriculum learning)?
>    - What is the role or significance of each phase?
>    - If the authors were inspired by other references for the training process, cite those references by **title**.
>
>2. Loss Function:
>    - Which loss functions (including any custom ones) are used in each phase, and what outputs do they apply to?
>    - Include relevant equations and references by **title**, explaining their purpose, function and training target(object being trained or optimized) in detail.
>
>3. Optimization:
>    - Which optimization methods and hyperparameters (e.g., learning rate schedule) are used, and how are they set?
>
>4. Special Training Techniques:
>    - If any special training techniques (e.g., curriculum learning, multi-task learning, self-supervised learning, semi-supervised learning, regularization) are applied, explain the objective and method for each. If not, state so.
>
>Please write your 'ANSWER' as long as necessary, including all relevant formulas and references in each section. In the 'SOURCES' section, list all references relevant to the 'ANSWER' used in unnumbered IEEE format (titles, authors, publication years, and publication venues).

>[!answer]


The training process of RetouchFormer is a sophisticated semi-supervised framework that integrates prior learning, imperfection localization, and high-fidelity image synthesis. The authors optimize the system by balancing multiple loss terms across different constituent networks.

### 1. Training Process

The training of RetouchFormer is not explicitly divided into sequential temporal phases (like pre-training followed by fine-tuning) but is instead structured as a **joint optimization of three main aspects**: prior-based reconstruction, imperfection prediction, and retouching evaluation. As stated in the "Model Training" section, the constituent networks are optimized with different loss terms, where only the encoder $E$ is shared across tasks, while other networks are exclusive to individual tasks.

The three core aspects of the training process are:
- **Prior-based Reconstruction**: This aspect focuses on learning the "clean face prior" using a feature dictionary. The encoder $E$ and dictionary $P$ are optimized to reconstruct the features of retouching images $y_-$. This process is inspired by the **Vector Quantization (VQ)** algorithm from "Neural discrete representation learning."
- **Imperfection Prediction**: This involves training the mask generation networks $H$ and $\bar{H}$ to identify blemishes. The authors use a semi-supervised approach where paired data provides direct supervision, and unpaired data is handled via a confidence-based loss.
- **Retouching Evaluation**: This aspect ensures the final synthesized image $\hat{y}^x_-$ matches the ground truth (for paired data) and maintains high realism (via adversarial training).

The overall optimization formulation is expressed in the "Model Training" section as:
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

The RetouchFormer utilizes a variety of loss functions to optimize different components of the framework.

#### A. Prior-based Reconstruction Losses
To capture high-quality clean face priors, the encoder $E$ and dictionary $P$ are optimized using the following:
- **Prior Loss ($L_{prior}$)**: This loss ensures the dictionary can accurately represent the features of clean images $y_- \in \{ya, yu\}$.
$$\color{orange}
L_{prior} = \mathbb{E}_{y_-} \left[ \sum_{(i,j)} \|E(y_-)|_{(i,j)} - \rho_*|_{(i,j)}\|_2^2 \right]
$$
- **Reconstruction Consistency Loss ($L_{con}^{rec}$)**: To ensure high-quality reconstruction, a combination of $L_1$ loss and a perceptual loss (using a pre-trained VGG-19 network) is applied:
$$\color{orange}
L_{con}^{rec} = \mathbb{E}_{y_-} [\eta\|y_- - \tilde{y}_-\|_1 + \|V(y_-) - V(\tilde{y}_-)\|_2^2]
$$
where $\tilde{y}_- = R(E_\rho(y_-))$ and $V(\cdot)$ denotes VGG-19 features. This is detailed in the "Model Training" section.

#### B. Imperfection Prediction Losses
The mask generation network $H$ is trained to identify imperfections:
- **Paired Mask Loss ($L_{\bar{H}}$)**: For paired data, the output of $\bar{H}$ is compared to the actual difference between raw and retouched images:
$$\color{orange}
L_{\bar{H}} = \mathbb{E}_{(x_a,ya)} \|\bar{H}(x_a, x_a) - \phi(x_a, ya)\|_2^2
$$
- **Confidence-based Loss ($L_H$)**: For both paired and unpaired data, this loss enforces confident predictions:
$$\color{orange}
L_H = \mathbb{E}_{(x_a,ya)} [-I_{\{\phi(x_a,ya)>0\}} \log \sigma(H(x_a, x_a))] + \lambda \mathbb{E}_{x_u} [-\sigma(\bar{H}(x_u, x_u)) \log \sigma(H(x_u, x_u))]
$$
This approach is inspired by the **Exponential Moving Average (EMA)** technique from "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results."

#### C. Retouching Evaluation Losses
- **Paired Consistency Loss ($L_{con}^{x2y}$)**: Measures the degree of consistency between the synthesized image $\hat{y}^x_a$ and the ground truth $y_a$:
$$\color{orange}
L_{con}^{x2y} = \mathbb{E}_{(x_a,ya)} [\zeta\|y_a - \hat{y}^x_a\|_1 + \|V(y_a) - V(\hat{y}^x_a)\|_2^2]
$$
- **Adversarial Losses ($L_{adv}^{syn}$ and $L_{adv}^{disc}$)**: Based on "Generative adversarial nets," these losses improve realism. The discriminator $D$ performs pixel-level real-fake identification, with the imperfection mask $\bar{H}$ injected to focus attention on retouched regions:
$$\color{orange}
L_{adv}^{syn} = \mathbb{E}_{x_-} [\log(1 - D(\hat{y}^x_-, \bar{H}(x_-)))]
$$
$$\color{orange}
L_{adv}^{disc} = \mathbb{E}_{y_-} [\log D(y_-, \bar{H}(y_-))] + \mathbb{E}_{x_-} [\log(1 - D(\hat{y}^x_-, \bar{H}(x_-))) + \log(1 - D(x_-, \bar{H}(x_-)))]
$$

### 3. Optimization

The optimization details are provided in the "Implementation Details" section:
- **Optimizer**: The parameters are updated using the **Adam optimizer**, as proposed in "Adam: a method for stochastic optimization."
- **Learning Rate**: The initial learning rate is set to $2e^{-4}$.
- **Schedule**: The learning rate is modified using a **cosine decay schedule**.
- **Iterations and Batch Size**: The model is trained for a total of **300k iterations**, with each batch containing a **single image**.
- **Hyperparameters**: The weighting factors are set as $\lambda = 0.5$, $\mu = 0.99$, $\eta = 10$, and $\zeta = 10$. The dictionary $P$ contains $K = 1024$ elements.

### 4. Special Training Techniques

RetouchFormer employs several special techniques to enhance performance:

- **Semi-supervised Learning**: The model is designed to reduce dependence on expensive paired data. The authors define a proportion $\gamma = \|X_a\| / \|X\|$ and demonstrate that the model remains stable even with very low $\gamma$ (e.g., 0.01).
- **Exponential Moving Average (EMA)**: To produce more reliable imperfection masks for unpaired images, the parameters of the stream $\bar{H}$ are updated as a moving average of $H$:
$$\color{orange}
\theta_{\bar{H}} \leftarrow \mu\theta_{\bar{H}} + (1 - \mu)\theta_H
$$
This technique is based on "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results."
- **Vector Quantization (VQ)**: Used to learn the clean face prior dictionary $P$, allowing the model to "know" what healthy skin looks like before attempting to retouch. This is based on "Neural discrete representation learning."
- **Multi-task Learning**: The framework jointly performs imperfection prediction and local content synthesis, where the two tasks inform each other through the selective self-attention mechanism.

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," Advances in Neural Information Processing Systems, 2017.
- D. P. Kingma and J. L. Ba, "Adam: a method for stochastic optimization," International Conference on Learning Representations, 2015.
- I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Neural Information Processing Systems, 2014.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.


## Inference and Application
>[!question]
>After reviewing the “Methodology” and, if applicable, the “Inference” or “Application” sections of the attached file, please provide a detailed answer to the following:
>
>1. Inference Process:
>    - How is the inference stage of the entire system or framework carried out?
>    - Describe the inputs, outputs, and data flow step by step, including any relevant equations or figures.
>
>2. Use Case:
>    - If the paper proposes real-world application scenarios (use cases), describe all such examples in full detail.
>
>3. Practical Advantages:
>    - Summarize the practical advantages (e.g., real-time processing, memory efficiency, scalability) emphasized by the authors.
>
>Make your 'ANSWER' section as lengthy and detailed as needed. If relevant equations or figures exist, incorporate them into your explanation. Cite references by **title** only if needed, and list them in the 'SOURCES' section in unnumbered IEEE format (with titles, authors, publication years, and publication venues).

>[!answer]


The inference process of RetouchFormer is designed to be a streamlined, end-to-end operation that transforms a face image with imperfections into a high-quality, retouched version. By leveraging learned clean face priors and a selective self-attention mechanism, the model can adaptively identify and fill blemish regions with realistic skin textures.

### 1. Inference Process

During the inference stage, the model follows a feed-forward path through the encoder-transformer-decoder architecture. The process can be broken down into the following steps:

**Step 1: Multi-scale Feature Extraction**
The input to the system is a raw face image $x_-$ that may contain various imperfections such as acne, spots, or dark circles. This image is passed through the encoder $E$, which extracts multi-scale features $f_E(x_-)$. As noted in the "Introduction," these multi-scale tokens provide the flexibility needed to handle imperfections of various sizes.

**Step 2: Imperfection Localization via Prior-based Reconstruction**
To identify where the imperfections are, the model utilizes the learned feature dictionary $P$. The encoder features are quantized to find the nearest "clean" elements in the dictionary:
$$\color{orange}
\rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(x_-)|(i,j) - \rho_k\|_2^2
$$
The reconstruction decoder $R$ then uses these quantized features to generate a reconstructed "clean" image $\tilde{x}_-$. The raw image $x_-$ and the reconstruction $\tilde{x}_-$ are concatenated and passed through the soft mask generation network $\bar{H}$ to produce a weighting map $f_H(x_-)$, which indicates the location and intensity of imperfections. This workflow is visually represented in "Figure 2: The workflow of RetouchFormer."

**Step 3: Selective Feature Transformation**
The multi-scale features $f_E(x_-)$ and the weighting map $f_H(x_-)$ are fed into the Transformer $T$. Inside the Selective Self-Attention Blocks (SSAB), the features are transformed. The queries $Q$ are focused on the imperfection regions, while the keys $K$ and values $V$ are derived from the normal skin regions:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
The attention mechanism calculates a residual feature map $\Delta f_E$ that contains synthesized "normal" skin information for the blemish areas. The features are then updated as follows:
$$\color{orange}
\hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
$$
This equation, found in the "Selective Self-attention" section, shows how the model suppresses imperfection information and replaces it with contextually relevant skin textures.

**Step 4: Final Image Synthesis**
The transformed features $\hat{f}_E(x_-)$ are passed through the synthesis decoder $G$ to produce the final retouched output image $\hat{y}^x_-$. The decoder ensures that the synthesized details are integrated seamlessly with the preserved facial structures.

### 2. Use Case

The paper identifies several real-world application scenarios for RetouchFormer, primarily centered on automated professional-grade portrait editing:

- **Blemish Removal**: The primary use case is the removal of diverse face imperfections, including "moderate-to-severe acne regions," "pockmarks," and "spots with small scales," as mentioned in the "Introduction" and "Qualitative Results on FR-wild."
- **Skin Smoothing and Beautification**: Beyond just removing spots, the model is used for "erasing dark circles and smoothing skin" to create a "flawless skin tone," which typically takes professional retouchers hours to achieve manually.
- **Handling Diverse Real-World Data**: The model is tested on the "FR-wild" dataset, which contains "1,000 in-the-wild face images with different types of facial blemishes," demonstrating its utility for photos taken in uncontrolled environments.
- **High-Resolution Retouching**: The framework is designed to synthesize "clean face images with high fidelity," making it suitable for high-quality photography applications.

### 3. Practical Advantages

The authors emphasize several practical advantages of RetouchFormer over existing state-of-the-art methods:

- **Efficiency in Training and Convergence**: As shown in "Figure 3: Convergence properties of RetouchFormer," the model matches the best results of the base model up to **6 times faster** and achieves a higher overall PSNR.
- **Reduced Data Requirements**: By formulating the task as a semi-supervised "soft inpainting" problem, RetouchFormer "significantly reduces the requirement of training data collection" for paired images, which are expensive and time-consuming to obtain. This is highlighted in the "Overview" of the "Proposed Approach."
- **Structural Simplicity**: Compared to two-stage methods like "Blemish-Aware and Progressive Face Retouching With Limited Paired Data" (BPFRe), RetouchFormer "adopts a simpler structure and achieves better retouching performance," as noted in the "Vision Transformer" section.
- **Scale Flexibility**: The use of multi-scale feature tokens allows the model to be "increased flexibility in dealing with the imperfections at various scales," ensuring that both tiny freckles and large acne patches are handled effectively.
- **High Fidelity and Realism**: The model outperforms advanced methods like "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo" and "RestoreFormer: High-quality blind face restoration from degraded key-value pairs" in terms of PSNR, SSIM, and LPIPS, particularly in removing blemishes while preserving essential facial structures.

**SOURCES**

- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Neural Information Processing Systems, 2017.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from degraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: hierarchical vision transformer using shifted windows," International Conference on Computer Vision, 2021.


## Method Summary
>[!question]
>Based on all the information presented so far, please provide a comprehensive summary of the entire methodology, covering:
>
>1. Framework Structure
>2. Data Flow
>3. Training Strategies
>4. Loss Functions
>5. Inference Procedures
>6. Potential Applications (Usability)
>
>If there is any additional methodology-related content not previously explained, please include it as well. In your 'ANSWER' section, explain the relevant key formulas and cite references by **title** only (not by authors’ names or numbers). Make your summary as long and systematic as necessary. Finally, list all references relevant to your summary in the 'SOURCES' section using unnumbered IEEE format that includes the titles, authors, publication years, and publication venues.

>[!answer]


RetouchFormer is a transformer-based framework designed for high-quality face retouching, specifically targeting the removal of diverse facial imperfections while preserving essential facial structures. The methodology treats face retouching as a generalized 'soft' version of face image inpainting, where imperfections are identified and then replaced with realistic skin textures synthesized from surrounding normal skin.

### 1. Framework Structure

The architecture of RetouchFormer follows an encoder-transformer-decoder design, as illustrated in "Figure 2: The workflow of RetouchFormer." The system is composed of several specialized modules:

- **Multi-scale Encoder ($E$)**: This module extracts features from the input image at multiple intermediate layers. This multi-scale approach, inspired by the hierarchical representation in "Swin transformer: hierarchical vision transformer using shifted windows," allows the model to handle imperfections of various sizes, from small spots to large acne regions.
- **Feature Dictionary ($P$) and Reconstruction Decoder ($R$)**: To capture "clean face priors," the model learns a latent dictionary using **Vector Quantization (VQ)**, a technique detailed in "Neural discrete representation learning." This dictionary represents high-quality skin textures. The reconstruction decoder $R$ uses these priors to generate a clean version of the face, which serves as a reference for identifying blemishes.
- **Soft Mask Generation Network ($\{H, \bar{H}\}$)**: This component identifies imperfection regions. It uses a two-stream U-Net structure. To ensure stable predictions in a semi-supervised setting, the parameters of the stream $\bar{H}$ are updated using the **Exponential Moving Average (EMA)** of $H$, following the strategy in "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results."
- **Transformer ($T$) with Selective Self-Attention Blocks (SSAB)**: The core of the model, which performs the actual retouching in the feature space. It uses a novel selective self-attention mechanism to suppress blemish features and replace them with normal skin features.
- **Synthesis Decoder ($G$) and Discriminator ($D$)**: The decoder $G$ converts the transformed features back into a high-fidelity image. The discriminator $D$, based on "Generative adversarial nets," ensures the realism of the output by performing pixel-level real-fake identification, with the predicted mask injected to focus the discriminator's attention on the retouched areas.

### 2. Data Flow

The data flow through the RetouchFormer is a systematic process of localization and synthesis:

1.  **Feature Extraction**: The raw input image $x_-$ is processed by the encoder $E$ to produce multi-scale features $f_E(x_-)$.
2.  **Prior-based Reconstruction**: The features are quantized using the dictionary $P$. The nearest dictionary element $\rho_*$ is selected for each feature vector:
    $$\color{orange}
    \rho_*|(i,j) = \arg \min_{\rho_k \in P} \|E(y_-)|(i,j) - \rho_k\|_2^2
    $$
    The decoder $R$ then reconstructs a "clean" image $\tilde{x}_-$ from these quantized features.
3.  **Mask Generation**: The raw image $x_-$ and the reconstruction $\tilde{x}_-$ are concatenated and fed into the mask network to produce a weighting map $f_H(x_-)$, indicating the location of imperfections.
4.  **Selective Self-Attention**: In the transformer, the features $f_E(x_-)$ and the mask $f_H(x_-)$ are used to compute Queries ($Q$), Keys ($K$), and Values ($V$):
    $$\color{orange}
    Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
    $$
    $$\color{orange}
    K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
    $$
    $$\color{orange}
    V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
    $$
    This ensures that queries represent imperfections and keys/values represent normal skin. The attended features $\Delta f_E$ are then used to update the feature map:
    $$\color{orange}
    \hat{f}_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
    $$
5.  **Synthesis**: The transformed features $\hat{f}_E(x_-)$ are passed through the decoder $G$ to generate the final retouched image $\hat{y}^x_-$.

### 3. Training Strategies

RetouchFormer is trained using a **semi-supervised approach** to reduce the reliance on expensive paired data. The training set $X$ consists of a small subset of paired images $X_a$ and a larger subset of unpaired images $X_u$. 

The strategy involves:
- **Joint Optimization**: Simultaneously optimizing the dictionary for priors, the U-Net for localization, and the transformer-decoder for synthesis.
- **EMA for Stability**: Using EMA to update the mask generation stream $\bar{H}$ provides more reliable targets for the unpaired data $X_u$.
- **Adversarial Training**: Utilizing a discriminator to ensure that the synthesized skin textures match the statistical distribution of real, clean skin.

### 4. Loss Functions

The model is optimized using a combination of several loss functions:

- **Prior Loss ($L_{prior}$)**: Optimizes the dictionary $P$ to represent clean skin features.
- **Reconstruction Consistency Loss ($L_{con}^{rec}$)**: Ensures the reconstruction decoder $R$ can produce high-quality images using $L_1$ and perceptual losses (VGG-19).
- **Mask Losses ($L_{\bar{H}}$ and $L_H$)**: $L_{\bar{H}}$ provides direct supervision for paired data, while $L_H$ is a confidence-based loss for both paired and unpaired data.
- **Retouching Consistency Loss ($L_{con}^{x2y}$)**: Measures the similarity between the synthesized output and the ground truth for paired data.
- **Adversarial Losses ($L_{adv}^{syn}$ and $L_{adv}^{disc}$)**: Enforce realism in the synthesized regions.

### 5. Inference Procedures

During inference, the process is a straightforward feed-forward pass:
1.  The input image is encoded into multi-scale features.
2.  The clean face prior is used to generate a reconstruction, which helps the mask network locate imperfections.
3.  The selective self-attention mechanism in the transformer replaces the features of the identified imperfections with features aggregated from the surrounding normal skin.
4.  The decoder synthesizes the final high-fidelity, retouched image.

### 6. Potential Applications (Usability)

RetouchFormer is highly usable in professional and consumer-level portrait editing:
- **Automated Blemish Removal**: Efficiently removes acne, pockmarks, and spots, which are diverse and challenging to identify manually.
- **Skin Smoothing**: Erases dark circles and smooths skin texture while preserving the underlying facial structure.
- **High-Fidelity Portrait Editing**: Synthesizes realistic details for moderate-to-severe skin conditions, outperforming generic image enhancement methods like those discussed in "MPRNet: Multi-stage progressive image restoration."
- **Resource-Efficient Retouching**: Due to its semi-supervised nature and fast convergence (as shown in "Figure 3"), it is a practical solution for scenarios where large-scale paired datasets are unavailable.

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
>After carefully reviewing the “Experiments” (or “Results”) and “Reference” sections of the attached file, please provide a detailed answer addressing the following:
>
>1. Dataset Description:
>    - Describe **all** the datasets used in the experiments (type or labels, size, and any notable characteristics). Please provide the relevant reference about dataset by **title** not by author name or number.
>    - If the paper mentions how the datasets are split (train/validation/test), please include details.
>
>2. Data Collection Process:
>    - If the authors collected any data themselves, explain the collection process and the dataset’s properties (type, labels, size, etc.).
>
>3. Role of Datasets:
>    - Clarify how each dataset is utilized within the experimental setup (e.g., for training, evaluation, or application).
>
>Please include references by **title** only (not by author name or number) in each relevant section of your 'ANSWER.' Make your explanation as long as necessary, and list all references used in the 'SOURCES' section using unnumbered IEEE format (titles, authors, publication years, and publication venues).

>[!answer]


### Dataset Description

The experiments conducted in the provided paper utilize several key datasets to train and evaluate the **RetouchFormer** model. These datasets vary in their characteristics, size, and levels of supervision.

1.  **FFHQR Dataset**  
    The primary dataset used for face retouching experiments is **FFHQR**, as introduced in "AutoRetouch: automatic professional face retouching." This dataset consists of 70,000 pairs of raw and professionally retouched face images. The raw data for this dataset is sourced from the **Flickr-Face-HQ (FFHQ)** dataset, which was originally proposed in "A style-based generator architecture for generative adversarial networks."
    - **Size and Splits**: Following the settings established in "AutoRetouch: automatic professional face retouching," the dataset is split into:
        - **Training set**: 56,000 image pairs.
        - **Validation set**: 7,000 image pairs.
        - **Test set**: 7,000 image pairs.
    - **Characteristics**: It provides high-quality pairwise data, allowing for the evaluation of retouching performance against a ground truth.

2.  **FR-wild Dataset**  
    To evaluate the model's generalization capability on real-world data, the authors use the **FR-wild** dataset.
    - **Size**: It contains 1,000 "in-the-wild" face images.
    - **Characteristics**: These images feature diverse types of facial blemishes encountered in real-world scenarios. Notably, this dataset does not have retouching ground truth, making it suitable for qualitative evaluation rather than quantitative benchmarking.

### Data Collection and Preparation Process

The authors did not collect the raw images themselves; instead, they performed a specific sampling and preparation process to facilitate a semi-supervised learning environment.

#### Semi-supervised Data Preparation
To reduce the dependence on expensive paired retouching data, the authors define a set of training data $\color{orange}X = X_a \cup X_u$ , where:
- $\color{orange}X_a = \{(x_a, y_a)\}$  represents a small subset of paired raw-retouching images.
- $\color{orange}X_u = \{x_u, y_u\}$  represents a larger subset of unpaired images, where the raw image $\color{orange}x_u$  and the retouching image $\color{orange}y_u$  may correspond to different identities.

The proportion of paired training data is defined by the factor $\color{orange}\gamma$ :
$$\color{orange}
\gamma = \frac{\|X_a\|}{\|X\|}
$$
In their experiments, the authors randomly sample paired images from the **FFHQR** dataset to create subsets with $\color{orange}\gamma$  values ranging across $\color{orange}\{0.01, 0.1, 0.2, 1\}$ . This process allows them to demonstrate the stability and effectiveness of **RetouchFormer** even when paired data is extremely limited (e.g., only 1% of the total data).

### Role of Datasets

Each dataset serves a distinct purpose within the experimental framework of the paper:

- **FFHQR (Training and Quantitative Evaluation)**:  
  This dataset is used to train the encoder-transformer-decoder architecture and the discriminator. Because it contains ground truth retouched images, it is the basis for quantitative comparisons using metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS), as described in "The unreasonable effectiveness of deep features as a perceptual metric."

- **FR-wild (Qualitative Evaluation and Generalization)**:  
  Since ground truth is unavailable for this dataset, it is used to visually demonstrate the model's superiority in handling moderate-to-severe acne, pockmarks, and dark circles in unconstrained, real-world environments. It serves to prove that the model, though trained on **FFHQR**, generalizes well to unseen real-world blemishes.

- **FFHQ (Source Material)**:  
  While not used directly as a separate experimental set, it provides the underlying raw facial data that constitutes the "raw" side of the **FFHQR** pairs.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and R. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.


## Implementation Details
>[!question]
>After carefully reviewing the “Experiments” (or “Results”) sections of the attached file, please provide a detailed answer addressing the following:
>
>1. Implementation Details:
>    - Describe all experimental settings and hyperparameters (learning rate, batch size, optimizer, number of epochs, etc.) in depth.
>
>2. GPU Information:
>    - Specify the type and number of GPUs used for training.
>    - If available, include the approximate training time.
>
>3. Reproducibility:
>    - If the paper provides guidance for reproducibility (open-sourced code or models, etc.), summarize the instructions or details given.
>
>Please make your 'ANSWER' as long and detailed as necessary.

>[!answer]


### 1. Implementation Details

The implementation of **RetouchFormer** involves specific configurations for image processing, network architecture, and the optimization process to ensure high-quality face retouching. According to the "Implementation Details" section on page 5, the following settings were employed:

**Image Pre-processing and Architecture Settings**
- **Image Resolution**: For a fair comparison with competing methods, all training and testing images are resized to a resolution of $512 \times 512$ pixels, unless specified otherwise.
- **Feature Dictionary**: The feature dictionary $\mathcal{P}$, which is used to capture clean face priors, consists of $K = 1024$ elements.
- **Token Representation**: As described in the "Selective Self-attention" section on page 4, spatial patches are extracted from multiple intermediate layers of the encoder $E$ and reshaped into 512-dimensional vectors to serve as tokens for the transformer.

**Hyperparameters and Weighting Factors**
The model utilizes several weighting factors to balance the different components of its multi-task loss function. As specified in the "Implementation Details" and "Model Training" sections:
- **$\lambda = 0.5$**: This factor, used in Equation (3), controls the impact of unpaired training data within the imperfection prediction loss ($L_H$).
- **$\mu = 0.99$**: This is the momentum coefficient used in Equation (4) for the Exponential Moving Average (EMA) update of the mask generation network parameters, as referenced in "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results."
- **$\eta = 10$**: This weighting factor is applied in Equation (9) for the consistency loss ($L_{con}^{rec}$), which ensures high-quality reconstruction by combining $L_1$ loss and VGG-19 based perceptual features.
- **$\zeta = 10$**: This factor is used in Equation (10) for the consistency loss on paired data ($L_{con}^{x2y}$), measuring the degree of consistency between the synthesized retouching result and the ground truth.

**Optimization Strategy**
- **Optimizer**: The parameters of the model are updated using the **Adam optimizer**, a popular choice for stochastic optimization in deep learning, as detailed in "Adam: a method for stochastic optimization."
- **Learning Rate**: The initial learning rate is set to $2e^{-4}$.
- **Learning Rate Schedule**: The learning rate is adjusted throughout the training process using a **cosine decay schedule**.
- **Training Duration**: The model is trained for a total of 300,000 iterations.
- **Batch Size**: Each training batch contains a single image.

### 2. GPU Information

Regarding the hardware used for the experiments:
- **GPU Type and Number**: The information regarding the specific type (e.g., NVIDIA RTX 3090, V100) and the number of GPUs used for training **does not exist in the provided file.**
- **Training Time**: While the exact total training time is not explicitly stated, the paper notes in the "Comparison with State-of-the-arts" section on page 5 that **RetouchFormer** is highly efficient. Specifically, it can match the best result of the base model up to **6 times faster** and converges to a higher PSNR, as illustrated in "Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR ($\gamma=0.1$)."

### 3. Reproducibility

Regarding the availability of resources for reproducing the results:
- **Code and Models**: The provided file **does not contain information** regarding the release of open-source code, pre-trained models, or a link to a public repository (such as GitHub). The "Conclusion" and "Acknowledgments" sections do not mention any external links for reproducibility.

**SOURCES**

- D. P. Kingma and J. L. Ba, "Adam: a method for stochastic optimization," in Proc. International Conference on Learning Representation, 2015.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Advances in Neural Information Processing Systems, 2017.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.


## Quantitative Results
>[!question]
>After carefully reviewing the “Experiments” (or “Results”) and “Reference” sections of the attached file, please provide a detailed answer to the following:
>
>1. Evaluation Metrics:
>    - Which metrics (e.g., Accuracy, Precision, Recall, F1-score, IoU) are used? Please provide the explanation of all the metrics used.
>    - Please provide all the relevant equations and the **references** about each metric using paper **title**, and explain how each metric is interpreted.
>
>2. Reference Tables:
>    - Which tables should we look at to see the quantitative results?
>    - What comparisons do these tables show? Don't show the numbers in the table, explain what the table is comparing.
>
>3. Interpretation:
>    - How do the authors interpret these tables?
>    - What do the numbers signify, and what are the strengths and weaknesses of the proposed method according to these results?
>
>Please cite references in your 'ANSWER' by **title** only (not by author name or number). Make your response as long as needed, and list all relevant references in the 'SOURCES' section in unnumbered IEEE format (including titles, authors, publication years, and publication venues).

>[!answer]


### 1. Evaluation Metrics

To evaluate the performance of **RetouchFormer**, the authors utilize four primary metrics. These metrics assess the quality of the synthesized images from both mathematical and perceptual perspectives.

*   **Peak Signal-to-Noise Ratio (PSNR)**: This is a standard metric used to measure the quality of reconstruction in image processing. It calculates the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. In the context of this paper, a higher PSNR indicates that the retouched image is mathematically closer to the ground truth.
*   **Structural Similarity Index Measure (SSIM)**: This metric is used for measuring the similarity between two images. Unlike PSNR, which focuses on absolute errors, SSIM considers changes in structural information, luminance, and contrast, which are more aligned with human visual perception. A higher SSIM value (closer to 1) signifies better preservation of image structures.
*   **Learned Perceptual Image Patch Similarity (LPIPS)**: As introduced in **"The unreasonable effectiveness of deep features as a perceptual metric,"** LPIPS measures the perceptual distance between two images using deep features from a pre-trained network. It is designed to mimic human judgment of image similarity more accurately than traditional metrics. A lower LPIPS score indicates that the synthesized image is perceptually more similar to the ground truth.
*   **Soft-IoU (Intersection over Union)**: This metric is specifically used in the "Analysis of Design Elements" section (Table 2) to evaluate the accuracy of the imperfection prediction (mask generation). It measures the overlap between the predicted imperfection regions and the actual regions. A higher Soft-IoU indicates more precise localization of blemishes.

**Equations and References**:
The mathematical equations for PSNR, SSIM, LPIPS, and Soft-IoU **do not exist in the provided file.** While the paper references **"The unreasonable effectiveness of deep features as a perceptual metric"** for the LPIPS metric, it does not provide the specific formulas for any of these evaluation metrics.

### 2. Reference Tables

The quantitative results and comparisons are presented in three main tables:

*   **Table 1: Quantitative comparison with competing methods on FFHQR**: This table provides a comprehensive comparison between **RetouchFormer** and several state-of-the-art methods, including generic image translation models like Pix2PixHD (referenced as **"Image-to-image translation with conditional adversarial networks"**), restoration models like GPEN (**"GAN prior embedded network for blind face restoration in the wild"**), and specialized retouching models like BPFRe (**"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"**). The comparison is conducted across four different levels of supervision ($\gamma = 0.01, 0.1, 0.2, 1$), showing how each model performs as the amount of paired training data increases.
*   **Table 2: Quantitative results of RetouchFormer with and without the priors $\mathcal{P}$ in soft mask generation**: This table is an ablation study focusing on the "Clean Face Prior." It compares the performance of the full **RetouchFormer** against a variant that does not use the feature dictionary $\mathcal{P}$ for mask generation. It uses PSNR, SSIM, and Soft-IoU to show the impact of the prior on both image quality and blemish localization.
*   **Table 3: Results of RetouchFormer and ablative models on FFHQR ($\gamma=0.1$)**: This table evaluates the importance of specific design elements. It compares the full model against three variants:
    1.  **w/ SPS**: Uses a single patch size instead of multi-scale tokens.
    2.  **w/ GSA**: Uses Global Self-Attention instead of the proposed Selective Self-Attention.
    3.  **w/o $X_u$**: Trained without the use of unpaired training data.

### 3. Interpretation of Results

The authors interpret the quantitative results to highlight the strengths of **RetouchFormer** and the necessity of its core components.

**Significance of the Numbers**:
*   **Superiority over SOTA**: In Table 1, **RetouchFormer** consistently achieves the highest PSNR and SSIM and the lowest LPIPS across all supervision levels. For instance, at $\gamma=0.1$, it outperforms the second-best method (BPFRe) by 0.56 dB in PSNR. The improvement in LPIPS is particularly notable, with a 12 percentage point improvement over BPFRe, indicating significantly higher perceptual realism.
*   **Efficiency**: As shown in **"Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR,"** the model matches the best results of the base model up to 6 times faster, signifying high computational efficiency.
*   **Impact of Design Elements**: Table 3 reveals that removing multi-scale tokens (SPS) leads to a significant performance drop (4.93 dB in PSNR), proving their role in handling blemishes of various scales. Substituting Selective Self-Attention with Global Self-Attention (GSA) results in a 1.06 dB drop in PSNR, confirming that the selective mechanism is crucial for effectively replacing imperfections with normal skin features.

**Strengths and Weaknesses**:
*   **Strengths**:
    *   **High Fidelity and Realism**: The model excels at synthesizing clean face images that are both mathematically accurate and perceptually pleasing.
    *   **Robustness to Limited Data**: The model maintains high performance even with very little paired data ($\gamma=0.01$), thanks to the semi-supervised framework and the use of unpaired data ($X_u$).
    *   **Precise Localization**: The use of clean face priors significantly improves blemish detection, as evidenced by the 17.9% gain in Soft-IoU shown in Table 2.
*   **Weaknesses**:
    The provided file **does not explicitly list specific weaknesses** of the proposed **RetouchFormer** method. However, the authors note that without the selective self-attention mechanism, the model "fails to neutralize pockmarks and erase dark circles," which highlights the complexity of the task and the model's reliance on that specific component for success.

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
>After carefully reviewing the “Experiments” (or “Results”) sections of the attached file, please provide a detailed answer focusing on the following:
>
>1. Qualitative Results:
>    - Which figures (images, graphs, example outputs, etc.) illustrate the qualitative results? (not for ablation study)
>    - List them all and explain what each figure represents in detail.
>
>2. Interpretation:
>    - How do the authors interpret these figures?
>    - What conclusions and insights do they draw, and what strengths or weaknesses of the proposed method are highlighted?
>
>3. Failure Case:
>    - If the paper includes any failure or edge cases, describe those examples and discuss potential causes.
>
>Make your 'ANSWER' as lengthy as needed to be comprehensive.

>[!answer]


### 1. Qualitative Results

The qualitative performance of **RetouchFormer** is illustrated through several figures that showcase the model's ability to identify and remove facial imperfections while maintaining high image fidelity. The following figures provide visual evidence of the model's effectiveness:

*   **Figure 1: The main idea behind RetouchFormer** (Page 1)  
    This figure serves as an introductory visual summary of the proposed approach. It displays a **Source image** of a face with visible acne, followed by an **Imperfection prediction** map (a heatmap indicating the location of blemishes). It then illustrates the **Selective self-attention** mechanism, where features from imperfection regions (queries) are replaced by features from normal skin regions (keys and values). Finally, it shows the **Retouching result**, where the acne has been removed and replaced with realistic skin texture.

*   **Figure 4: Visual comparison with competing methods on FR-wild images** (Page 6)  
    This figure provides a side-by-side qualitative comparison between **RetouchFormer** and several state-of-the-art methods, including GPEN (from **"GAN prior embedded network for blind face restoration in the wild"**), AutoRetouch (from **"AutoRetouch: automatic professional face retouching"**), MPRNet (from **"Multi-stage progressive image restoration"**), RestoreFormer (from **"RestoreFormer: High-quality blind face restoration from undegraded key-value pairs"**), ABPN (from **"ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"**), and BPFRe (from **"Blemish-Aware and Progressive Face Retouching With Limited Paired Data"**). The images are sourced from the **FR-wild** dataset, which contains real-world faces with moderate-to-severe acne, pockmarks, and dark circles. Red boxes are used to highlight specific areas where other methods struggle to completely remove blemishes.

*   **Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild** (Page 6)  
    This figure presents three sets of high-resolution before-and-after examples. It specifically demonstrates the model's versatility in handling different types of facial concerns. The examples highlight the model's success in **removing acne**, **erasing dark circles** under the eyes, and **smoothing skin** while preserving the underlying facial structure and essential features.

### 2. Interpretation of Qualitative Results

The authors interpret these visual results to emphasize the superiority of **RetouchFormer** over existing techniques.

*   **Handling Large-Scale Blemishes**: In the discussion of **Figure 4**, the authors observe that while advanced methods like BPFRe and RestoreFormer produce better results than generic models, they still "fail to remove the blemishes at large scales." **RetouchFormer**, however, is described as producing "significantly pleasant" results, successfully clearing even severe imperfections that other models leave behind.
*   **Realism and Fidelity**: The authors conclude that the retouching images synthesized by **RetouchFormer** achieve "high realism and fidelity." This is attributed to the model's ability to "approximately predict imperfections" and "simultaneously synthesize realistic details" by leveraging clean face priors.
*   **Generalization**: A key insight drawn from these figures is the model's strong generalization capability. Although the model is trained on the **FFHQR** dataset, the qualitative results on the **FR-wild** dataset (which the model did not see during training) demonstrate that it can effectively handle diverse, real-world facial blemishes.
*   **Strengths Highlighted**: The primary strengths highlighted are the precise localization of imperfections and the effective use of contextual information from normal skin to fill in those regions. The authors note that the model can "utilize the spatial information of imperfections and synthesize the content from the contextual features of normal skin."

### 3. Failure Case

The information regarding specific failure cases or edge cases for the proposed **RetouchFormer** model **does not exist in the provided file.** The authors focus on the model's successes and its improvements over competing methods. While they mention that certain *ablated* versions of the model (such as the one using Global Self-Attention) fail to neutralize pockmarks, they do not provide examples of the full **RetouchFormer** model failing on specific inputs.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Multi-stage progressive image restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2022.


## Ablation Study
>[!question]
>After carefully reviewing the “Experiments” (or “Results”) sections of the attached file, please provide a detailed answer addressing the following:
>
>1. Ablation Study List:
>    - List all ablation studies performed (e.g., removing or altering modules, loss terms, or algorithms).
>    - Explain the purpose of each study.
>
>2. Reference Tables or Figures:
>    - Which tables or figures present the results of each ablation study?
>    - Please specify their table/figure numbers (e.g., Table X, Figure Y).
>
>3. Interpretation:
>    - According to the authors, which components are most crucial for performance?
>    - How does adding or removing each component affect the results?
>
>Make your 'ANSWER' as long as necessary to be thorough.

>[!answer]


The authors of the paper perform a series of ablation studies to validate the effectiveness of the individual design elements of **RetouchFormer**. These studies are primarily detailed in the section titled **"Analysis of Design Elements"** on page 7.

### 1. Ablation Study List

The authors conducted four distinct ablation studies to evaluate the contribution of the clean face prior, multi-scale tokens, selective self-attention, and unpaired training data.

*   **Clean Face Prior ($\mathcal{P}$)**:
    *   **Purpose**: To verify if learning a feature dictionary $\mathcal{P}$ on retouching images $\{y_a, y_u\}$ actually aids in face imperfection prediction.
    *   **Method**: The authors created a variant called 'w/o $\mathcal{P}$', where the network $H$ (responsible for mask generation) generates an imperfection mask directly from the raw image without the guidance of the reconstructed features from the dictionary.

*   **Multi-scale Tokens**:
    *   **Purpose**: To determine if extracting features at multiple intermediate layers of the encoder (multi-scale) is necessary for handling imperfections of varying sizes.
    *   **Method**: They developed a variant called 'w/ SPS' (Single Patch Size), which uses only the final output of the encoder to form tokens, rather than the multi-scale patch-based representation.

*   **Selective Self-Attention**:
    *   **Purpose**: To evaluate whether the proposed selective self-attention mechanism, which suppresses interactions between imperfection regions and normal skin, is superior to standard attention mechanisms.
    *   **Method**: They built a variant called 'w/ GSA' (Global Self-Attention) by substituting the selective mechanism with standard global self-attention.

*   **Unpaired Training Data ($X_u$)**:
    *   **Purpose**: To analyze the impact of using a large amount of unpaired data in a semi-supervised setting.
    *   **Method**: They tested a variant called 'w/o $X_u$', where the model is trained using only the limited set of paired data $X_a$.

### 2. Reference Tables and Figures

The results of these ablation studies are presented in the following tables and figures:

*   **Clean Face Prior**: Results are shown in **"Table 2: Quantitative results of RetouchFormer with and without the priors $\mathcal{P}$ in soft mask generation"** and visualized in **"Figure 6: The soft masks generated by RetouchFormer with and without the prior $\mathcal{P}$."**
*   **Multi-scale Tokens, Selective Self-Attention, and Unpaired Data**: Quantitative results for these three studies are compiled in **"Table 3: Results of RetouchFormer and ablative models on FFHQR ($\gamma=0.1$)."**
*   **Visual Comparisons**: Qualitative differences for the SPS, GSA, and $X_u$ variants are illustrated in **"Figure 7: Representative retouching images synthesized by RetouchFormer and ablative models on FR-wild images."**

### 3. Interpretation of Components

According to the authors' analysis, each component plays a vital role, but some have a more drastic impact on specific performance metrics.

#### Most Crucial Components
*   **Multi-scale Tokens**: This appears to be the most crucial component for maintaining high **PSNR**. Removing it ('w/ SPS') resulted in the largest performance drop of **4.93 dB** in PSNR. The authors state in **"Analysis of Design Elements"** that multi-scale tokens are essential for handling "imperfections with different scales."
*   **Unpaired Training Data ($X_u$)**: This is critical for perceptual quality and model stability. The variant 'w/o $X_u$' underperformed the full model by approximately **74%** in terms of the **LPIPS** metric. The authors conclude that $X_u$ is "important for stabilizing the performance of RetouchFormer."
*   **Selective Self-Attention**: This is crucial for the actual removal of specific blemishes. The 'w/ GSA' variant showed a **1.06 dB** drop in PSNR and, as seen in **"Figure 7: Representative retouching images synthesized by RetouchFormer and ablative models on FR-wild images,"** it "fails to neutralize pockmarks and erase dark circles under the eyes."

#### Effects of Adding/Removing Components
*   **Adding the Clean Face Prior ($\mathcal{P}$)**: As shown in **"Table 2,"** the inclusion of the prior leads to a **17.9% gain** in **Soft-IoU** for imperfection prediction. Visually, **"Figure 6"** demonstrates that the prior helps the model discover "more imperfections" compared to the 'w/o $\mathcal{P}$' version.
*   **Removing Selective Self-Attention**: Without explicit imperfection prediction and suppression, the model cannot effectively replace blemish features with normal skin features. This leads to significantly lower fidelity in the final retouched image.
*   **Removing Multi-scale Tokens**: The model loses its flexibility. It becomes less capable of addressing both fine-grained details (like small spots) and larger-scale blemishes (like severe acne) simultaneously.
*   **Removing Unpaired Data**: The task of face retouching becomes "highly challenging" because the model lacks the regularization provided by the larger, unpaired dataset, leading to a significant loss in perceptual similarity to clean faces.

**SOURCES**

- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," in Proc. IEEE Winter Conference on Applications of Computer Vision, 2021.
- L. Xie, W. Xue, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and R. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- A. Tarvainen and H. Valpola, "Mean teachers are better role models: weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Advances in Neural Information Processing Systems, 2017.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," in Proc. Neural Information Processing Systems, 2017.
- T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2019.


## Results Summary
>[!question]
>Please provide a comprehensive and systematic summary of the quantitative/qualitative results and the ablation studies discussed so far, addressing the following:
>
>1. Summary of Quantitative and Qualitative Results
>2. Ablation Study Overview
>3. Strengths and Weaknesses of the Methodology
>
>If there are any additional experimental details or results not yet covered, please include them as well. In your 'ANSWER,' incorporate relevant key formulas and cite references by **title** only (not by authors or numbers). Make your summary as lengthy as needed. Finally, list all references used in the 'SOURCES' section in unnumbered IEEE format (titles, authors, publication years, and publication venues).

>[!answer]


The **RetouchFormer** is a transformer-based framework designed for high-quality face retouching, treating the task as a generalized "soft" version of face image inpainting. It simultaneously identifies imperfections and synthesizes realistic skin content. Below is a comprehensive summary of the quantitative and qualitative results, the ablation studies, and the methodology's strengths and weaknesses based on the provided research.

### 1. Summary of Quantitative and Qualitative Results

**Quantitative Performance**
The quantitative evaluation of **RetouchFormer** is primarily conducted on the **FFHQR** dataset, as described in "AutoRetouch: automatic professional face retouching." The model is compared against several state-of-the-art (SOTA) methods across different levels of supervision ($\gamma$), where $\gamma$ represents the proportion of paired training data.

*   **Superiority in Metrics**: As shown in **"Table 1: Quantitative comparison with competing methods on FFHQR,"** **RetouchFormer** consistently outperforms all competing methods, including BPFRe (from "Blemish-Aware and Progressive Face Retouching With Limited Paired Data"), ABPN (from "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo"), and RestoreFormer (from "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs").
*   **Key Gains**: At a supervision level of $\gamma=0.1$, **RetouchFormer** achieves a PSNR of 45.13 dB, which is 0.56 dB higher than the second-best method, BPFRe. More significantly, it achieves an LPIPS score of 0.0093, representing a 12 percentage point improvement over BPFRe, indicating much higher perceptual fidelity.
*   **Efficiency**: The model is highly efficient. According to **"Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR,"** **RetouchFormer** converges to a higher PSNR and matches the best results of the base model up to 6 times faster.

**Qualitative Performance**
The qualitative results demonstrate the model's ability to handle diverse and severe real-world blemishes.
*   **Real-World Generalization**: Using the **FR-wild** dataset, **"Figure 4: Visual comparison with competing methods on FR-wild images"** shows that while other methods often leave traces of large-scale acne or pockmarks, **RetouchFormer** successfully removes them, replacing the regions with realistic skin texture.
*   **Versatility**: **"Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild"** illustrates the model's success in three distinct areas: removing inflammatory acne, erasing dark circles under the eyes, and smoothing overall skin texture while preserving essential facial features.

### 2. Ablation Study Overview

The authors performed systematic ablation studies to validate the three core design elements of **RetouchFormer**.

**A. Clean Face Prior ($\mathcal{P}$)**
The model learns a feature dictionary $\mathcal{P}$ to capture clean face priors via Vector Quantization (VQ), as proposed in "Neural discrete representation learning." The nearest element $\rho_*$ in the dictionary is found by:
$$\color{orange}
\rho_*|(i,j) = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|(i,j) - \rho_k\|_2^2
$$
As shown in **"Table 2,"** using this prior leads to a **17.9% gain** in Soft-IoU for imperfection prediction. **"Figure 6"** confirms that the prior helps the model identify significantly more blemish regions than a model without it.

**B. Multi-scale Tokens**
Instead of a fixed scale, the model extracts patches from multiple intermediate layers of the encoder $E$. **"Table 3"** shows that using a Single Patch Size (SPS) leads to a massive performance drop of **4.93 dB** in PSNR, proving that multi-scale features are essential for addressing imperfections of various sizes.

**C. Selective Self-Attention**
This is the most innovative component, where the model suppresses interactions between imperfections and normal skin. The queries ($Q$), keys ($K$), and values ($V$) are formulated using the predicted imperfection mask $f_H(x_-)$:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v
$$
The final feature update is:
$$\color{orange}
f_E(x_-) = f_E(x_-) \otimes (1 - f_H(x_-)) + \Delta f_E \otimes f_H(x_-)
$$
According to **"Table 3,"** replacing this with Global Self-Attention (GSA) results in a **1.06 dB** PSNR loss and a failure to erase pockmarks and dark circles (as seen in **"Figure 7"**).

### 3. Strengths and Weaknesses of the Methodology

**Strengths**
*   **Semi-supervised Stability**: The methodology effectively utilizes unpaired data ($X_u$) to stabilize training. Removing $X_u$ results in a 74% degradation in LPIPS performance (**"Table 3"**).
*   **High Fidelity**: By formulating retouching as "soft inpainting," the model ensures that only the blemish regions are modified while the rest of the face remains untouched, preserving identity and essential details.
*   **Computational Efficiency**: The transformer-based design with selective attention is faster and more effective than global convolution-based methods.

**Weaknesses**
The provided file **does not explicitly list weaknesses** of the **RetouchFormer**. However, based on the methodology, one can infer:
*   **Dependency on Mask Accuracy**: The success of the selective self-attention is heavily dependent on the accuracy of the imperfection prediction mask $f_H$. If the mask fails to identify a blemish, the selective attention will not trigger for that region.
*   **Complexity of the Two-Stream Mask Generation**: The use of a U-Net with a two-stream structure and Exponential Moving Average (EMA) updates adds architectural complexity compared to simpler end-to-end models.

### 4. Additional Experimental Details

**Loss Functions**
The model is optimized using a combination of four loss terms:
1.  **Prior Loss ($L_{prior}$)**: Optimizes the encoder and dictionary to reconstruct clean face features.
2.  **Imperfection Prediction Loss ($L_H$ and $L_H$)**: Enforces accurate mask generation using both paired and unpaired data.
3.  **Consistency Loss ($L_{con}^{rec}$ and $L_{con}^{x2y}$)**: Ensures the synthesized images match the ground truth and maintain high-quality details using VGG-19 features.
4.  **Adversarial Loss ($L_{adv}^{syn}$ and $L_{adv}^{disc}$)**: Uses a discriminator $D$ to ensure the synthesized images match the statistics of real clean faces.

**Training Setup**
The model is trained for **300,000 iterations** using the **Adam optimizer** with an initial learning rate of **$2e^{-4}$** and a **cosine decay schedule**. Each batch contains a single image of $512 \times 512$ resolution.

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
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Multi-stage progressive image restoration," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," in Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2021.

# Conclusion
## Limitations ans Future works
>[!question]
>After carefully reviewing the “Conclusion” (or “Limitation”/“Future Works”) sections of the attached file, please provide a detailed answer focusing on the following:
>
>1. Limitations:
>    - What unresolved problems or generalization issues does the paper mention?
>    - Are there any constraints or shortcomings explicitly stated?
>
>2. Future Works:
>    - Which research directions or potential extensions do the authors propose (e.g., applying to larger datasets, expanding the model architecture)?
>    - If the paper does not explicitly mention certain points, please note that they are not stated.
>
>Please make your 'ANSWER' as lengthy as needed to cover all details.

>[!answer]


### Overview of RetouchFormer

RetouchFormer is a transformer-based framework designed for high-quality face retouching. It addresses the limitations of previous methods, such as the lack of imperfection localization and the heavy reliance on large-scale pairwise training data. The model treats face retouching as a "soft inpainting" task, where it simultaneously identifies imperfections and synthesizes realistic skin textures in those regions. This is achieved through a combination of clean face priors, a reconstruction-oriented localization module, and a selective self-attention mechanism.

As described in "RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention," the architecture follows an encoder-transformer-decoder design. The encoder extracts multi-scale features, the transformer performs selective self-attention to replace imperfection features with normal skin features, and the decoder synthesizes the final retouched image.

---

### 1. Limitations

Based on the provided file, the authors do not include a dedicated "Limitations" section, and they primarily focus on the model's superior performance compared to existing state-of-the-art methods. However, the following points regarding constraints and challenges can be inferred from the text:

*   **Unresolved Problems and Generalization Issues**: The information regarding specific unresolved problems or generalization issues of the RetouchFormer model itself **does not exist in the provided file**. On the contrary, the authors highlight the model's strong generalization capabilities on real-world data, such as the FR-wild dataset, as shown in "Figure 4: Visual comparison with competing methods on FR-wild images" and "Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild."
*   **Explicit Constraints**:
    *   **Input Resolution**: In the "Implementation Details" section on page 5, the authors state that for fair comparison, both training and testing images are resized to $\color{orange}512 \times 512$  pixels unless noted otherwise. This indicates a fixed spatial constraint for the model's standard operation.
    *   **Data Dependency**: While the model is semi-supervised and reduces the need for paired data, the authors note in the "Analysis of Design Elements" (page 7) that "with a decrease in the amount of paired data, the task of face retouching becomes highly challenging." They emphasize that unpaired data ($\color{orange}X_u$ ) is crucial for stabilizing performance, implying that the model still requires a balanced dataset to maintain high fidelity.
*   **Discriminator Sensitivity**: The authors mention in the "Model Training" section (page 5) that "imperfection regions may be very small, and generic image-level discriminators tend to be deceived in this case." To mitigate this, they had to design a specific discriminator that performs pixel-level real-fake identification by injecting mask features into multiple layers.

---

### 2. Future Works

Regarding potential extensions or future research directions:

*   **Proposed Research Directions**: The information regarding specific future research directions or potential extensions (such as applying the model to larger datasets or expanding the architecture) **does not exist in the provided file**.
*   **Conclusion Summary**: The "Conclusion" section on page 7 summarizes the current achievements of the work—formulating retouching as soft inpainting, learning clean face priors, and utilizing selective self-attention—but it does not explicitly state what the authors intend to investigate next.

---

### Technical Details of the RetouchFormer Mechanism

To understand why the authors may not have focused on limitations, it is helpful to review the core innovations that they claim solve previous issues:

#### Clean Face Priors
The model learns a feature dictionary $\color{orange}\mathcal{P}$  to capture clean face priors using Vector Quantization (VQ). The feature vector at a location $\color{orange}(i, j)$  is approximated by the nearest element in the dictionary:
$$\color{orange}
\rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2
$$
where $\color{orange}E(y_-)$  is the encoder feature of a retouching image. This prior is used to help the network identify what "clean" skin should look like, aiding in the localization of imperfections (as discussed in "RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention").

#### Selective Self-Attention
The selective self-attention mechanism is the key to the "soft inpainting" approach. It suppresses interactions between imperfection regions and instead encourages the model to fill those regions using features from normal skin. The queries ($\color{orange}Q$ ), keys ($\color{orange}K$ ), and values ($\color{orange}V$ ) are formulated as:
$$\color{orange}
Q = W_q(f_E(x_-) \otimes f_H(x_-)) + b_q,
$$
$$\color{orange}
K = W_k(f_E(x_-) \otimes (1 - f_H(x_-))) + b_k,
$$
$$\color{orange}
V = W_v(f_E(x_-) \otimes (1 - f_H(x_-))) + b_v,
$$
where $\color{orange}f_H(x_-)$  represents the predicted imperfection mask. This ensures that the tokens associated with imperfections serve as queries, while the keys and values are drawn from normal skin regions (as detailed in the "Selective Self-attention" section on page 4).

**SOURCES**

- L. Xie, W. Xue, W. Xu, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Proceedings of Neural Information Processing Systems, 2017.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.


## Conclusion
>[!question]
>After carefully reviewing the “Conclusion” section of the attached file, please provide a thorough summary that addresses:
>
>1. Main Claims:
>    - What are the key arguments or findings the authors emphasize?
>    - How do they position their contribution in the field?
>
>2. Supporting Evidence:
>    - Which methodological strengths or experimental results do the authors cite to justify their claims?
>
>Please make your 'ANSWER' as long as necessary to be comprehensive.

>[!answer]


The "Conclusion" section of the paper, titled "RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention," summarizes the research by emphasizing the novel formulation of face retouching and the architectural innovations that allow the model to outperform existing methods.

### 1. Main Claims

The authors emphasize several key arguments and findings that define the contribution of RetouchFormer to the field of image processing and computer vision:

*   **Formulation as 'Soft Inpainting'**: The primary claim is the treatment of face retouching as a generalized "soft" version of face image inpainting. Unlike traditional methods that apply global transformations, RetouchFormer is designed to "jointly detect and complete imperfection regions," as stated in the "Proposed Approach" section and reiterated in the "Conclusion."
*   **Clean Face Priors for Localization**: The authors argue that learning "clean face priors" in the form of a feature dictionary is essential for accurate imperfection prediction. By measuring the difference between the original features and those reconstructed from the dictionary, the model can approximately predict where imperfections are located. This is mathematically represented by the nearest-neighbor search in the dictionary $\color{orange}\mathcal{P}$ :
    $$\color{orange}
    \rho_*|_{(i,j)} = \arg \min_{\rho_k \in \mathcal{P}} \|E(y_-)|_{(i,j)} - \rho_k\|_2^2,
    $$
    where $\color{orange}E(y_-)$  represents the encoder features of the input image (as shown in "RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention").
*   **Selective Self-Attention Mechanism**: A major claim is the effectiveness of the "prior-based selective self-attention" mechanism. This mechanism is designed to "suppress face imperfections" by limiting spatial interactions. Instead of global attention, it ensures that the features in imperfection regions are synthesized using information from "normal skin" regions.
*   **Semi-supervised Stability**: The authors claim that their approach effectively utilizes "unpaired training data" to stabilize performance. This reduces the heavy reliance on perfectly paired raw-and-retouched datasets, which are often expensive and difficult to collect in real-world scenarios.

### 2. Supporting Evidence

To justify these claims, the authors provide a combination of methodological strengths and experimental results:

*   **Quantitative Superiority on FFHQR**: The authors cite the results in "Table 1: Quantitative comparison with competing methods on FFHQR" to prove that RetouchFormer outperforms advanced methods like BPFRe, ABPN, and RestoreFormer. For instance, at $\color{orange}\gamma=0.1$  (10% paired data), RetouchFormer achieves a PSNR of 45.13 dB and the lowest LPIPS score of 0.0093, indicating higher fidelity and perceptual quality.
*   **Qualitative Realism on Real-World Data**: The "Conclusion" refers to "extensive comparisons" that demonstrate superior capability. This is supported by "Figure 4: Visual comparison with competing methods on FR-wild images" and "Figure 5: Representative high-quality retouching results of RetouchFormer on FR-wild," which show the model successfully removing moderate-to-severe acne, pockmarks, and dark circles while preserving facial structures.
*   **Ablation Studies (Design Validation)**:
    *   **Effect of Priors**: In "Table 2: Quantitative results of RetouchFormer with and without the priors P in soft mask generation," the authors show that including the dictionary-based priors leads to a 17.9% gain in Soft-IoU for imperfection localization.
    *   **Selective vs. Global Attention**: The authors justify the selective attention mechanism by comparing it to Global Self-Attention (GSA). As shown in "Table 3: Results of RetouchFormer and ablative models on FFHQR," substituting selective attention with GSA leads to a significant drop in PSNR (from 45.13 dB to 44.07 dB) and a failure to erase specific blemishes like dark circles (as seen in "Figure 7: Representative retouching images synthesized by RetouchFormer and ablative models on FR-wild images").
    *   **Multi-scale Tokens**: The use of multi-scale features is justified by the "w/ SPS" (Single Patch Size) variant in Table 3, which resulted in a performance drop of 4.93 dB, proving that multi-scale tokens are necessary to handle imperfections of various sizes.
*   **Efficiency and Convergence**: The authors highlight the model's efficiency in "Figure 3: Convergence properties of RetouchFormer and the base model on FFHQR," noting that RetouchFormer converges to a higher PSNR and matches the base model's best result up to 6 times faster.

**SOURCES**

- L. Xie, W. Xue, W. Xu, Z. Xu, S. Wu, Z. Yu, and H. S. Wong, "Blemish-Aware and Progressive Face Retouching With Limited Paired Data," IEEE Conference on Computer Vision and Pattern Recognition, 2023.
- Z. Wang, J. Zhang, R. Chen, W. Wang, and P. Luo, "RestoreFormer: High-quality blind face restoration from undegraded key-value pairs," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. van den Oord, O. Vinyals, and K. Kavukcuoglu, "Neural discrete representation learning," Proceedings of Neural Information Processing Systems, 2017.
- B. Lei, X. Guo, H. Yang, M. Cui, X. Xie, and D. Huang, "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo," IEEE Conference on Computer Vision and Pattern Recognition, 2022.
- A. Shafaei, J. J. Little, and M. Schmidt, "AutoRetouch: automatic professional face retouching," IEEE Winter Conference on Applications of Computer Vision, 2021.
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M. H. Yang, and L. Shao, "Multi-stage progressive image restoration," IEEE Conference on Computer Vision and Pattern Recognition, 2021.
- T. Yang, P. Ren, X. Xie, and L. Zhang, "GAN prior embedded network for blind face restoration in the wild," IEEE Conference on Computer Vision and Pattern Recognition, 2021.