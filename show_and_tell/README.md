# Show And Tell: A Neural Image Caption Generator

- [Paper Link](https://arxiv.org/abs/1411.4555)
- [Flickr 8k Dataset Link](https://www.kaggle.com/adityajn105/flickr8k?select=captions.txt)

## Paper Short Review

### Abstract

- Computer Vision과 Machine Translation을 결합하여 이미지를 설명하는 자연어를 생성하는 모델을 제안함.
- 모델(NIC)은 학습 이미지를 넣었을 때 설명하는 문장의 최대 가능도를 학습하며, SOTA 이상의 성능을 달성.

### Introduction

Image Description의 Task는 본래 매우 어려웠다. 이미지를 설명하기 위해서는 이미지 속 객체들을 포착해야할 뿐만 아니라 각 객체들간의 연관성을 고려해야하는데, 그뿐만아니라 그러한 연관성을 자연어로 표현해야 하기 때문이다.

이전까진 해당 과제를 sub-problems로 나누어 해결하고자 하는 시도들이 있었지만, 본 논문에서는 이미지 I가 들어왔을 때 설명문장(시퀀스) S에 대한 `최대가능도 p(S|I)`를 학습하는 sigle joint model을 제안한다.

이 작업은 Machine Translation 분야에서 SOTA를 기록하고있는 RNN에서 영감을 받았으며, 본 모델은 RNN의 encoder 부분을 CNN으로 대체한 뒤, CNN의 output을 RNN decoder의 입력으로 사용한다.

### Related Work

이전까지의 관련 연구들에서 나타났던 문제점들은 제한된 도메인에서만 작동한다거나, 객체들의 구성요소들을 설명하지 못하거나, Evaluating 과정에서 적합한 설명을 만들지 못했다는 것이다.

반면, 본 연구에서는 Image Classification을 위한 Convolution Network와 Sequence Modeling을 위한 Recurrent Network를 결합하여 하나의 end-to-end network를 만들었다는 것에 의의가 있다.

### Model

Machine Translation Task에서는 training과 inference phase에서 input 문장에 대한 Target 번역문의 확률을 바로 최대화하는 방식의 end-to-end 모델이 SOTA를 기록했다. 그러한 모델들은 RNN을 사용하고 있으며, 다양한 길이의 input sequence를 고정된 차원의 벡터로 표현하고 output으로 decode하게 된다. 이것과 마찬가지로 NIC에서는 input sentence 대신에 input 이미지를 동일한 차원의 벡터로 표현한 뒤 `translating` 원칙을 적용하여 description 문장을 decode하게 된다.

이 때 적절한 설명문에 대한 최대가능도를 계산하는 공식은 아래와 같다.

$$
\theta^{\star}=\arg \max _{\theta} \sum_{(I, S)} \log p(S \mid I ; \theta)
$$

$\theta$ 는 모델의 parameters를 의미하며, $I$ 는 이미지, $S$ s는 적절한 설명문을 의미하며, 그 길이는 가변적이다. 그리고 chain rule에 의거하여 모델은 결합확률을 계산하게 되며, 그 공식은 아래와 같다.

$$
\log p(S \mid I)=\sum_{t=0}^{N} \log p\left(S_{t} \mid I, S_{0}, \ldots, S_{t-1}\right)
$$

이렇게 나오는 log 확률들의 합을 optimize하게 되고, 학습 시 SGD를 optimizer로 사용한다.

본 논문에서 제안하는 모델은 RNN의 hidden state를 더 정교하게 디자인하기위해 LSTM을 사용한다.

#### LSTM-based Sentence Generator

- LSTM을 사용하는 이유는 RNN에서 흔히 발생하는 기울기 소실과 폭발에 대한 가능성 때문이다.
- LSTM의 핵심은 memory cell이 매 타임스텝마다 input의 관측치를 학습하는 것이며, 이러한 셀들은 세 개의 gate에 의해 조절된다.
- 이미지와 각각의 sentence word들은 LSTM 내에서 동일한 파라미터를 공유한다.

$$
\begin{aligned}
x_{-1} &=\operatorname{CNN}(I) \\
x_{t} &=W_{e} S_{t}, \quad t \in\{0 \ldots N-1\} \\
p_{t+1} &=\operatorname{LSTM}\left(x_{t}\right), \quad t \in\{0 \ldots N-1\}
\end{aligned}
$$

