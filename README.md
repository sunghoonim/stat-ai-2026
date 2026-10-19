# 통계청 AI 교육 실습 노트북

**2026.10.27(화) · 4–6교시 딥러닝 기초 실습** · 강사: 임성훈 (KAIST 문화기술대학원 RVLab)

아래 표에서 **Open in Colab** 버튼을 누르면, 설치 없이 브라우저에서 바로 실습할 수 있습니다.
실습 1은 모델(CNN · RNN · Transformer) 강의가 끝난 뒤, Loss · Optimizer 강의 **전에** 모두 함께 합니다(20분, 학습 없이 추론만).
실습 2는 Loss · Optimizer 강의 뒤에 모두 함께 하고(50분), 실습 3–5는 관심 있는 것을 1~2개 골라서 해 보세요(각 25분). 오래 걸리는 칸은 저장된 결과를 보면 됩니다.

| 실습 | 내용 | 시간 | 런타임 | 열기 |
|---|---|---|---|---|
| 실습 1 · 학습 없이 써 보기 | 사전학습 모델로 **추론만**: CNN 사진 분류 · 모델 6개 바꿔 보기 · 필터 계산 · 층별 특징 맵, 가중치를 정해 둔 RNN(= 지수평활), Transformer 빈칸 맞히기 · Self-attention | 20분 | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghoonim/stat-ai-2026/blob/main/lab1_inference_cnn_rnn_transformer.ipynb) |
| 실습 2 · 딥러닝 4요소 기초 | Data · Model · Loss · Optimizer, 로지스틱 회귀 vs 신경망(인구센서스), 사전학습 문장 모델로 산업분류 | 50분 | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghoonim/stat-ai-2026/blob/main/lab2_deep_learning_basics.ipynb) |
| 실습 3 · 위성영상과 CLIP | 토지피복 분류: CNN 처음부터 vs 전이학습 vs CLIP(zero-shot · few-shot), 글로 사진 검색 | 25분 | CPU (GPU면 더 빠름) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghoonim/stat-ai-2026/blob/main/lab3_satellite_cnn_vs_clip.ipynb) |
| 실습 4 · 언어모델 바꿔 보기 | 뉴스 제목 분류로 문장 모델 4종 비교, 라벨 수에 따른 변화, 미세조정 | 25분 | CPU (GPU면 더 빠름) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghoonim/stat-ai-2026/blob/main/lab4_language_models.ipynb) |
| 실습 5 · 통계 지표 예측 | 실업률 · 취업자 등 월별 지표: 계절 naive · ETS · SARIMA vs 신경망 vs Chronos | 25분 | CPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunghoonim/stat-ai-2026/blob/main/lab5_time_series_forecasting.ipynb) |

## 시작하기

1. **개인 Google 계정**으로 로그인합니다. (기관 계정은 Colab이 막혀 있을 수 있습니다.)
2. 위 표에서 **Open in Colab** 버튼을 누릅니다.
3. 메뉴 **파일 → Drive에 사본 저장**을 누릅니다(위쪽의 **Drive로 복사** 버튼도 같습니다). 이걸 해야 고친 내용이 내 Google Drive에 남습니다.
4. 첫 실행 때 "이 노트북은 Google에서 작성하지 않았습니다"라는 경고가 나오면 **무시하고 계속**을 누릅니다.
5. 셀을 위에서부터 하나씩 실행합니다 (`Shift + Enter`). 한 번에 돌리려면 **런타임 → 모두 실행**.

## 알아 두면 좋은 것

- **실행 결과가 미리 저장되어 있습니다.** 실행이 막혀도 결과를 보며 따라오면 됩니다. 이 페이지에서 노트북 파일을 누르면 로그인 없이도 결과를 볼 수 있습니다.
- 처음 실행할 때는 데이터와 모델을 내려받느라 시간이 조금 걸립니다. 내려받기는 Colab 서버가 하므로 교육장 인터넷 속도와는 상관이 없습니다.
- 실습 3 · 4는 **런타임 → 런타임 유형 변경 → T4 GPU**로 바꾸면 빨라집니다. GPU가 없다고 나오면 CPU 그대로 해도 됩니다.
- 한동안 아무것도 하지 않으면 연결이 끊깁니다. 다시 연결한 뒤 위에서부터 다시 실행하세요.
- `HF_TOKEN`이 없다는 경고는 무시해도 됩니다. 모두 로그인이 필요 없는 공개 모델입니다.
- 다시 실행하면 숫자가 저장된 결과와 조금 다를 수 있습니다 (난수 · 버전 차이).

## 쓰는 데이터와 모델

| 실습 | 데이터 (출처 · 라이선스) | 모델 |
|---|---|---|
| 1 | scikit-image 예시 사진(고양이 · 커피 등), 통계청 월별 실업률 사본 [`data/kr_monthly_indicators.csv`](data/kr_monthly_indicators.csv) | ResNet18(ImageNet, 45MB) · 비교용 5종(ResNet50 · MobileNetV3 · EfficientNet-B0 · ViT-B/16 · ConvNeXt-Tiny, 약 0.6GB), klue/roberta-small(270MB) |
| 2 | Adult 인구센서스 (UCI · OpenML, CC BY 4.0), 사업체 설명 예시 문장 | 작은 신경망(직접 학습), 한국어 문장 모델 |
| 3 | EuroSAT 위성영상 (Helber et al. 2019, MIT) · Hugging Face `tanganke/eurosat` | ResNet18, CLIP |
| 4 | KLUE-YNAT 뉴스 제목 (KLUE 2021, CC BY-SA 4.0) · Hugging Face `klue/klue` | 다국어 · 한국어 문장 모델, klue/roberta-small |
| 5 | 통계청 월별 지표 (OECD 경유, FRED) · 사본: [`data/kr_monthly_indicators.csv`](data/kr_monthly_indicators.csv) | Chronos-Bolt, Chronos-2 |

모든 데이터와 모델은 공개 자료이고, 노트북이 실행될 때 자동으로 내려받습니다.
