# BDA: Olist 판매자용 챗봇

브라질 전자상거래 플랫폼 Olist의 데이터로 **판매자를 돕는 챗봇**의 기능을 만든 기말 프로젝트예요.
판매자가 질문하면 챗봇(LLM)이 `seller_tools.py`의 도구 함수를 불러서, 전처리한 데이터를 근거로 답해요.

## 흐름

```
원본 CSV 9개 (Kaggle)  →  전처리  →  정리된 데이터 (preprocessing/*.parquet)
                                        └→ 챗봇 도구 (seller_tools.py, 23개)  →  메뉴별 노트북 (0~6)
```

## 폴더 구조

```
BDA/
├─ README.md
├─ seller_tools.py                 # 챗봇이 불러 쓰는 도구 함수 23개
├─ olist_preprocessing_v2.ipynb    # 전처리 노트북 (원본 → 정리된 데이터)
├─ olist_analysis_class_methods.ipynb  # 수업 기법(2~6장, regex)으로 한 분석 (Kaggle 원본 필요)
├─ predictor_experiments.ipynb     # 예측 모델 비교 실험 (지연, 도착일, 1~2점, 수요, 리뷰 글)
├─ 0_공통 ~ 6_블랙프라이데이 노트북    # 메뉴별 기능 시연 (아래 표)
└─ preprocessing/                  # 전처리 결과 데이터
   ├─ *.parquet                    # 정리된 데이터 10개 (주문, 주문×판매자, 상품, 판매 집계, 판매자·상품 통계, 리뷰, 이벤트)
   ├─ summaries.json               # 챗봇이 바로 꺼내 쓰는 요약표
   ├─ data_dictionary.json         # 열 설명
   └─ exclusion_log.json           # 데이터 정리 기록 (뺀 행·표시한 행과 이유)
```

## 메뉴별 노트북 (클릭하면 Colab에서 열려요)

| 노트북 | 기능 |
| --- | --- |
| [0_공통_브리핑_진단_챗봇연결.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/0_%EA%B3%B5%ED%86%B5_%EB%B8%8C%EB%A6%AC%ED%95%91_%EC%A7%84%EB%8B%A8_%EC%B1%97%EB%B4%87%EC%97%B0%EA%B2%B0.ipynb) | 오늘의 브리핑, 가게 진단, 자유 질문, 챗봇 연결법 |
| [1_배송지연_쿠폰.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/1_%EB%B0%B0%EC%86%A1%EC%A7%80%EC%97%B0_%EC%BF%A0%ED%8F%B0.ipynb) | 지연 책임, 발송 what-if, 도착일 안내, 쿠폰 금액·예산, 약속일 임박 경고 |
| [2_평점_리뷰.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/2_%ED%8F%89%EC%A0%90_%EB%A6%AC%EB%B7%B0.ipynb) | 별점·불만 유형, 리뷰 번역·답글, 1~2점 위험 주문, 나쁜 리뷰의 값, 묶음 주문 |
| [3_내고객이해.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/3_%EB%82%B4%EA%B3%A0%EA%B0%9D%EC%9D%B4%ED%95%B4.ipynb) | 약속 vs 기간, 지역 불만(쓰나미식 비율 barh), 권역별 배송일, 리뷰 속도, 날짜 세는 고객, 도착 전 리뷰, 지역 기회 |
| [4_발주.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/4_%EB%B0%9C%EC%A3%BC.ipynb) | 수요 예측, 발주량, 품절 위험 조정, 발주 우선순위, 이벤트 배수 |
| [5_생산량_매출.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/5_%EC%83%9D%EC%82%B0%EB%9F%89_%EB%A7%A4%EC%B6%9C.ipynb) | 시기별 매출 증감, 카테고리 순위·파이, 가격·운송비·할부, 생산 계획 |
| [6_블랙프라이데이_이벤트.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/6_%EB%B8%94%EB%9E%99%ED%94%84%EB%9D%BC%EC%9D%B4%EB%8D%B0%EC%9D%B4_%EC%9D%B4%EB%B2%A4%ED%8A%B8.ipynb) | 작년 BF 성적표, 카테고리별 배수, 내 가게 BF, 준비 체크리스트, 할인 시뮬레이터, 다른 이벤트 |
| [olist_analysis_class_methods.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/olist_analysis_class_methods.ipynb) | 수업 기법으로 한 분석: 정리, merge, groupby·pivot·crosstab, 쓰나미식 비율 barh, regex (Kaggle 원본 `archive.zip`을 Colab에 올려야 해요) |
| [predictor_experiments.ipynb](https://colab.research.google.com/github/jc020230/BDA/blob/main/predictor_experiments.ipynb) | 예측 모델 비교: 시간 순서 분할 + 경계 30일 간격, 검증 기간으로 모델 선택, 결과는 `preprocessing/experiments/model_report.json` |

**실행법**: 위 링크로 Colab에서 열고 **런타임 → 모두 실행**. 첫 셀이 이 저장소를 자동으로 내려받아서 `seller_tools.py`와 `preprocessing/`의 데이터를 찾아요. 노트북끼리는 독립이라 아무거나 먼저 열어도 돼요.

## 데이터

- 출처: [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). 원본(`archive.zip`)은 용량이 커서 저장소에 올리지 않았어요.
- 기간: 원본은 2016년 9월 ~ 2018년 10월이지만, 분석은 **2017년 1월 1일 ~ 2018년 8월 15일**만 써요. 2016년과 2018년 9~10월은 주문이 거의 없고, 8월 하순은 데이터 수집이 끊겨 주문이 다 들어오지 않았어요 (하루 250건 → 8/29 14건). 주간 표는 다 채워진 주(8/12까지), 월별 증감은 다 채워진 달(7월까지)만 써요.
- 정리된 데이터는 `preprocessing/` 폴더에 있어서 챗봇과 노트북은 원본 없이 바로 돌아가요.
- 데이터를 처음부터 다시 만들고 싶으면: Kaggle에서 `archive.zip`을 받아 [전처리 노트북](https://colab.research.google.com/github/jc020230/BDA/blob/main/olist_preprocessing_v2.ipynb)을 실행하세요.

## 기준 시점: 2018년 8월 15일

| 용도 | 기준 | 이유 |
| --- | --- | --- |
| 연구 분석 (`summaries.json`, 분석 노트북, 이벤트 배수) | 8/15까지 **구매한** 주문의 배송·리뷰 결과를 끝까지 반영 | 8/15 뒤 결과를 빼면 늦게 오는 주문만 골라서 빠져서 결과가 좋아 보이는 착시가 생겨요 |
| 챗봇의 현재 상태 (`seller_stats`, 도구의 진단·리뷰·브리핑, 예측 모델 학습) | 8/15을 '오늘'로 보고 **그날까지 도착·답변된** 정보만 | 챗봇이 "오늘 기준"으로 말할 때 미래 정보를 쓰지 않게 |
| 예측 실험 | 구매 날짜로 학습(~2017-12) / 검증(2018-01~03) / 시험(2018-04-01~08-15), 경계 직전 30일 구매 주문 제외 | 학습에 검증 기간의 결과가 섞이지 않게 |

## 주요 처리와 한계

- 테이블을 합칠 때 매출이 중복 계산되지 않게 주문 단위로 먼저 묶고, 매출 합계가 모든 표에서 일치하는지 검증했어요.
- 예측 입력에는 구매 시점에 알 수 있는 정보만 써요 (판매자 과거 이력도 그 주문 이전에 끝난 기록만).
- 쿠폰이 나쁜 리뷰를 막는 효과는 데이터에 없어서 가정값이에요 (도구 입력으로 바꿀 수 있어요).
- 데이터에 생산 정보는 없어서 "판매량 = 생산(준비)해야 할 양"으로 봐요.
