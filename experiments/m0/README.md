# M0 첫 모델

- `past_test.py`: 2024년 말까지 학습하고 2025년 1~2월 발전량을 예측한다.
- `make_submission.py`: 과거 자료 전체로 학습해 대회 평가 기간의 제출 CSV를 만든다.
- 루트의 `requirements.txt`: 두 코드에 필요한 Python 패키지 목록.

실행:

```bash
.venv/bin/python experiments/m0/past_test.py
```

결과는 `outputs/m0_f5/site_predictions.parquet`에 저장된다. 발전소 253개 × 1,416시간의 예측이며, 이 파일을 팀원에게 전달해 **G1/G2 공식 점수**를 계산한다. `outputs/m0_f5/report.json`의 발전소별 평균 오차는 공식 점수가 아니다.
