# 🎱 Han's Lotto Lab v2.0

**Brother Lotto System v2.0 Coverage Optimized — AI 기반 로또 번호 예측 웹 앱**

## ✨ 주요 기능

- **7가지 알고리즘**: 순수랜덤, 빈도가중, 밸런스, 합계구간, 패턴분산, 정밀필터, 하이브리드
- **z-score 기반 Hot/Cold 분류**: 절대빈도가 아닌 기대값 대비 상대빈도 분석
- **글로벌 커버리지 최적화**: 21세트가 가능한 한 많은 번호를 커버하도록 최적화
- **인기번호 회피**: 분배금 극대화를 위한 인기번호 페널티 시스템
- **데이터 기반 동적 필터**: 95% 신뢰구간 기반 합계/AC/홀짝/끝수합 필터

## 🚀 실행 방법

```bash
pip install -r requirements.txt
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## 📊 기술 스택

- **Backend**: Flask + Pandas + NumPy
- **Frontend**: Vanilla JS + CSS3 (프레임워크 없음)
- **배포**: Render.com (Gunicorn)

## ⚠️ 면책 조항

본 시스템은 통계적 분석 도구이며, 당첨을 보장하지 않습니다.
