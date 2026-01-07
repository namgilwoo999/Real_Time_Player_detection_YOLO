# FOOTBALL_ANALYSIS

YOLOv3 기반 축구 경기 영상 분석 시스템으로, 선수/골키퍼/심판/공을 자동으로 탐지하고 팀 색상을 자동 할당하여 추적합니다.

## Problem Statement

### Background
축구 경기 영상 분석에서 **선수를 팀별로 구분**하는 것은 모든 분석의 전제조건입니다. 기존에는 수동으로 팀을 표시하거나, 사전에 유니폼 색상을 입력해야 했습니다.

### Problem
- **수동 입력의 번거로움**: 경기마다 유니폼 색상을 매번 설정해야 함
- **조명 변화 대응 불가**: 경기장 조명, 그림자로 색상 인식 실패
- **잔디 색상 간섭**: 초록색 유니폼 팀은 잔디와 혼동되어 인식 오류
- **실시간 처리 요구**: 웹캠 입력 시 즉각적인 팀 구분 필요

### Objective
경기 영상에서 **유니폼 색상을 자동으로 학습하여 팀을 구분**하는 실시간 시스템
- **자동 색상 학습**: 첫 프레임에서 KMeans 클러스터링으로 2개 팀 색상 자동 추출
- **잔디 마스킹**: HSV 색공간에서 잔디 영역 제거로 정확도 향상
- **IoU 추적**: 프레임 간 선수 ID 유지로 안정적인 추적

### Value Proposition
- **자동화**: 사용자 입력 없이 즉시 팀 구분 가능
- **범용성**: 어떤 유니폼 조합에도 작동 (빨강 vs 파랑, 흰색 vs 검정 등)
- **교육용**: 유소년 팀 경기 분석, 전술 교육에 활용 가능
- **실시간 지원**: 웹캠 모드로 훈련 중 즉시 피드백

### Approach
- **YOLOv3** 선수/골키퍼/심판/공 탐지
- **KMeans(n=2)** 유니폼 색상 클러스터링 (잔디 마스킹 적용)
- **IoU > 0.5** 기반 선수 ID 추적
- **실시간 범례** 화면 상단 표시

---
## Features

- **자동 객체 탐지**: YOLOv3 모델로 선수, 골키퍼, 심판, 공 실시간 탐지
- **팀 색상 자동 인식**: KMeans 클러스터링으로 선수 유니폼 색상 분석 및 팀 자동 분류
- **IoU 기반 추적**: 프레임 간 객체 ID 유지로 안정적인 추적
- **비디오 파일 분석**: 녹화된 경기 영상 분석 (yolov3_inference.py)
- **실시간 웹캠 분석**: 라이브 카메라 입력 분석 (yolov3_inference_live.py)
- **시각화**: 색상별 바운딩 박스 + 범례로 직관적인 결과 제공

## Demo

### 입력 영상
축구 경기 영상 (mp4, avi 등)

### 출력 결과
- 선수: 팀별로 다른 색상의 바운딩 박스
- 골키퍼: 노란색 바운딩 박스
- 심판: 노란색 바운딩 박스
- 공: 초록색 바운딩 박스
- 화면 상단 범례: Team 1, Team 2, Goalkeeper, Ball, Referees

## Installation

### 1. Requirements

```bash
pip install -r requirements.txt
```

필수 패키지:
- Python 3.8+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- ultralytics >= 8.0.0
- scikit-learn >= 1.3.0
- torch >= 2.0.0

### 2. YOLOv3 모델 다운로드

프로젝트에서 사용하는 파인튜닝된 YOLOv3 모델 파일 (`best.pt`)을 다운로드하여 `models/` 폴더에 배치하세요.

**다운로드 링크**: [Releases 페이지에서 다운로드](https://github.com/당신의유저명/FOOTBALL_ANALYSIS/releases)

```bash
# 폴더 구조
FOOTBALL_ANALYSIS/
├── models/
│   └── best.pt    # 여기에 모델 파일 배치
```

## Usage

### 비디오 파일 분석

녹화된 축구 경기 영상을 분석합니다.

```bash
python yolov3_inference.py
```

**설정 변경**:
- 입력 파일 경로: `yolov3_inference.py` 파일 내 `video_path` 변수 수정
- 출력 파일 경로: `output_path` 변수 수정

```python
# yolov3_inference.py (라인 약 340)
video_path = "input_videos/ex.mp4"          # 입력 경로
output_path = "output_videos/result.mp4"    # 출력 경로
```

### 실시간 웹캠 분석

웹캠을 통해 실시간으로 축구 경기를 분석합니다.

```bash
python yolov3_inference_live.py
```

- 종료: 키보드에서 `q` 키 누르기
- 출력 파일: `output_videos/live_output.mp4` 자동 저장

**카메라 번호 변경**:
```python
# yolov3_inference_live.py (라인 약 140)
cap = cv2.VideoCapture(0)  # 0: 기본 카메라, 1,2,3...: 다른 카메라
```

## Project Structure

```
FOOTBALL_ANALYSIS/
├── yolov3_inference.py          # 비디오 파일 분석 메인 스크립트
├── yolov3_inference_live.py     # 실시간 웹캠 분석 스크립트
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py         # TeamAssigner 클래스 (팀 색상 할당)
├── utils/
│   ├── __init__.py
│   └── video_utils.py           # 비디오 읽기/저장 유틸리티
├── models/
│   └── best.pt                  # YOLOv3 파인튜닝 모델
├── input_videos/                # 입력 영상 폴더
├── output_videos/               # 출력 영상 저장 폴더
├── requirements.txt             # Python 패키지 의존성
└── README.md                    # 프로젝트 설명서
```

## How It Works

### 1. 객체 탐지 (YOLOv3)
- 파인튜닝된 YOLOv3 모델로 각 프레임에서 선수, 골키퍼, 심판, 공 탐지
- 클래스: `players`, `goalkeepers`, `referees`, `ball`

### 2. 팀 색상 할당 (TeamAssigner)
```python
class TeamAssigner:
    def assign_team_color(self, frame, detections)
    def get_player_team(self, frame, bbox, player_id)
```

**알고리즘**:
1. 선수 바운딩 박스 상반부 영역 추출
2. HSV 색공간 변환 후 잔디(초록색) 마스킹
3. KMeans(n=2) 클러스터링으로 2개 주요 색상 추출
4. 채도가 높은 클러스터를 유니폼 색상으로 선택
5. 전체 선수의 색상을 다시 KMeans(n=2)로 클러스터링하여 2개 팀으로 분류

### 3. IoU 기반 추적
```python
def iou_bbox(boxA, boxB):
    # Intersection over Union 계산
```

- 프레임 간 바운딩 박스 IoU 계산
- IoU > 0.5인 경우 동일 객체로 판단하여 ID 유지
- 새로운 객체는 새 ID 부여

### 4. 시각화
```python
def draw_legend(frame, team_colors, gk_color, ball_color, referees_color):
    # 화면 상단에 범례 표시
```

- Team 1: 빨간색 (또는 감지된 팀 색상)
- Team 2: 파란색 (또는 감지된 팀 색상)
- Goalkeeper: 노란색
- Ball: 초록색
- Referees: 노란색

## Configuration

### 색상 커스터마이징

`yolov3_inference.py` 또는 `yolov3_inference_live.py`에서 색상 변경:

```python
# 고정 색상 (BGR 형식)
gk_color = (255, 255, 0)        # 골키퍼: 노란색
ball_color = (0, 255, 0)        # 공: 초록색
refs_color = (0, 255, 255)      # 심판: 노란색

# 팀 색상 (자동 할당 또는 수동 설정)
fixed_team_colors = {
    1: (0, 0, 255),    # Team 1: 빨간색
    2: (255, 0, 0)     # Team 2: 파란색
}
```

### IoU 임계값 조정

추적 민감도 조정:

```python
iou_threshold = 0.5  # 낮출수록 더 쉽게 새 ID 부여, 높일수록 더 엄격하게 추적
```

## Troubleshooting

### 모델 로딩 오류
```
Error: Can't load model from models/best.pt
```
**해결**: `models/best.pt` 파일이 존재하는지 확인. Releases에서 다운로드.

### 비디오 파일 읽기 오류
```
ValueError: Can't open video file
```
**해결**:
- 파일 경로가 올바른지 확인
- 지원 형식인지 확인 (mp4, avi, mov 등)
- OpenCV가 해당 코덱을 지원하는지 확인

### 웹캠이 열리지 않음
```
Can't open camera
```
**해결**:
- 카메라가 다른 프로그램에서 사용 중인지 확인
- 카메라 번호를 0, 1, 2 등으로 변경해보기
- 카메라 권한 확인

### 팀 색상이 잘못 할당됨
**해결**:
- 수동으로 팀 색상 고정: `fixed_team_colors` 사전 수정
- 경기 초반 프레임에서 팀 색상 할당이 더 정확함 (선수가 많이 보일 때)

## Performance

- **처리 속도**: GPU 사용 시 약 30 FPS (RTX 3060 기준)
- **정확도**: YOLOv3 모델 성능에 따라 달라짐 (파인튜닝 데이터셋 품질 의존)
- **메모리 사용**: 약 2GB VRAM (GPU), 4GB RAM

## Limitations

- 선수가 겹치면 일부 탐지 누락 가능
- 극단적인 조명 조건에서 팀 색상 인식 정확도 하락
- 카메라 각도가 너무 멀거나 가까우면 성능 저하
- 유니폼 색상이 매우 유사한 두 팀은 구분 어려움

## Future Improvements

- [ ] SORT/DeepSORT 알고리즘 적용으로 더 안정적인 추적
- [ ] YOLOv8/YOLOv9로 모델 업그레이드
- [ ] 선수별 이동 경로 시각화
- [ ] 포메이션 분석 기능
- [ ] 패스 네트워크 분석
- [ ] 히트맵 생성

## License

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

**모델 라이선스**: YOLOv3 모델은 Ultralytics의 라이선스를 따릅니다.

## Acknowledgments

- [Ultralytics YOLOv3](https://github.com/ultralytics/ultralytics) - 객체 탐지 모델
- OpenCV - 컴퓨터 비전 라이브러리
- scikit-learn - KMeans 클러스터링

## Contact

프로젝트에 대한 질문이나 제안사항이 있으시면 Issue를 열어주세요.

---

**Last Updated**: 2026-01-03
