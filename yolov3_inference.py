import os
import cv2
import numpy as np
from ultralytics import YOLO
from team_assigner import TeamAssigner
from utils.video_utils import read_video, save_video

def iou_bbox(boxA, boxB):
    """
    두 bounding box(boxA, boxB)가 [x1, y1, x2, y2] 형태로 주어졌을 때 IoU(Intersection-over-Union)를 계산해서 반환합니다.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_legend(frame, team_colors: dict, gk_color: tuple, ball_color: tuple, referees_color: tuple):
    """
    왼쪽 상단에 범례(Legend)를 그립니다.
      - team_colors: {1: BGR_color_for_team1, 2: BGR_color_for_team2}
      - gk_color: 골키퍼용 BGR 튜플 (예: (255,255,0))
      - ball_color: 공용 BGR 튜플 (예: (0,255,0))
      - referees_color: 심판용 BGR 튜플 (예: (0,255,255))
    """
    x0, y0 = 10, 10
    box_w, box_h = 20, 20
    spacing   = 30
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_offset_x = box_w + 10

    # 범례 배경 (반투명 사각형)
    legend_items = ["Team 1", "Team 2", "Goalkeeper", "Ball", "Referees"]
    legend_h = spacing * len(legend_items) + 10
    legend_w = 200
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0 - 5, y0 - 5),
        (x0 + legend_w, y0 + legend_h),
        (50, 50, 50),
        thickness=cv2.FILLED
    )
    alpha = 0.6
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Team 1
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + box_w, y0 + box_h),
        tuple(int(c) for c in team_colors.get(1, (0, 0, 255))),
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame, "Team 1",
        (x0 + text_offset_x, y0 + box_h - 4),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    # Team 2
    yy = y0 + spacing
    cv2.rectangle(
        frame,
        (x0, yy),
        (x0 + box_w, yy + box_h),
        tuple(int(c) for c in team_colors.get(2, (255, 0, 0))),
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame, "Team 2",
        (x0 + text_offset_x, yy + box_h - 4),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    # Goalkeeper
    yy = y0 + spacing * 2
    cv2.rectangle(
        frame,
        (x0, yy),
        (x0 + box_w, yy + box_h),
        tuple(int(c) for c in gk_color),
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame, "Goalkeeper",
        (x0 + text_offset_x, yy + box_h - 4),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    # Ball
    yy = y0 + spacing * 3
    cv2.rectangle(
        frame,
        (x0, yy),
        (x0 + box_w, yy + box_h),
        tuple(int(c) for c in ball_color),
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame, "Ball",
        (x0 + text_offset_x, yy + box_h - 4),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    # Referees
    yy = y0 + spacing * 4
    cv2.rectangle(
        frame,
        (x0, yy),
        (x0 + box_w, yy + box_h),
        tuple(int(c) for c in referees_color),
        thickness=cv2.FILLED
    )
    cv2.putText(
        frame, "Referees",
        (x0 + text_offset_x, yy + box_h - 4),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def main():
    # 1) 모델 및 모듈 초기화
    # YOLO 모델 가중치 경로
    model_weights = "models/best.pt"
    video_name    = "ex"
    project_dir   = os.getcwd()

    # 1-1) YOLO 모델 로드
    model = YOLO(model_weights)

    # 1-2) TeamAssigner 초기화
    team_assigner = TeamAssigner()

    # 2) 입력 비디오 로드
    input_path = os.path.join(project_dir, 'input_videos', f'{video_name}.mp4')
    try:
        frames, fps = read_video(input_path)
    except ValueError as e:
        print(f"[오류] 비디오 로드 실패: {e}")
        return
    print(f"[INFO] 총 프레임 수: {len(frames)}, FPS: {fps}")

    # 3) 첫 프레임에서만 팀 색상 계산 → 고정된 색상 맵 생성
    first_frame = frames[0]
    # 3-1) 첫 프레임에서 YOLO로 검출 수행
    results_0 = model(first_frame)[0]  # 첫 번째 프레임에 대한 결과
    names_dict = results_0.names       # {class_id: class_name} 형태

    # 3-2) 'player' 클래스(bbox)만 뽑아서 첫 프레임용 detections 생성
    first_detections = {}
    tmp_id = 0
    for box, cls_id in zip(results_0.boxes.xyxy, results_0.boxes.cls):
        class_name = names_dict[int(cls_id)]
        if class_name.lower() in ('player', 'person'):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            first_detections[tmp_id] = {'bbox': [x1, y1, x2, y2]}
            tmp_id += 1

    # 3-3) 첫 프레임의 선수들로 팀 색상 클러스터링 계산 (TeamAssigner)
    if len(first_detections) == 0:
        print("[경고] 첫 프레임에 검출된 선수(클래스 'player')가 없습니다.")
    else:
        team_assigner.assign_team_color(first_frame, first_detections)

    # 3-4) 고정된 팀별 색상 맵 복사 (NumPy 배열 → BGR 튜플)
    fixed_team_colors = {}
    for tid, color in team_assigner.team_colors.items():
        fixed_team_colors[tid] = tuple(int(c) for c in color)

    # 3-5) 골키퍼, 공, 심판용 색상 미리 지정
    gk_color   = (255, 255, 0)
    ball_color = (0, 255, 0)
    refs_color = (0, 255, 255)

    # 4) 프레임별 처리: 검출→간단 트래킹(ID 부착)→팀 할당→디스플레이
    processed_frames = []
    prev_players    = {}  # 이전 프레임 {player_id: bbox}
    next_player_id  = 0   # 새로운 ID를 줄 때마다 증가

    total_frames = len(frames)
    for idx, frame in enumerate(frames):
        # 4-1) 현재 프레임 YOLO 검출 수행
        results = model(frame)[0]
        names_dict = results.names

        # 4-2) 클래스별 bbox 리스트 분리
        players_bboxes    = []
        gk_bboxes         = []
        ball_bboxes       = []
        referees_bboxes   = []

        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            class_name = names_dict[int(cls_id)].lower()

            if class_name in ('player', 'person'):
                players_bboxes.append([x1, y1, x2, y2])
            elif 'goalkeeper' in class_name or 'gk' in class_name:
                gk_bboxes.append([x1, y1, x2, y2])
            elif 'ball' in class_name:
                ball_bboxes.append([x1, y1, x2, y2])
            elif 'referee' in class_name or 'ref' in class_name:
                referees_bboxes.append([x1, y1, x2, y2])
            else:
                # 그 외 클래스 무시
                pass

        # 4-3) 간단한 IoU 기반 트래킹
        curr_players = {}
        used_prev_ids = set()

        for box in players_bboxes:
            best_iou = 0.0
            best_id  = None
            for pid_prev, prev_box in prev_players.items():
                if pid_prev in used_prev_ids:
                    continue
                iou_val = iou_bbox(prev_box, box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = pid_prev

            if best_id is not None and best_iou >= 0.5:
                # 이전 프레임의 선수 ID 그대로 가져오기
                curr_players[best_id] = box
                used_prev_ids.add(best_id)
            else:
                # 새로운 ID 부여
                curr_players[next_player_id] = box
                next_player_id += 1

        # 4-4) prev_players 갱신
        prev_players = curr_players.copy()

        # 4-5) 팀 할당 및 색깔 적용
        #       TeamAssigner.get_player_team(frame, bbox, player_id) 사용
        for pid, bbox in curr_players.items():
            # TeamAssigner에 넘겨서 팀 ID 얻기 (1 또는 2)
            team_id = team_assigner.get_player_team(frame, bbox, pid)
            # 미리 계산해둔 고정 색상맵에서 꺼내오기
            color = fixed_team_colors.get(team_id, (0, 0, 255))
            # 바운딩 박스와 ID, 팀 색상까지 info에 담아서 나중 그리기 위해 덮어쓰기
            curr_players[pid] = {
                'bbox': bbox,
                'team_color': color
            }

        # 4-6) 프레임에 모든 객체 그리기
        annotated = frame.copy()

        # 4-6-1) 선수(Player) 그리기
        for pid, info in curr_players.items():
            x1, y1, x2, y2 = info['bbox']
            color = info['team_color']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(
                annotated,
                f"P{pid}",  # 선수 ID 표시
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        # 4-6-2) 골키퍼(Goalkeeper) 그리기
        for box in gk_bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), gk_color, thickness=2)
            cv2.putText(
                annotated,
                "G",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                gk_color,
                2,
                cv2.LINE_AA
            )

        # 4-6-3) 공(Ball) 그리기
        for box in ball_bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), ball_color, thickness=2)
            cv2.putText(
                annotated,
                "Ball",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ball_color,
                2,
                cv2.LINE_AA
            )

        # 4-6-4) 심판(Referees) 그리기
        for box in referees_bboxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), refs_color, thickness=2)
            cv2.putText(
                annotated,
                "Ref",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                refs_color,
                2,
                cv2.LINE_AA
            )

        # 4-7) 범례(legend) 그리기 (항상 고정된 색상 사용)
        draw_legend(
            annotated,
            fixed_team_colors,
            gk_color,
            ball_color,
            refs_color
        )

        processed_frames.append(annotated)

        # 진행 상황 출력 (100프레임마다 또는 마지막 프레임)
        if (idx + 1) % 100 == 0 or (idx + 1) == total_frames:
            print(f"[진행] 프레임 {idx+1}/{total_frames} 처리 완료")

    # 5) 결과 비디오 저장
    output_path = os.path.join(project_dir, "output_videos", "result.mp4")
    try:
        save_video(processed_frames, output_path, fps)
    except ValueError as e:
        print(f"[오류] 비디오 저장 실패: {e}")
        return

    print(f"[완료] 결과 비디오 저장 경로: {output_path}")


if __name__ == "__main__":
    main()
