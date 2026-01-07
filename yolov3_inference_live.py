import os
import cv2
import numpy as np
from ultralytics import YOLO
from team_assigner import TeamAssigner

def iou_bbox(boxA, boxB):
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
    return interArea / float(boxAArea + boxBArea - interArea)

def draw_legend(frame, team_colors: dict, gk_color: tuple, ball_color: tuple, referees_color: tuple):
    x0, y0 = 10, 10
    box_w, box_h = 20, 20
    spacing = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 1
    text_dx = box_w + 10

    items = ["Team 1", "Team 2", "Goalkeeper", "Ball", "Referees"]
    lh = spacing * len(items) + 10
    lw = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0-5, y0-5), (x0+lw, y0+lh), (50,50,50), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Team 1
    cv2.rectangle(frame, (x0, y0), (x0+box_w, y0+box_h), tuple(team_colors.get(1, (0,0,255))), -1)
    cv2.putText(frame, "Team 1", (x0+text_dx, y0+box_h-4), font, fs, (255,255,255), th, cv2.LINE_AA)
    # Team 2
    yy = y0 + spacing
    cv2.rectangle(frame, (x0, yy), (x0+box_w, yy+box_h), tuple(team_colors.get(2, (255,0,0))), -1)
    cv2.putText(frame, "Team 2", (x0+text_dx, yy+box_h-4), font, fs, (255,255,255), th, cv2.LINE_AA)
    # GK
    yy = y0 + spacing*2
    cv2.rectangle(frame, (x0, yy), (x0+box_w, yy+box_h), gk_color, -1)
    cv2.putText(frame, "Goalkeeper", (x0+text_dx, yy+box_h-4), font, fs, (255,255,255), th, cv2.LINE_AA)
    # Ball
    yy = y0 + spacing*3
    cv2.rectangle(frame, (x0, yy), (x0+box_w, yy+box_h), ball_color, -1)
    cv2.putText(frame, "Ball", (x0+text_dx, yy+box_h-4), font, fs, (255,255,255), th, cv2.LINE_AA)
    # Referees
    yy = y0 + spacing*4
    cv2.rectangle(frame, (x0, yy), (x0+box_w, yy+box_h), referees_color, -1)
    cv2.putText(frame, "Referees", (x0+text_dx, yy+box_h-4), font, fs, (255,255,255), th, cv2.LINE_AA)

def main():
    # 1) 모델 불러오기
    model = YOLO("models/best.pt")
    team_assigner = TeamAssigner()

    # 2) 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 카메라를 열 수 없습니다.")
        return
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3) 저장용 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_videos/live_output.mp4", fourcc, fps, (width, height))

    # 4) 첫 프레임으로 팀 색상 지정
    ret, first = cap.read()
    if not ret:
        print("[오류] 첫 프레임을 가져올 수 없습니다.")
        return
    res0 = model(first)[0]
    det0 = {}
    tidx = 0
    for box, cls_id in zip(res0.boxes.xyxy, res0.boxes.cls):
        name = res0.names[int(cls_id)].lower()
        if name in ("player","person"):
            x1,y1,x2,y2 = box.cpu().numpy().astype(int)
            det0[tidx] = {"bbox":[x1,y1,x2,y2]}
            tidx += 1
    team_assigner.assign_team_color(first, det0)
    fixed_colors = {tid: tuple(map(int,col)) for tid,col in team_assigner.team_colors.items()}
    gk_col   = (255,255,0)
    ball_col = (0,255,0)
    ref_col  = (0,255,255)

    prev_players   = {}
    next_player_id = 0

    # 5) 실시간 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model(frame)[0]
        players, gks, balls, refs = [],[],[],[]
        for box, cls_id in zip(res.boxes.xyxy, res.boxes.cls):
            x1,y1,x2,y2 = box.cpu().numpy().astype(int)
            cname = res.names[int(cls_id)].lower()
            if cname in ("player","person"):
                players.append([x1,y1,x2,y2])
            elif "goalkeeper" in cname or "gk" in cname:
                gks.append([x1,y1,x2,y2])
            elif "ball" in cname:
                balls.append([x1,y1,x2,y2])
            elif "referee" in cname or "ref" in cname:
                refs.append([x1,y1,x2,y2])

        # IoU 트래킹
        curr = {}
        used = set()
        for b in players:
            best_iou, best_id = 0, None
            for pid, pb in prev_players.items():
                if pid in used: continue
                val = iou_bbox(pb, b)
                if val>best_iou:
                    best_iou, best_id = val, pid
            if best_id is not None and best_iou>=0.5:
                curr[best_id]=b
                used.add(best_id)
            else:
                curr[next_player_id]=b
                next_player_id+=1
        prev_players = curr.copy()

        # 그리기
        ann = frame.copy()
        for pid, box in curr.items():
            team_id = team_assigner.get_player_team(frame, box, pid)
            color   = fixed_colors.get(team_id,(0,0,255))
            cv2.rectangle(ann, tuple(box[:2]), tuple(box[2:]), color, 2)
            cv2.putText(ann, f"P{pid}", (box[0],box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for b in gks:
            x1,y1,x2,y2 = b
            cv2.rectangle(ann,(x1,y1),(x2,y2),gk_col,2)
            cv2.putText(ann,"G",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,gk_col,2)
        for b in balls:
            x1,y1,x2,y2 = b
            cv2.rectangle(ann,(x1,y1),(x2,y2),ball_col,2)
            cv2.putText(ann,"Ball",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,ball_col,2)
        for b in refs:
            x1,y1,x2,y2 = b
            cv2.rectangle(ann,(x1,y1),(x2,y2),ref_col,2)
            cv2.putText(ann,"Ref",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,ref_col,2)

        draw_legend(ann, fixed_colors, gk_col, ball_col, ref_col)

        cv2.imshow("실시간 추론", ann)
        out.write(ann)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[완료] 실시간 추론 세션 종료됨.")

if __name__ == "__main__":
    main()
