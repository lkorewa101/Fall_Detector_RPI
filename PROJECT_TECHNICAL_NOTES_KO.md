# IWR6843ISK C3CD 낙상/심박/호흡 기술 메모

이 문서는 다른 개발 환경에서 현재 패키지의 구성과 작업 상태를 빠르게 복원하기 위한 기술 메모입니다. 경로는 이 문서가 들어 있는 패키지 루트 폴더를 기준으로 봅니다.

## 패키지 구성

- `firmware`: 바로 플래시해서 쓸 수 있는 완성 펌웨어 BIN과 cfg
- `desktop_app`: 수정된 PyQt 데스크탑 앱, pkl 모델, 설정, 테스트
- `web_dashboard`: 실시간 웹 대시보드 백엔드/프론트엔드
- `mobile_app`: Flutter 모바일 앱 소스와 APK 산출물
- `RUN_ALL_APP.bat`, `RUN_ALL_APP_RPI.sh`: 데스크탑 앱과 웹 서버 실행
- `RUN_PUBLIC_DASHBOARD.bat`, `RUN_PUBLIC_DASHBOARD_RPI.sh`: 데스크탑 앱, 웹 서버, 공개 터널 실행
- `VERIFY_COPIED_PACKAGE.bat`: 패키지 무결성 및 모델 로드 검증
- `BUILD_FIRMWARE_SOURCE.bat`: TI 빌드 도구가 있는 PC에서 펌웨어 소스 재빌드

## 현재 완성 기준

- 대상 보드: TI IWR6843ISK
- 완성 펌웨어: `firmware\firmware\iwr6843isk_fall_vital_3D_people_track_6843_demo.bin`
- SHA256: `C3CD0D8A0F32A0B03870065A5A5DE7B6B03AAAD75E206D90320C29F6004C498C`
- 기본 cfg: `firmware\cfg\vital_signs_ISK_6m_multi.cfg`
- 앱 기본 cfg: `desktop_app\configs\vital_signs_ISK_6m_multi.cfg`
- 60초 strict live gate 통과
- 한 명 누운/정적 조건에서 fallback vital subject 1명 안정 출력
- invalid vital record 0, malformed frame 0
- 데스크탑 앱 모델 로드 및 테스트 통과

## S1 스위치

펌웨어 플래시 모드:

```text
ON, OFF, ON, ON, OFF, X
```

앱 실행/일반 동작 모드:

```text
OFF, OFF, ON, ON, OFF, X
```

S1을 바꾼 뒤에는 반드시 `NRST`를 한 번 눌렀다 떼야 합니다.

## 같은 센서를 그대로 쓰는 경우

이미 C3CD 펌웨어가 올라간 같은 IWR6843ISK라면 다시 플래시하지 않습니다.

1. 센서 S1을 functional mode로 설정: `OFF, OFF, ON, ON, OFF, X`
2. `NRST` 누르고 떼기
3. `VERIFY_COPIED_PACKAGE.bat` 실행해서 패키지 확인
4. Windows는 `RUN_ALL_APP.bat`, Raspberry Pi는 `RUN_ALL_APP_RPI.sh` 실행
5. 앱에서 COM 포트 확인
   - 일반적으로 CLI: Enhanced COM Port
   - Data: Standard COM Port
6. 앱 로그에 `First valid TLV frame parsed. Started.`가 나오면 연결 성공

## 새 센서를 쓰는 경우

새 IWR6843ISK이거나 펌웨어가 다르면 먼저 플래시합니다.

1. S1을 flash mode로 설정: `ON, OFF, ON, ON, OFF, X`
2. `NRST` 누르고 떼기
3. `firmware\firmware\iwr6843isk_fall_vital_3D_people_track_6843_demo.bin` 플래시
4. S1을 functional mode로 변경: `OFF, OFF, ON, ON, OFF, X`
5. 다시 `NRST` 누르고 떼기
6. Windows는 `RUN_ALL_APP.bat`, Raspberry Pi는 `RUN_ALL_APP_RPI.sh` 실행

## 펌웨어 설명

이 펌웨어는 TI Radar Toolbox의 `3D People Tracking 6843`을 기반으로 만든 커스텀 C3CD 빌드입니다. 완전히 새로 만든 펌웨어가 아니라, TI 3D People Tracking에 심박/호흡과 낙상 이벤트 출력을 MSS 쪽에 추가한 구조입니다.

기존 3D People Tracking에서 유지한 것:

- point cloud 출력
- tracker target list 출력
- target index 출력
- target height 출력
- presence/people tracking 흐름

추가한 것:

- 심박/호흡 vital TLV `0x410`
- tracker가 없는 누운 자세용 fallback vital subject
- fallback vital ID와 실제 사람 ID 구분
- 낙상 이벤트 TLV `0xF001`
- CLI 명령 `vitalsign`, `VSRangeIdxCfg`
- invalid vital record 방지용 sanitize
- 앱과 맞는 TLV 파서/표시 규칙

## 바이탈 만든 방식

바이탈은 AI 모델이 아니라 펌웨어 C 코드의 phase 기반 추정기입니다.

핵심 파일:

- `people_tracking_fall\mss\vital_signs_estimator.c`
- `people_tracking_fall\mss\vital_signs_estimator.h`

처리 흐름:

1. 3D People Tracking 결과에서 tracker target을 확인
2. target이 있으면 target 거리 근처 range bin 선택
3. target이 없으면 `VSRangeIdxCfg`의 fallback range bin 사용
4. radar cube complex 값에서 좋은 range/antenna/chirp cell 선택
5. `atan2(imag, real)`로 phase 추출
6. phase unwrap 후 최대 300프레임 history 저장
7. phase history에서 주파수 탐색
8. 호흡은 `0.10 ~ 0.60 Hz` 범위, 즉 약 `6 ~ 36 bpm`
9. 심박은 `0.80 ~ 2.00 Hz` 범위, 즉 약 `48 ~ 120 bpm`
10. 품질, 최근 history, rate 안정성으로 후보 점수화
11. 좋은 후보만 UART TLV `0x410`으로 출력

vital TLV 구조:

```c
uint16 id;
uint16 rangebin;
float breathingDeviation;
float heartRate;
float breathingRate;
float heartHistory[15];
float breathHistory[15];
```

tracker가 잡힌 경우 `id`는 tracker ID입니다. tracker가 안 잡힌 경우 `id >= 0x8000`인 fallback ID를 씁니다. fallback ID는 사람 ID가 아니며 앱에서 사람 수로 세면 안 됩니다.

## 낙상 만든 방식

펌웨어 낙상은 AI가 아니라 tracker/height 기반 휴리스틱 이벤트입니다.

핵심 파일:

- `people_tracking_fall\mss\fall_detector.c`
- `people_tracking_fall\mss\fall_detector.h`

보는 값:

- tracker ID
- target height
- target maxZ/minZ
- 속도
- vertical velocity
- 이전 프레임 대비 height drop
- low posture 유지 여부

펌웨어 낙상 TLV는 `0xF001`입니다. 실제 최종 낙상 판단과 시각화는 데스크탑 앱의 AI/pkl/스켈레톤 처리도 같이 사용합니다.

## 코드 라인 수

TI 원본 3D People Tracking 기준:

```text
19 files / 8,793 lines
```

현재 C3CD 펌웨어 소스:

```text
25 files / 10,970 lines
```

증가분:

```text
+6 files / +2,177 lines
```

바이탈 핵심 코드:

```text
vital_signs_estimator.c  1292 lines
vital_signs_estimator.h    49 lines
total                    1341 lines
```

낙상 핵심 코드:

```text
fall_detector.c  276 lines
fall_detector.h   42 lines
total            318 lines
```

원본 수정량:

```text
mss_main.c       +303 / -80
pcount3D_cli.c    +77 / -9
```

## 앱 쪽 중요 수정

데스크탑 앱은 패키지 안 cfg를 우선 사용하도록 수정되어 있습니다. 예전 절대경로 cfg를 물고 가지 않도록 처리했습니다.

주요 수정:

- 패키지 내부 `configs\vital_signs_ISK_6m_multi.cfg` 우선 사용
- Data UART read를 gate/monitor 방식에 가깝게 변경
- watchdog이 시작된 스트림을 함부로 끊지 않도록 수정
- point cloud tilt 부호 보정
- ghost box 감소
- skeleton 크기 fitting
- velocity 색상 적용
- 신체 비율 튜닝 UI 제거

정상 연결 로그 예:

```text
First valid TLV frame parsed. Started.
Data UART chunk #1: ...
```

아래 로그가 나오면 Data UART는 살아 있고 파싱도 진행 중인 상태입니다.

```text
bytes=..., frames=..., lastRaw=0.1s
```

## 빠른 복원 요약

```text
이 폴더는 IWR6843ISK C3CD 낙상/심박/호흡 완성 패키지입니다.
패키지 루트 기준 README_FIRST.md와 PROJECT_TECHNICAL_NOTES_KO.md를 먼저 읽고 진행하세요.

센서는 TI IWR6843ISK입니다.
펌웨어 BIN은 `firmware\firmware\iwr6843isk_fall_vital_3D_people_track_6843_demo.bin` 입니다.
SHA256은 C3CD0D8A0F32A0B03870065A5A5DE7B6B03AAAD75E206D90320C29F6004C498C 입니다.

같은 센서를 쓰면 새로 플래시하지 말고 functional mode OFF, OFF, ON, ON, OFF, X 로 놓고 NRST 후 Windows는 `RUN_ALL_APP.bat`, Raspberry Pi는 `RUN_ALL_APP_RPI.sh`를 실행하세요.
새 센서면 flash mode ON, OFF, ON, ON, OFF, X 로 놓고 NRST 후 위 BIN을 플래시한 뒤 functional mode OFF, OFF, ON, ON, OFF, X 로 바꾸고 다시 NRST 하세요.

펌웨어는 TI 3D People Tracking 6843 기반에 vital TLV 0x410, fallback vital subject, fall TLV 0xF001, vitalsign/VSRangeIdxCfg CLI를 추가한 C3CD 빌드입니다.
바이탈은 AI가 아니라 radar cube complex phase를 추출해 phase unwrap/history/주파수 탐색으로 심박과 호흡을 추정합니다.
fallback vital ID는 0x8000 이상이며 실제 사람 ID가 아니므로 사람 수로 세면 안 됩니다.

데스크탑 앱은 `desktop_app` 안에 있고, pkl 모델과 configs가 포함되어 있습니다.
먼저 `VERIFY_COPIED_PACKAGE.bat`로 패키지를 검증하고, 실행은 Windows에서 `RUN_ALL_APP.bat` 또는 `RUN_PUBLIC_DASHBOARD.bat`, Raspberry Pi에서 `RUN_ALL_APP_RPI.sh` 또는 `RUN_PUBLIC_DASHBOARD_RPI.sh`를 사용하세요.
```
