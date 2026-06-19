# 2026-05-19 IWR6843ISK 낙상감지기 완성본

이 폴더는 다른 컴퓨터 또는 다음 작업에서 그대로 이어 쓰기 위한 패키지입니다.

## 폴더 구성

- `firmware`: IWR6843ISK C3CD 펌웨어 BIN과 cfg 포함
- `desktop_app`: 수정된 데스크탑 앱, pkl 모델, 테스트/학습/검증 도구 포함
- `web_dashboard`: 실시간 웹 대시보드 백엔드/프론트엔드 포함
- `mobile_app`: Flutter 모바일 앱 소스와 APK 산출물 포함
- `PROJECT_TECHNICAL_NOTES_KO.md`: 다른 개발 환경에서 프로젝트를 복원하기 위한 상세 기술 메모

## 현재 완료 기준

- C3CD 펌웨어 flash OK
- 60초 strict live gate 통과
- 한 명 fallback vital subject 안정 검출
- invalid vital record 0, malformed frame 0
- 앱 pkl 모델 로드 OK
- 데스크탑 앱 단위 테스트 통과

## 펌웨어 정보

- BIN: `firmware\firmware\iwr6843isk_fall_vital_3D_people_track_6843_demo.bin`
- SHA256: `C3CD0D8A0F32A0B03870065A5A5DE7B6B03AAAD75E206D90320C29F6004C498C`
- 기본 config: `firmware\cfg\vital_signs_ISK_6m_multi.cfg`
- S1 flash mode: `ON, OFF, ON, ON, OFF, X`
- S1 functional mode: `OFF, OFF, ON, ON, OFF, X`
- `NRST`는 S1을 바꾼 뒤 눌렀다 떼야 합니다.

## 펌웨어 소스/빌드

- 실제 수정/빌드 대상: `people_tracking_fall`
- 참고 원본/비교용: `sdk_oob_mmw`
- 빌드 도구 예시: `C:\ti\xdctools_3_50_08_24_core\gmake.exe`
- 루트의 `BUILD_FIRMWARE_SOURCE.bat`를 실행하면 `people_tracking_fall`에서 `gmake all`을 수행합니다.
- 다시 빌드한 결과 BIN은 `people_tracking_fall\3D_people_track_6843_demo.bin`에 생성됩니다.

다른 컴퓨터에서 소스로 다시 빌드하려면 TI mmWave SDK/Radar Toolbox 계열 의존성, TI ARM/C674 compiler, XDCtools/gmake가 같은 경로 또는 BAT 파일에 지정한 경로로 설치되어 있어야 합니다.

## 앱 실행

1. Windows: `RUN_ALL_APP.bat`
2. Raspberry Pi: `RUN_ALL_APP_RPI.sh`
3. 외부 공개 대시보드가 필요하면 `RUN_PUBLIC_DASHBOARD.bat` 또는 `RUN_PUBLIC_DASHBOARD_RPI.sh`

다른 컴퓨터에서는 Python 3.10, TI serial driver, 필요 시 UniFlash가 필요합니다.

## 앱 최신 시각화 수정

- TI `sensorPosition` 양수 tilt를 하향 센서 피치로 처리하도록 포인트 좌표 보정 부호 수정
- 여러 명 지원은 유지: `Max Tracks=6`
- 한 명 앉은 환경에서 잡음 상자 남발을 줄이도록 기본 `MinPts=6`
- 다중 프레임 잔상 통합 클러스터링 기본 OFF
- 화면 wireframe ghost box는 현재 프레임에서 실제 업데이트된 track만 표시
- 신체 비율 튜닝 UI 제거
- 스켈레톤은 실제 포인트 범위에 맞춰 과도하게 커지지 않게 축소/fitting
- 일반 모드의 포인트 색상은 Doppler 값 기준으로 표시

## AI 모델 및 데이터 출처 메모

- `fall_model_fallsim2_unity.pkl` 계열 Unity 학습/시뮬레이션 모델은 CMU 모션 캡처 기반 FBX 동작 데이터를 활용한 synthetic 학습 모델입니다.
- 관련 데이터셋: https://huggingface.co/datasets/gbionics/cmu-fbx
- 해당 Hugging Face 데이터셋의 라이선스 표기는 `cmu-mocap`입니다. 데이터셋 카드 기준으로 원본 애니메이션은 CMU Motion Capture Database 이용 조건을 따릅니다.
- 사용 가능 범위 요약: 복사, 수정, 재배포 및 연구/제품 내 사용은 가능하나, 모션 캡처 데이터 자체 또는 변환본을 데이터 상품처럼 직접 재판매하면 안 됩니다.
- 외부 발표, 논문, 보고서, 배포물에는 다음 출처를 표기합니다: CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu, NSF EIA-0196217 funding acknowledgement.
- 이 프로젝트의 레이더 실측 낙상 판단은 IWR6843ISK 실시간 포인트 클라우드와 자체 로직/모델을 사용하며, Unity/CMU 기반 모델은 보조 synthetic 학습 자원으로 구분합니다.
- Unity 시뮬레이션의 사람 모델은 MakeHuman Community로 생성/내보낸 모델을 사용했습니다: https://static.makehumancommunity.org/makehuman.html
- MakeHuman 공식 문서 기준으로 core/exported asset은 CC0로 안내되며, MakeHuman 소스코드는 AGPL 계열로 구분됩니다. 별도 의상, 헤어, 외부 애셋을 추가로 사용한 경우에는 해당 애셋의 개별 라이선스를 따로 확인해야 합니다.

## 빠른 검증 명령

펌웨어:

```powershell
cd firmware
powershell -NoProfile -ExecutionPolicy Bypass -File .\VERIFY_PACKAGE.ps1
```

펌웨어 소스 빌드:

```powershell
.\BUILD_FIRMWARE_SOURCE.bat
```

앱:

```powershell
cd desktop_app
set PYTHONPATH=%USERPROFILE%\py310_pkgs;%PYTHONPATH%
python tools\smoke_model_load.py
python -m unittest discover -s tests -v
```
