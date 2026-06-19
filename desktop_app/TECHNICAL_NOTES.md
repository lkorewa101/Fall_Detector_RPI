# Desktop App Technical Notes

## Project Summary
- Project type: Python desktop radar monitoring / fall-detection app using `PyQt5` and `pyqtgraph.opengl`.
- Main entrypoint: `main.py`.
- Runtime target: TI `IWR6843ISK` with the custom fall/vital 3D People Tracking firmware.
- OOB configs and Visualizer-generated config assets have been removed from this workspace.

## Required Runtime Context
- Expected board firmware: `iwr6843isk_fall_vital_3D_people_track_6843_demo.bin`.
- Default config: `configs/vital_signs_ISK_6m_multi.cfg`.
- The config must contain `trackingCfg`, `sensorPosition`, and `vitalsign`, with
  `trackingCfg` allowing at least 6 tracks for the current firmware package.
- The app rejects older non-vital configs and current-incompatible configs, then
  falls back to the current fall/vital config.
- CSV/ZIP replay accepts Cartesian point columns, tall TI spherical point
  columns such as `range/azimuth/elevation/doppler`, and Zenodo-style wide list
  columns such as `range_0/azimuth_0/elevation_0/doppler_0` with common frame
  aliases such as `FrameNumber` and `frameNo`.

## Coordinate Policy
- Tracker target boxes are treated as already being in the tracker/world frame.
- Tracker boxes are not adjusted by the host-side height/tilt controls.
- Point clouds are aligned once in `main.py` using the UI height/tilt controls.
- The aligned point cloud is used consistently for rendering, recording, clustering, skeleton/AI input, and web payloads.
- Default point-height filter is `Z min=-0.3m`, `Z max=3.0m` to preserve floor-level fall points.

## Environment Setup
```powershell
py -3 -m pip install -r requirements.txt
py -3 main.py
```

## Files Most Worth Reading First
- `main.py`
- `modules/tlv_parser.py`
- `modules/processing.py`
- `modules/visualization_manager.py`
- `modules/config_manager.py`

## Known Verification Priorities
- Confirm live point cloud alignment against the tracker box in a controlled one-person scene.
- Confirm the rebuilt 6-track/2-UART-vital firmware on hardware once COM ports are visible.
- Verify that `height=0`, `tilt=0`, `Z min=-0.3`, `Z max=3.0` does not hide points.
- Tune point cloud height/tilt only after the box remains stable.
- Recheck fall-detection thresholds after coordinate alignment is correct.
- Use `tools/validate_mmwave_dataset.py` to verify D: datasets before training
  or replay. The latest D: fall dataset report is
  `reports/d_fall_dataset_validation.json`.
- Use `tools/validate_vital_datasets.py` for Mendeley multi-person IWR6843ISK
  ADC/reference ECG-PCG pairing. The latest report is
  `reports/d_mendeley_vital_dataset_validation.json`.

## 2026-05-19 Offline Patch
- Firmware package now contains SHA256
  `722D3D277B8BBEB1A5992FF50C4A3ADDFD48065ED6F4492C3C86432579F0802A`, which
  keeps six internal vital subject histories, caps public UART vital output to
  2 records per frame, and moves the `0x410` vital TLV before larger UART TLVs.
  It was flashed successfully on 2026-05-19 after reconnect. Strict live
  validation streamed real frames but still needs one clean rerun after
  functional-mode `NRST`.
- Desktop TLV parsing now accepts the vital-first packet order, filters invalid
  vital records before UI/web output, and exposes `invalidVitalRecords` /
  `malformedTlvs` in frame metadata.
- Root `vital_cli.py` now applies the same vital sanity filter and reports
  `invalid_vitals` in monitor summaries.
- CSV/ZIP replay now auto-detects ZIP members with point columns, so ordinary
  Cartesian fall datasets such as `frame,x,y,z,v` CSV bundles can be opened
  without `pointcloud` in the member name.
- `tools/validate_mmwave_dataset.py` now recurses into ZIP archives under a
  directory. The D: fall dataset root was checked into
  `reports/d_fall_dataset_validation.json`: 168 sources found, 161 OK, 7 empty
  source files, 0 parser errors; formats were 102 Cartesian CSVs and 66
  Zenodo-style wide spherical CSVs.
- Added `tools/validate_vital_datasets.py` for Mendeley multi-person vital data.
  The D: Mendeley IWR6843ISK report found 162 ADC cases, all complete with two
  target reference logs each: 162 ADC BIN files, 324 ECG/PCG CSV files, 0 pairing
  errors. The parser accepts the dataset's `positon` filename typo.
- The desktop AI loader now tries model candidates in order, validates
  `predict_proba()` at load time, and falls back from the NumPy-2-pickled Unity
  model to `models/latest/fall_model.pkl` when needed.
- Python runtime was verified with `numpy==1.26.4` and `scikit-learn==1.3.2`.
  The current fallback model loads and `predict_all()` returns successfully.
- 2026-05-19 hardware reconnect: COM3/COM4 were visible. Two C8AC flash
  attempts failed at bootloader connect while the application firmware was still
  running, then a later retry succeeded (`flash OK`). C8AC strict live gate
  failed by one UART-truncated frame / one invalid vital record at the third
  vital record tail. This 722D next-step note is historical and was superseded
  by later 0B77 and E17A builds.
- 2026-05-19 12:51 KST reconnect retry: package verifier passed, but raw
  bootloader probe timed out on COM4 and a short functional 2-record live gate
  captured 0 bytes / 0 frames. COM ports are visible, but S1/NRST must be
  latched into either flash mode or functional mode before the next command.
- 2026-05-19 12:56 KST flash helper was hardened with a watchdog subprocess for
  UniFlash, so future bad S1/NRST states should fail with a bounded diagnostic
  instead of hanging. `FLASH_IWR6843_FIRMWARE.ps1` now exposes
  `-FlashOperationTimeout`. Firmware release verifier passed after the change.
- 2026-05-19 12:56 KST desktop gates passed again:
  `python -m compileall -q main.py modules tests tools` and
  `python -m unittest discover -s tests -v` with 25 tests OK.
- 2026-05-19 13:30 KST added `tools/evaluate_fall_dataset.py`, which replays
  CSV/ZIP point-cloud sources through the current fall detector and writes a
  confusion-matrix JSON report. It fixes label inference for ZIPs under a root
  folder named `fall` by using the ZIP member path first. Latest report:
  `reports/d_fall_detector_replay_eval.json`, first 80 D: sources, `tp=0`,
  `tn=29`, `fp=0`, `fn=51`, `fall_recall=0.0`. This is a baseline/failure
  signal, not a completed accuracy gate.
- 2026-05-19 13:30 KST fall detector was extended with a guarded sustained
  low-posture-after-baseline-drop latch path. Regression now has 29 tests OK.
- 2026-05-19 18:15 KST replaced the older low-quality latest models with D:
  data-derived artifacts. Previous `models/latest` files were backed up under
  `models/backup_20260519_180945_before_d_models`.
- New runtime fall model:
  `models/latest/fall_model.pkl`, trained from D: fall point-cloud data with
  the same 11 features used by `FallAI`. Report:
  `reports/d_fall_frame_model_eval.json`; source-level test accuracy and
  balanced accuracy were both `0.9230769231` on 26 held-out sources.
- New sequence-level fall artifact:
  `models/latest/fall_sequence_model_d_fall.pkl`. It is now loaded by
  `modules.ai_model.FallSequenceAI` and contributes a smoothed sequence score
  inside `modules.fall_detector.HybridFallDetector`. Report:
  `reports/d_fall_sequence_model_eval.json`; held-out accuracy and balanced
  accuracy were both `0.9230769231`.
- New runtime skeleton model:
  `models/latest/pose_model.pkl`, loaded by `PoseEstimator` through the sklearn
  backend. Torch is not required for this runtime path. The labels are
  deterministic geometry pseudo-skeletons derived from D: mmWave point clouds,
  not external mocap ground truth. Report:
  `reports/d_pose_pseudo_model_eval.json`; median pseudo-joint error was
  `0.1719806045 m`.
- `models/latest/model_manifest.json` now reflects the D: model replacements.
  `tools/smoke_model_load.py` loaded the frame fall model, sequence fall model,
  and pose model successfully: `fall_enabled=True`, `sequence_enabled=True`,
  `pose_backend=sklearn`, `skeleton_shape=(28, 3)`.
- 2026-05-19 19:25 KST 722D hardware gate attempt after flash:
  `frames=220`, `malformedFrames=0`, `platformMismatches=0`,
  `vitalTlvs=440`, `vitalNonzero=376`, `maxVitalRecordsPerFrame=2`,
  `nonzeroVitalSubjects=2`, `invalidVitalRecords=2`. The two invalid records
  were isolated to non-current breath history buffer tail values; current
  heart/breath values were sane. A subsequent rerun captured 0 bytes, so the
  current physical next step is functional mode `S1 = OFF, OFF, ON, ON, OFF,
  X`, press/release `NRST`, then rerun strict live validation.
- 2026-05-19 19:40 KST after functional-mode `NRST`, the 722D strict live gate
  passed: capture `iwr6843_vital_capture_722d_2out_live_gate_after_nrst.bin`,
  `bytes=228064`, `frames=668`, `malformedFrames=0`,
  `platformMismatches=0`, `vitalTlvs=1336`, `vitalNonzero=1274`,
  `maxVitalRecordsPerFrame=2`, `nonzeroVitalSubjects=2`,
  `invalidVitalRecords=0`. This proved the 2-record UART cap, but it is
  historical; 0B77 was flashed later, and the current package is E17A awaiting
  flash/live validation. Remaining accuracy work requires real HR/RR reference
  values or a comparable TI/prebuilt reference capture.
- 2026-05-19 20:45 KST desktop runtime sequence integration:
  `FallSequenceAI` now loads `models/latest/fall_sequence_model_d_fall.pkl`,
  maintains recent per-track frame features in `HybridFallDetector`, and adds
  `sequence_score` / `should_run_sequence` diagnostics to detector details.
  The root launchers and monitor scripts were updated to use the package-local
  paths, so the desktop runtime can be started from the current package layout.
- Offline checks passed:
  `python -m unittest discover -s tests -v` ran 34 tests OK,
  `python -m compileall -q main.py modules tests tools` OK,
  `python tools\smoke_model_load.py` OK,
  `python vital_cli.py --self-test` OK,
- 2026-05-19 21:20 KST fall runtime replay was improved after removing the
  baseline-only AI gate and adding a guarded model-assisted latch for sustained
  grounded/low-posture tracks. The current D: replay reports are:
  `reports/d_fall_detector_replay_eval_sequence_latched_runtime_ai_recheck.json`
  for the first 80 labelled sources (`tp=47`, `tn=29`, `fp=0`, `fn=4`,
  `accuracy=0.95`, `fall_recall=0.9215686275`) and
  `reports/d_fall_detector_replay_eval_sequence_latched_runtime_ai_120.json`
  for the first 120 sources (`tp=47`, `tn=51`, `fp=0`, `fn=4`, `unknown=18`,
  labelled accuracy `0.9607843137`), and
  `reports/d_fall_detector_replay_eval_sequence_latched_runtime_ai_all168_final.json`
  for all 168 discovered D: sources (`tp=47`, `tn=51`, `fp=0`, `fn=4`,
  `unknown=66`, labelled accuracy `0.9607843137`). Unknown sources are excluded
  from confusion metrics because the path did not reliably encode fall/non-fall.
- `modules.ai_model` now caches loaded fall/sequence pickle models by path, so
  repeated `HybridFallDetector` instances in replay tools do not repeatedly
  unpickle the same large artifacts.
- 2026-05-19 20:50 KST firmware E17A was built after 0B77 passed strict live
  validation but still alternated one lying person between fallback range bins
  20 and 28 over time. E17A limits no-track fallback vital output to one
  subject. Release/user firmware packages now contain SHA256
  `E17ABD1D10A0700CE87873954856FBB9C3CD3167A2D5763D9F11C51559C904AC`, and
  `VERIFY_PACKAGE.ps1` passes. E17A is built/copied/package-verified but has
  not yet been flashed or live-gated.
- 2026-05-19 21:40-21:50 KST E17A flash retry did not reach a valid
  bootloader ACK. Package verifier and COM4/COM3 preflight passed, but the
  UniFlash operation exceeded the 300 s watchdog while waiting for bootloader.
  A follow-up `python tools\flash_iwr6843_uart.py --probe --cli-port COM4
  --timeout 8` returned unexpected byte `00` then timed out. This points to a
  physical S1/NRST latch issue, not a BIN/package issue. Next hardware step:
  set S1 to `ON, OFF, ON, ON, OFF, X`, press/release `NRST`, then run
  `powershell -NoProfile -ExecutionPolicy Bypass -File
  .\FLASH_IWR6843_FIRMWARE.ps1 -Flash -Yes -CliPort COM4 -DataPort COM3
  -ConnectDelay 30 -ConnectTimeout 60 -FlashOperationTimeout 300`. If a
  standalone `--probe` is run first, press/release `NRST` again before the real
  flash command because the probe can consume the bootloader handshake.
- 2026-05-19 22:06 KST firmware-release D: vital diagnostics were improved in
  `tools/offline_mendeley_adc_vital_estimator.py`: added `--candidate-mode
  mvdr` for a range-angle MVDR candidate path inspired by the Mendeley MATLAB
  workflow, added ADC-level `--match-mode pair-oracle` with unique assignment
  of candidates to Target1/Target2 references, and cached ADC estimates so
  target pairs are not recomputed. Full D: Mendeley reports:
  `reports/mendeley_adc_offline_estimate_all_t2_range_pair_oracle.json`
  (`rows=324`, HR MAE `9.361`, RR MAE `4.496`),
  `reports/mendeley_adc_offline_estimate_all_t2_mvdr_pair_oracle.json`
  (HR MAE `11.712`, RR MAE `5.611`),
  `reports/mendeley_adc_offline_estimate_all_t6_range_pair_oracle.json`
  (six-candidate upper bound HR MAE `5.584`, RR MAE `2.420`), and
  `reports/mendeley_adc_offline_estimate_all_t6_range_quality.json`
  (non-reference quality selector HR MAE `16.603`, RR MAE `4.375`). The
  conclusion is that the D: ADCs contain useful candidates, but selecting only
  two candidates without a reference remains the weak link; MVDR did not improve
  the full-dataset upper bound and should remain diagnostic for now.
- 2026-05-19 22:18-22:22 KST firmware C3CD superseded E17A and was flashed
  successfully. C3CD changes: breathing search upper band is now 0.60 Hz, and
  the two public UART vital records are selected using current nonzero status,
  recent valid HR/RR history count, HR/RR rate stability, breathing deviation,
  and a small fallback penalty. Release/user packages now contain SHA256
  `C3CD0D8A0F32A0B03870065A5A5DE7B6B03AAAD75E206D90320C29F6004C498C`, size
  `638916`. Flash log: package verifier OK, COM4/COM3 preflight OK, SFLASH
  erase OK, META_IMAGE1 download OK, `flash OK`. Next hardware step is not
  another flash: set S1 to `OFF, OFF, ON, ON, OFF, X`, press/release `NRST`,
  then run the 60 s strict C3CD live gate.
- First C3CD live-gate attempt after flash failed before capture because the
  CLI did not return `Done` for line 1 `sensorStop`. Follow-up preflight still
  showed COM4/COM3 visible and config OK, with no python process left. This
  means the board had not booted into the functional app CLI yet. Repeat the
  functional-mode S1 + `NRST` step, then rerun the live gate.
- 2026-05-19 22:45 KST after functional-mode `NRST`, the C3CD strict live
  gate passed: capture
  `iwr6843_vital_capture_c3cd_live_gate_after_nrst_retry.bin`,
  `bytes=136512`, `frames=667`, `malformedFrames=0`,
  `platformMismatches=0`, `vitalTlvs=667`, `vitalRecords=667`,
  `vitalNonzero=634`, `maxVitalRecordsPerFrame=1`,
  `nonzeroVitalSubjects=1`, `invalidVitalRecords=0`. Parsed details:
  tracker target-list TLVs were absent, point/object stats were
  `framesWithPoints=168`, `zeroPointFrames=499`, `objectsTotal=364`,
  `objectsMean=0.546`, `objectsMedian=0`, `objectsP90=2`, `objectsMax=23`.
  The single public vital subject was fallback ID `32789`, rangebin `23`,
  `samples=634`, first nonzero frame `33`, last nonzero frame `668`; HR bpm
  median/mean/p10/p90/last was `90.396/83.382/54.801/109.215/107.464`, and
  RR bpm median/mean/p10/p90/last was `13.181/13.352/7.812/21.008/26.917`.
  This passes the single-person lying fallback stability gate, but does not
  yet prove tracker-backed people count or two-person HR/RR accuracy.
- User acceptance update: two-person live testing is not physically available
  in the current environment, so the practical completion criterion is one
  detected stable vital subject with no malformed/invalid records plus the D:
  fall replay/model gates already completed. 기준 장비 대비 HR/RR 오차와 실제
  낙상 동작 시험은 separate future validation items, not blockers for the
  current technical notes.
- 2026-05-19 22:51 KST desktop visualization defaults were corrected after
  live screen feedback. `apply_point_cloud_alignment()` now treats positive
  TI `sensorPosition` tilt as downward sensor pitch when converting
  sensor-frame points to room coordinates; the same sign is used if firmware
  boxes are explicitly transformed. On the C3CD live capture this moved the
  median rendered point height from about `1.61 m` to about `0.41 m`, matching
  the lying/bed setup much better. App-side sparse-point clustering is now ON
  by default with `MinPts=4` instead of OFF/`34`, velocity coloring remains ON,
  track wireframes use approach/recede colors when velocity coloring is active,
  and the old debug red 3D background was removed. Verification after the
  patch: `python -m compileall -q main.py modules tests tools` OK,
  `python -m unittest discover -s tests -v` ran 36 tests OK, and
  `python tools\smoke_model_load.py` loaded fall/sequence/pose models OK.
