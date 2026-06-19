# 00 병원 낙상 감지 모바일 앱

웹 대시보드를 Android 앱 안에서 그대로 열어 보는 WebView 앱입니다.

## 사용 순서

1. PC에서 `RUN_PUBLIC_DASHBOARD.bat`을 실행합니다.
2. 터미널에 출력되는 `https://...trycloudflare.com` 주소를 확인합니다.
3. Android 휴대폰에 `artifacts/00_hospital_fall_dashboard.apk`를 설치합니다.
4. 앱을 열고 Cloudflare Tunnel 주소를 입력합니다.

## 빌드 산출물

- `artifacts/00_hospital_fall_dashboard.apk`

## 다시 빌드

한글 경로에서는 Android 릴리즈 빌드가 실패할 수 있어서, 빌드는 영문 경로에서 진행하는 것을 권장합니다.

```powershell
$env:JAVA_HOME='C:\devtools\jdk17'
$env:ANDROID_HOME='C:\Android\Sdk'
$env:ANDROID_SDK_ROOT='C:\Android\Sdk'
$env:Path='C:\devtools\flutter\bin\mingit\cmd;C:\devtools\jdk17\bin;C:\devtools\flutter\bin;C:\Android\Sdk\platform-tools;C:\Android\Sdk\cmdline-tools\latest\bin;' + $env:Path
flutter build apk --release
```
