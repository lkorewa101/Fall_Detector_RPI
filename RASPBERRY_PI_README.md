# 라즈베리파이 실행 방법

이 압축 파일을 라즈베리파이에 풀고, 압축을 푼 폴더 안에서 실행합니다.

## 외부 인터넷에서도 접속

```bash
chmod +x RUN_ALL_APP_RPI.sh RUN_PUBLIC_DASHBOARD_RPI.sh
./RUN_PUBLIC_DASHBOARD_RPI.sh
```

실행하면 데스크탑 앱과 웹 서버가 켜지고, Cloudflare Tunnel 주소가 출력됩니다.

```text
https://xxxx.trycloudflare.com
```

이 주소를 다른 컴퓨터 브라우저나 안드로이드 앱에 입력하면 외부 장소에서도 볼 수 있습니다.

## 같은 와이파이/내부망에서만 접속

```bash
chmod +x RUN_ALL_APP_RPI.sh
./RUN_ALL_APP_RPI.sh
```

브라우저에서 아래 형식으로 접속합니다.

```text
http://라즈베리파이IP:8000
```

## 참고

- 라즈베리파이 OS Desktop 환경에서 실행하는 것을 권장합니다.
- 처음 실행할 때 Python, Node, PyQt 관련 패키지를 자동 설치할 수 있어 인터넷과 `sudo` 권한이 필요할 수 있습니다.
- 웹 화면은 라즈베리에서 Vite 개발 서버를 따로 켜지 않고, 백엔드 `8000`번에서 바로 표시되게 구성했습니다.
- Cloudflare 임시 주소는 실행할 때마다 바뀔 수 있습니다.
