[HP Envy NVIDIA GPU 문제 및 해결 과정 정리]

1. 초기 환경

노트북:
- HP Envy
- Ubuntu 22.04.5 LTS / Windows 듀얼부팅

GPU:
- Intel UHD Graphics 620
- NVIDIA GeForce MX250 4GB

초기 Ubuntu 커널:
- Linux 6.8.0-124-generic

NVIDIA 드라이버:
- nvidia-driver-580 설치되어 있었음


2. 최초 문제 확인

GPU 존재 여부를 확인하기 위해 다음 명령을 실행함.

lspci | grep -Ei "vga|3d|display"

결과:
- Intel UHD Graphics 620 정상 인식
- NVIDIA GeForce MX250도 PCIe 장치로 인식

하지만

nvidia-smi

실행 시 다음 오류 발생.

NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.

즉, MX250 하드웨어 자체는 목록에 나타나지만 NVIDIA 드라이버와 GPU 간 통신이 되지 않는 상태였음.


3. 드라이버 상태 확인

ubuntu-drivers devices

확인 결과 Ubuntu에서는 nvidia-driver-535를 recommended로 표시했지만,
실제로 시스템에는 nvidia-driver-580이 설치되어 있었음.

Secure Boot:
- enabled

설치된 NVIDIA 드라이버:
- nvidia-driver-580
- nvidia-utils-580

NVIDIA 커널 모듈도 로드되어 있었음.

cat /proc/driver/nvidia/version
modinfo nvidia

결과:
- NVIDIA Kernel Module 580.159.03

따라서 단순히 NVIDIA 드라이버가 설치되지 않은 문제는 아니었음.


4. 실제 오류 발견

커널 로그를 확인함.

sudo dmesg -T | grep -iE 'nvidia|NVRM|nouveau|module verification'

다음 오류가 반복적으로 발생함.

NVRM: The NVIDIA GPU 0000:02:00.0
has fallen off the bus and is not responding to commands.

nvidia: probe of 0000:02:00.0 failed with error -1

또한 lspci에서 GPU가 다음과 같이 표시됨.

GeForce MX250 (rev ff)

여기서 중요한 증상은 두 가지였음.

1) rev ff
2) GPU has fallen off the bus

이는 NVIDIA 드라이버 파일이 없는 문제가 아니라,
GPU가 PCIe 버스에서 정상적으로 응답하지 못하고 있다는 의미로 판단됨.


5. 완전 전원 리셋 시도

PCIe 장치의 전원 상태 문제 가능성을 확인하기 위해:

1) Ubuntu 종료
2) 충전기 제거
3) 전원 버튼 약 15~20초 유지
4) 충전기 재연결
5) Ubuntu 재부팅

후 다시 확인함.

하지만 여전히:

GeForce MX250 (rev ff)

및

nvidia-smi 통신 실패

상태가 유지됨.

따라서 단순한 일시적인 GPU 전원 상태 문제는 아니었음.


6. Linux 커널 문제 의심

당시 사용 중이던 환경:

Ubuntu 22.04.5
Linux Kernel 6.8.0-124
GeForce MX250
NVIDIA Driver 580

MX250과 Linux 6.x 계열에서 PCIe 전원 관리 문제로
rev ff / fallen off the bus 증상이 발생한 유사 사례가 있었음.

따라서 NVIDIA 드라이버를 먼저 바꾸기보다는,
Ubuntu 22.04의 기본 GA 계열인 Linux 5.15에서 비교 테스트하기로 함.


7. Linux 5.15 커널 추가 설치

기존 6.8 커널은 삭제하지 않고 테스트용으로 5.15를 추가 설치함.

sudo apt install linux-generic-5.15

설치된 커널:

Linux 5.15.0-191-generic

기존 커널은 그대로 유지했기 때문에 문제가 발생하면 언제든 6.8로 다시 부팅할 수 있는 상태였음.


8. Linux 5.15 최초 테스트

GRUB에서 직접

Ubuntu, with Linux 5.15.0-191-generic

을 선택하여 부팅함.

uname -r

결과:

5.15.0-191-generic

이후 MX250 확인:

lspci -nnk -s 02:00.0

결과가 이전과 달라짐.

이전 Linux 6.8:
GeForce MX250 (rev ff)

Linux 5.15:
GeForce MX250 (rev a1)

즉, Linux 5.15에서는 GPU가 PCIe 상에서 정상적으로 응답하기 시작함.

이 결과 때문에 GPU 하드웨어 자체가 고장난 것보다는
Linux 6.8 환경에서 MX250의 PCIe/전원 관리 호환 문제가 발생했을 가능성이 높다고 판단함.


9. 5.15에서 nvidia-smi가 아직 안 된 이유

GPU는 rev a1로 정상 인식됐지만 처음에는

nvidia-smi

가 여전히 실패함.

lspci 결과를 보면:

Kernel modules: nvidiafb, nouveau

만 표시되고

Kernel driver in use: nvidia

가 없었음.

원인은 Linux 5.15를 새로 추가했지만,
5.15용 NVIDIA 580 커널 모듈이 아직 설치되지 않았기 때문이었음.


10. Linux 5.15용 NVIDIA 모듈 설치

다음 패키지를 설치함.

linux-modules-nvidia-580-5.15.0-191-generic

설치 과정에서 5.15용 NVIDIA 모듈들이 정상적으로 생성됨.

- nvidia.ko
- nvidia-drm.ko
- nvidia-modeset.ko
- nvidia-uvm.ko
- nvidia-peermem.ko

모두 OK로 설치됨.

NVIDIA 드라이버도
580.159.03 → 580.178.04
버전으로 업데이트됨.


11. 최종 결과

Linux 5.15.0-191로 다시 부팅 후 확인.

uname -r

결과:

5.15.0-191-generic


lspci -nnk -s 02:00.0

결과:

GeForce MX250 (rev a1)

Kernel driver in use: nvidia

Kernel modules:
- nvidiafb
- nouveau
- nvidia_drm
- nvidia


lsmod | grep nvidia

결과:

- nvidia_uvm
- nvidia_drm
- nvidia_modeset
- nvidia

모두 정상 로딩.


nvidia-smi

결과 정상 출력.

GPU:
NVIDIA GeForce MX250

VRAM:
4096 MiB

Driver Version:
580.178.04

CUDA Version 표시:
13.0

GPU 온도:
약 42°C

따라서 NVIDIA GPU와 드라이버 간 통신이 최종적으로 정상화됨.


12. 문제 원인 정리

처음에는 NVIDIA 드라이버 문제처럼 보였지만,
실제로는 Linux 6.8.0-124 환경에서 MX250이 PCIe 버스에서 정상적으로 응답하지 못하는 문제가 핵심이었음.

Linux 6.8에서는:

MX250
→ rev ff
→ PCIe 응답 실패
→ NVIDIA probe 실패
→ "GPU has fallen off the bus"
→ nvidia-smi 실패

Linux 5.15에서는:

MX250
→ rev a1
→ PCIe 정상 인식
→ NVIDIA 580 커널 모듈 설치
→ Kernel driver in use: nvidia
→ nvidia-smi 정상

따라서 현재까지의 테스트 결과를 기준으로 보면
GPU 하드웨어 고장보다는
HP Envy + MX250 + Linux 6.8 환경의 PCIe/전원관리 호환 문제일 가능성이 높음.


13. Secure Boot

Secure Boot는 현재 Enabled 상태임.

하지만 Ubuntu에서 제공하는 서명된 NVIDIA 커널 모듈을 사용하고 있고,
최종적으로 nvidia 모듈도 정상 로드됐기 때문에
이번 문제의 직접적인 원인은 Secure Boot가 아니었던 것으로 판단됨.


14. 현재 설치된 커널

현재 여러 커널이 동시에 설치되어 있음.

- 6.8.0-138
- 6.8.0-124
- 6.8.0-110
- 5.15.0-191

6.8.0-138은 NVIDIA 패키지를 설치하는 과정에서
Ubuntu HWE 관련 의존성으로 추가 설치됨.

현재 GPU가 정상 작동하는 것이 확인된 커널은:

5.15.0-191-generic

따라서 당분간 종합설계 프로젝트의 GPU/CUDA 환경은
5.15.0-191을 기준으로 사용하는 것이 안전함.

기존 6.8 커널들은 아직 삭제하지 않고 유지하는 것이 좋음.


15. 현재 최종 상태

Ubuntu 22.04.5 LTS
Linux 5.15.0-191-generic

Intel UHD Graphics 620:
정상

NVIDIA GeForce MX250:
정상

PCIe:
rev a1 정상

NVIDIA Driver:
580.178.04 정상

NVIDIA Kernel Module:
정상

nvidia-smi:
정상

VRAM:
4GB

CUDA 사용 가능 상태:
O

따라서 현재 NVIDIA GPU 세팅은 정상적으로 완료된 상태임.