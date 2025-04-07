import numpy as np
import matplotlib.pyplot as plt

# 파일 로딩
filename = "la.mlc"  # 실제 파일명으로 수정하세요
data = np.fromfile(filename, dtype=np.complex64)

# 데이터 reshape
data = data.reshape((5914, 1650))  # height x width

# 복소수 값을 intensity (절댓값)로 변환
intensity = np.abs(data)

# 로그 스케일로 변환 (이미지 대비 개선)
intensity_log = 20 * np.log10(intensity + 1e-6)  # 1e-6은 log(0) 방지용

# 시각화
plt.figure(figsize=(10, 8))
plt.imshow(intensity_log, cmap='gray')
plt.title('SAR Intensity (log scale)')
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.colorbar(label='dB')
plt.tight_layout()
plt.savefig('la')
