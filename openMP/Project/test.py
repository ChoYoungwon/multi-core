import os

size_in_bytes = os.path.getsize("la.mlc")
num_elements = size_in_bytes // 8  # complex64는 8 bytes

print("파일 전체 크기 (bytes):", size_in_bytes)
print("complex64 기준 요소 수:", num_elements)