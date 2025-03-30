import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 읽기
df = pd.read_csv('test_results.csv')

# 버전별 실행 시간 비교 막대 그래프
plt.figure(figsize=(12, 6))
m_value = 1000
subset = df[df['M'] == m_value]
subset.plot(x='Threads', y=['Serial(ms)', 'Version1(ms)', 'Version2(ms)', 'Version3(ms)'], 
            kind='bar', width=0.8)
plt.title(f'Execution Time by Threads (M={m_value})')
plt.ylabel('Execution Time (ms)')
plt.xlabel('Number of Threads')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f'execution_time_m_{m_value}.png')  

# # 히트맵
# plt.figure(figsize=(8, 6))
# heatmap_data = pd.pivot_table(df, values="Version3(ms)", index="M", columns="Threads")
# sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
# plt.title('Execution Time Heatmap (Version 3)')
# plt.savefig('execution_time_heatmap.png')  # show() 대신 저장
