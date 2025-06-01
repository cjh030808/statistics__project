import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.0f}'.format #공학적 표현 비활성화

df = pd.read_csv(r"C:\Users\55\Desktop\코딩\pythonWorkspace\확통 과제\korea_rental_housing.csv", encoding="cp949")
#다운로드 한 데이터를 절대경로를 이용해 가져옴

df_seoul = df[df['광역시도'] == '서울특별시']      #전국의 지역 중 서울특별시
df_monthly_rent = df_seoul['월임대료'].describe()  # 서울 임대료 통계량 보기

print(df_seoul[['광역시도', '월임대료']])       # 서울 지역과 임대료만 출력
# print(df_monthly_rent)


#---------2번---------
plt.figure(figsize=(10, 6))
df_seoul['월임대료'].dropna().astype(float).hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Monthly rent in Seoul')
plt.xlabel('Monthly rent')
plt.ylabel('Number of buildings')
plt.ticklabel_format(style='plain', axis='x')     #공학적 표현 안 하고 강제적으로 정수형으로 표현현
plt.grid(True)
plt.show()