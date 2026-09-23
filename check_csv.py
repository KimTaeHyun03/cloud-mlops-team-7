import pandas as pd

PATH = r"D:\ysu_26_2\MLOPS\top_rated_movies_utf8.csv"
df = pd.read_csv(PATH, encoding="utf-8-sig")

print("행, 열:", df.shape)
print(df.dtypes)
print("\n결측:\n", df.isna().sum())
print("\nid 중복:", df["id"].duplicated().sum())
d = pd.to_datetime(df["release_date"], errors="coerce")
print("개봉일 범위:", d.min(), "~", d.max(), "/ 변환 실패:", d.isna().sum())
print(df[["popularity", "vote_average", "vote_count"]].describe())
print("\nadult 값:", df["adult"].value_counts().to_dict())
