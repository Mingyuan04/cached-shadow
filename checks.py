from src.prep import load_from_config, match_table

df = load_from_config("data/df2011_config.json")
m = match_table(df)
print("subjects:", df["subject"].nunique(), "sessions:", df["session"].nunique())

per_match = m.drop_duplicates(["session", "match"])
print(per_match.groupby("delta")["L"].agg(["mean", "max"]).round(2))

first = m[m["match_index"] == 1]
print((first.groupby(["delta", "R"])["c1"].mean() * 100).round(2))
print((m.groupby(["delta", "R"])["c1"].mean() * 100).round(2))
