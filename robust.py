from src.prep import load_from_config, match_table
from src.signature import run

m = match_table(load_from_config("data/df2011_config.json"))
print(run(m[m["delta"] == 0.75]))      # 只看 δ = 3/4：长度变化更大
half = m["match_index"] <= m.groupby("subject")["match_index"].transform("max") / 2
print(run(m[half]))                      # 只看每人前一半的局：学习阶段
