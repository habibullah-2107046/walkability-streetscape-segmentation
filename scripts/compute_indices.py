# compute_indices.py
import os
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV  = os.path.join(BASE_DIR, "pixel_statistics", "pixel_counts.csv")
OUT_CSV = os.path.join(BASE_DIR, "indices", "streetscape_indices.csv")

df = pd.read_csv(IN_CSV)

# derive point_id by stripping the "_viewN" suffix, e.g. "50_view0" -> "50"
df["point_id"] = df["image_id"].str.replace(r"_view\d+$", "", regex=True)

# sum all numeric pixel columns across the 6 views belonging to the same point
agg = df.groupby("point_id").sum(numeric_only=True).reset_index()

# sanity check: how many views contributed to each point (should be 6)
view_counts = df.groupby("point_id").size().reset_index(name="num_views")
agg = agg.merge(view_counts, on="point_id")

agg["GVI"] = agg["vegetation"] / agg["total_pixels"]
agg["SVI"] = agg["sky"] / agg["total_pixels"]
agg["EI"]  = (agg["building"] + agg["vegetation"]) / (agg["sidewalk"] + agg["road"] + agg["fence"])
agg["Sidewalk_Visibility"] = agg["sidewalk"] / agg["total_pixels"]

obstacle_cols = ["pole","traffic_light","traffic_sign","car","truck","bus","motorcycle","bicycle","person","rider"]
agg["Crowdedness"] = agg[obstacle_cols].sum(axis=1) / agg["total_pixels"]

final = agg[["point_id","num_views","GVI","SVI","EI","Sidewalk_Visibility","Crowdedness"]]

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
final.to_csv(OUT_CSV, index=False)

print(f"Saved {len(final)} study points to {OUT_CSV}")
print(final)

# flag any point that didn't get exactly 6 views (worth investigating)
bad = final[final["num_views"] != 6]
if len(bad) > 0:
    print("\nWARNING: these points do not have exactly 6 views:")
    print(bad)
else:
    print("\nAll points have exactly 6 views. Good.")