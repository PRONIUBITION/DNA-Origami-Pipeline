import json

with open("datapredict/outputs/hybrid_evaluation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("""Total samples:""", data["summary"]["total_images"])
print("""4-point sample:""", data["summary"]["four_dot_samples"])
print("""Correct:""", data["summary"]["correct"])
print("""Accuracy:""", f"{data['summary']['accuracy'] * 100:.1f}%")
print()

true_dist = {}
for r in data["results"]:
    t = r["true"]
    true_dist[t] = true_dist.get(t, 0) + 1
print("""True label distribution:""", true_dist)

pred_dist = {}
for r in data["results"]:
    p = r["pred"]
    pred_dist[p] = pred_dist.get(p, 0) + 1
print("""Predicted label distribution:""", pred_dist)

print()
print("""Abnormal ratio samples:""")
for r in data["results"]:
    if r["ratio"] > 1000:
        print(
            f"  {r['image']}: ratio={r['ratio']:.1f}, true={r['true']}, pred={r['pred']}, method={r['method']}"
        )
