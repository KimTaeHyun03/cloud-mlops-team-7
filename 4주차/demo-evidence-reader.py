"""기존 데모 결과를 읽습니다. 학습이나 실시간 예측을 실행하지 않습니다."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent
def read(name): return json.loads((p/name).read_text(encoding="utf-8"))
r=read("usedcar-baseline-examples.json")[0]
print("중고차 최초 모델의 저장 결과 한 건")
print(r["make_name"],r["model_name"],"실제",r["price"],"예측",round(r["predicted_price"],2))
print("절대오차:",round(abs(r["predicted_price"]-r["price"]),2),"(원본 가격 단위)")
c=read("usedcar-trim-comparison.json")["full_holdout"]
old,new=c["old"]["MAE"],c["new"]["MAE"]
print("같은",c["rows"],"건에서 트림 추가 MAE:",round(old,2),"->",round(new,2))
print("오차 감소율:",round((old-new)/old*100,2),"%; 새 외부 검증 아님")
r=read("subway-route-regression.json")[0]
print("지하철 저장 기록:",r["departure_at"],r["route_strategy"])
print("열차 변경",r["transfer_count"],"회 / 예상",r["estimated_minutes"],"분")
for leg in r["legs"]: print(leg["from_station"],"->",leg["to_station"],leg["line"],"호선",leg["service"])
print("2025-09-30 공개 시간표 적용 결과. 현재 운행·칸별 정확도 검증 아님.")
