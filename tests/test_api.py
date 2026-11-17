"""API 端到端测试：会话创建、校验、判断/参数版本、异步求解与迟到结果作废。"""

import io
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

client = TestClient(app)


def png_bytes(color=(200, 30, 30)):
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_session(n_imgs=3, dets=None):
    if dets is None:
        dets = [
            {"id": "a", "frame": 0, "rect": [1, 1, 8, 8], "vector": [1, 0]},
            {"id": "b", "frame": 1, "rect": [11, 1, 8, 8], "vector": [1, 0]},
        ]
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(n_imgs)]
    files.append(("detections", ("det.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError("求解任务超时未完成")


# ---------------------------------------------------------------- 会话与校验

def test_create_session_ok():
    p = make_session()
    assert p["n_frames"] == 3 and len(p["detections"]) == 2
    assert p["version"] == 1
    img = client.get(f"/api/sessions/{p['session_id']}/frames/0")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"


def test_too_many_images_rejected():
    files = [("images", (f"f{i}.png", png_bytes(), "image/png")) for i in range(11)]
    files.append(("detections", ("d.json", json.dumps([]), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "10" in r.json()["detail"]


def test_too_many_detections_rejected():
    dets = [{"id": f"b{i}", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1]} for i in range(31)]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "30" in r.json()["detail"]


@pytest.mark.parametrize("det,needle", [
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [0, 0]}, "非零"),
    ({"id": "a", "frame": 5, "rect": [0, 0, 4, 4], "vector": [1]}, "帧号"),
    ({"id": "a", "frame": 0, "rect": [0, 0, -4, 4], "vector": [1]}, "宽"),
    ({"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, float("nan")]}, "非法"),
])
def test_bad_detection_rejected(det, needle):
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps([det]), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and needle in r.json()["detail"]


def test_vector_dim_mismatch_rejected():
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 4, 4], "vector": [1, 0]},
        {"id": "b", "frame": 0, "rect": [8, 0, 4, 4], "vector": [1, 0, 0]},
    ]
    files = [("images", ("f0.png", png_bytes(), "image/png")),
             ("detections", ("d.json", json.dumps(dets), "application/json"))]
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 400 and "维度" in r.json()["detail"]


# ---------------------------------------------------------------- 示例与求解

def test_example_solves_to_two_animals():
    p = client.post("/api/example").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 11
    job = client.post(f"/api/sessions/{p['session_id']}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done"
    res = done["result"]
    assert res["feasible"] and res["animal_count"] == 2 and res["n_trajectories"] == 2
    assert res["total_cost"] == pytest.approx(res["assoc_cost"] + res["penalty_cost"])


def test_conflicting_judgments_report_infeasible_without_count():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # A0 与 B0 同处第 0 帧，标记同只 → 无解
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "A0", "b": "B0", "type": "same"}]})
    assert r.status_code == 200
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    res = done["result"]
    assert res["feasible"] is False
    assert "animal_count" not in res
    assert "无解" in res["message"]


def test_same_judgment_changes_result():
    p = client.post("/api/example").json()
    sid = p["session_id"]
    # 把 B0 与 A1 判为同只（它们帧差 1），全局解必须让二者同轨
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "B0", "b": "A1", "type": "same"}]})
    done = wait_job(client.post(f"/api/sessions/{sid}/solve").json()["job_id"])
    res = done["result"]
    assert res["feasible"]
    for t in res["trajectories"]:
        ids = {b["id"] for b in t["boxes"]}
        assert not ({"B0", "A1"} <= ids) or True  # 占位，下面统一检查
    same_traj = [t for t in res["trajectories"]
                 if {"B0", "A1"} <= {b["id"] for b in t["boxes"]}]
    assert len(same_traj) == 1


# ---------------------------------------------------------------- 版本与迟到结果

def test_version_bumps_and_stale_job_detectable():
    p = make_session()
    sid = p["session_id"]
    assert p["version"] == 1

    r = client.put(f"/api/sessions/{sid}/params",
                   json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    assert r.status_code == 200 and r.json()["version"] == 2

    job = client.post(f"/api/sessions/{sid}/solve").json()
    assert job["version"] == 2  # 任务记录启动时版本

    # 求解期间继续修改 → 版本再 +1，旧任务结果即为“迟到结果”
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "b", "type": "different"}]})
    assert r.json()["version"] == 3

    done = wait_job(job["job_id"])
    assert done["version"] == 2  # 结果仍携带旧版本号
    current = client.get(f"/api/sessions/{sid}").json()
    assert current["version"] == 3
    assert done["version"] != current["version"]  # 前端据此丢弃，不覆盖新输入


def test_params_validation():
    p = make_session()
    r = client.put(f"/api/sessions/{p['session_id']}/params",
                   json={"w_dist": -1, "w_app": 1, "penalty": 1})
    assert r.status_code == 422


# ---------------------------------------------------------------- 连续改判与求解交接

def make_chain_session():
    """三框一条链：a(f0)-b(f1)-c(f2)，外观相同、位置接近。
    无判断时全局最优为 1 条轨迹；a、c 判异只后必须拆成 2 条。"""
    dets = [
        {"id": "a", "frame": 0, "rect": [0, 0, 8, 8], "vector": [1, 0]},
        {"id": "b", "frame": 1, "rect": [10, 0, 8, 8], "vector": [1, 0]},
        {"id": "c", "frame": 2, "rect": [20, 0, 8, 8], "vector": [1, 0]},
    ]
    return make_session(n_imgs=3, dets=dets)


def solve_count(sid):
    job = client.post(f"/api/sessions/{sid}/solve").json()
    done = wait_job(job["job_id"])
    assert done["status"] == "done" and done["result"]["feasible"]
    return done["result"]["animal_count"]


def test_rapid_rejudgment_last_save_wins():
    """快速对同一对框先标同只再标异只（按用户操作顺序依次保存）：
    服务端以最后一次成功保存为准，下一次求解随之变化。"""
    p = make_chain_session()
    sid = p["session_id"]
    assert solve_count(sid) == 1  # 无判断：一条链

    r1 = client.put(f"/api/sessions/{sid}/judgments",
                    json={"judgments": [{"a": "a", "b": "c", "type": "same"}]})
    assert r1.status_code == 200
    r2 = client.put(f"/api/sessions/{sid}/judgments",
                    json={"judgments": [{"a": "a", "b": "c", "type": "different"}]})
    assert r2.status_code == 200
    assert r2.json()["version"] == r1.json()["version"] + 1
    cur = client.get(f"/api/sessions/{sid}").json()
    assert cur["judgments"] == [{"a": "a", "b": "c", "type": "different"}]
    assert solve_count(sid) == 2  # 异只判断生效：链被拆开

    # 再改回同只：最后一次保存再次生效，计数恢复 1
    client.put(f"/api/sessions/{sid}/judgments",
               json={"judgments": [{"a": "a", "b": "c", "type": "same"}]})
    assert solve_count(sid) == 1


def test_solve_started_before_judgment_save_goes_stale():
    """求解启动后、判断保存完成前：旧任务结果携带旧版本号（页面不得展示）；
    下一次求解以最后保存的判断为准。"""
    p = make_chain_session()
    sid = p["session_id"]
    job1 = client.post(f"/api/sessions/{sid}/solve").json()
    assert job1["version"] == 1
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [{"a": "a", "b": "c", "type": "different"}]})
    assert r.json()["version"] == 2

    done1 = wait_job(job1["job_id"])
    assert done1["version"] == 1  # 旧任务基于保存前的输入
    assert done1["result"]["animal_count"] == 1  # 旧计数，页面须作废不展示
    cur = client.get(f"/api/sessions/{sid}").json()
    assert cur["version"] == 2 and cur["judgments"][0]["type"] == "different"

    job2 = client.post(f"/api/sessions/{sid}/solve").json()
    assert job2["version"] == 2  # 下一次求解基于最后保存的判断
    done2 = wait_job(job2["job_id"])
    assert done2["result"]["animal_count"] == 2


def test_rapid_param_changes_last_save_wins():
    """快速连续修改参数：版本单调递增，服务端以最后一次保存为准。"""
    p = make_session()
    sid = p["session_id"]
    r1 = client.put(f"/api/sessions/{sid}/params",
                    json={"w_dist": 2.0, "w_app": 50.0, "penalty": 100.0})
    r2 = client.put(f"/api/sessions/{sid}/params",
                    json={"w_dist": 3.0, "w_app": 60.0, "penalty": 200.0})
    assert (r1.json()["version"], r2.json()["version"]) == (2, 3)
    cur = client.get(f"/api/sessions/{sid}").json()
    assert cur["version"] == 3
    assert cur["params"] == {"w_dist": 3.0, "w_app": 60.0, "penalty": 200.0}


def test_failed_judgment_save_leaves_server_state_unchanged():
    """保存失败（引用不存在的框）：不推进版本、不改变已存判断，
    页面失败后可据此重新同步到与服务端一致的状态。"""
    p = make_session()
    sid = p["session_id"]
    ok = client.put(f"/api/sessions/{sid}/judgments",
                    json={"judgments": [{"a": "a", "b": "b", "type": "same"}]})
    assert ok.status_code == 200
    v = ok.json()["version"]
    bad = client.put(f"/api/sessions/{sid}/judgments",
                     json={"judgments": [{"a": "a", "b": "ghost", "type": "different"}]})
    assert bad.status_code == 400
    cur = client.get(f"/api/sessions/{sid}").json()
    assert cur["version"] == v
    assert cur["judgments"] == [{"a": "a", "b": "b", "type": "same"}]


def test_judgments_response_reflects_stored_state():
    """PUT 响应返回服务端实际存储的判断：同一对框重复出现时以后到者为准。"""
    p = make_session()
    sid = p["session_id"]
    r = client.put(f"/api/sessions/{sid}/judgments",
                   json={"judgments": [
                       {"a": "a", "b": "b", "type": "same"},
                       {"a": "b", "b": "a", "type": "different"},
                   ]})
    assert r.status_code == 200
    assert r.json()["judgments"] == [{"a": "a", "b": "b", "type": "different"}]
    cur = client.get(f"/api/sessions/{sid}").json()
    assert cur["judgments"] == r.json()["judgments"]


def test_unknown_session_404():
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope/solve").status_code == 404


def test_healthz():
    assert client.get("/healthz").json() == {"ok": True}
