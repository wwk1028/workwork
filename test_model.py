"""
模型推理 & API 认证测试脚本

用法:
    python test_model.py                    # 测试翻译模型
    python test_model.py --api              # 测试 API 认证（需先启动后端）
    python test_model.py --all              # 全部测试
"""
import sys
import os
import argparse
import requests
import json
import time

# ---- 配置 ----
API_BASE = "http://localhost:8000/api/v1"


# ============================================================
# 1. 本地模型推理测试
# ============================================================

def test_model_local():
    """直接用模型做中->英 & 英->中各一句翻译"""
    print("=" * 50)
    print(" 本地模型推理测试")
    print("=" * 50)

    # 设置 HuggingFace 镜像（国内网络）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    # 测试 zh->en 模型
    print("\n[1/2] 加载 zh->en 模型 ...")
    zh2en = MarianMTModel.from_pretrained("opus_finetuned_full").to("cuda" if torch.cuda.is_available() else "cpu")
    zh2en_tok = MarianTokenizer.from_pretrained("opus_finetuned_full")
    zh2en.eval()

    # 测试 en->zh 模型
    print("[2/2] 加载 en->zh 模型 ...")
    en2zh = MarianMTModel.from_pretrained("opus_en2zh").to("cuda" if torch.cuda.is_available() else "cpu")
    en2zh_tok = MarianTokenizer.from_pretrained("opus_en2zh")
    en2zh.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}\n")

    # ---- 中 -> 英 ----
    zh_text = "人工智能正在深刻改变人类社会的方方面面"
    print(f"[中->英] 输入: {zh_text}")

    inputs = zh2en_tok(zh_text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = zh2en.generate(**inputs, num_beams=5, max_length=128)
    latency = (time.perf_counter() - t0) * 1000
    result = zh2en_tok.decode(outputs[0], skip_special_tokens=True)
    print(f"[中->英] 输出: {result}")
    print(f"[中->英] 延迟: {latency:.1f} ms")

    # ---- 英 -> 中 ----
    en_text = "Machine translation has made remarkable progress in recent years"
    print(f"\n[英->中] 输入: {en_text}")

    inputs = en2zh_tok(en_text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = en2zh.generate(**inputs, num_beams=5, max_length=128)
    latency = (time.perf_counter() - t0) * 1000
    result = en2zh_tok.decode(outputs[0], skip_special_tokens=True)
    print(f"[英->中] 输出: {result}")
    print(f"[英->中] 延迟: {latency:.1f} ms")

    print("\n本地模型测试完成!")


# ============================================================
# 2. API 认证测试
# ============================================================

def test_api_auth():
    """测试 API 的认证相关接口：
    - 注册 / 登录 / 无Token访问 / 带Token访问
    """
    print("=" * 50)
    print(" API 认证测试")
    print("=" * 50)

    # 用时间戳避免用户名冲突
    username = f"testuser_{int(time.time()) % 100000}"
    password = "test123456"

    # ---- 2.1 注册 ----
    print(f"\n[2.1] 注册新用户: {username}")
    resp = requests.post(f"{API_BASE}/auth/register", json={
        "username": username,
        "password": password,
        "email": f"{username}@test.com"
    })
    print(f"  状态码: {resp.status_code}")
    print(f"  响应: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")

    # ---- 2.2 登录 ----
    print(f"\n[2.2] 用户登录")
    resp = requests.post(f"{API_BASE}/auth/login", json={
        "username": username,
        "password": password
    })
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    token = data.get("access_token", "")
    print(f"  获取到 Token: {token[:40]}..." if token else "  未获取到 Token")

    # ---- 2.3 无 Token 访问受保护接口 ----
    print(f"\n[2.3] 无 Token 访问 /translate/history")
    resp = requests.get(f"{API_BASE}/translate/history")
    print(f"  状态码: {resp.status_code}")
    print(f"  响应: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")

    # ---- 2.4 带 Token 访问受保护接口 ----
    print(f"\n[2.4] 带 Token 访问 /translate/history")
    resp = requests.get(
        f"{API_BASE}/translate/history",
        headers={"Authorization": f"Bearer {token}"}
    )
    print(f"  状态码: {resp.status_code}")
    print(f"  响应: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")

    # ---- 2.5 带 Token 做一次翻译 ----
    if token:
        print(f"\n[2.5] 带 Token 翻译测试")
        resp = requests.post(
            f"{API_BASE}/translate",
            json={"text": "Hello world", "direction": "zh"},
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"  状态码: {resp.status_code}")
        print(f"  响应: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")

    print("\nAPI 认证测试完成!")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型翻译 & API 认证测试")
    parser.add_argument("--api", action="store_true", help="测试 API 认证")
    parser.add_argument("--all", action="store_true", help="测试全部")
    args = parser.parse_args()

    if args.all:
        test_model_local()
        print("\n")
        test_api_auth()
    elif args.api:
        test_api_auth()
    else:
        # 默认只做本地模型测试
        test_model_local()
