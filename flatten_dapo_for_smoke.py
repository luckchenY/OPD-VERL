"""
从 dapo-math-17k-processed.parquet 派生一个 smoke_test_process_reward.py 能直接吃的版本:
  - prompt:        从消息列表里抽 user content, 变成纯字符串
  - ground_truth:  从 reward_model 字典里抽出来, 提到顶层
  - data_source:   原样保留
不动原文件(verl 训练还需要消息列表格式)。
"""
import pandas as pd

SRC = "/data/chenyang/OPD/datasets/dapo-math-17k-processed.parquet"
DST = "/data/chenyang/OPD/datasets/dapo-math-17k-processed-flat.parquet"


def _extract_prompt_text(prompt_field) -> str:
    """prompt 字段是 numpy array/list of dict, 取第一条 user 消息的 content。
    兜底: 已经是字符串就直接返回。"""
    import numpy as np
    if isinstance(prompt_field, str):
        return prompt_field
    # numpy array / list / tuple
    if isinstance(prompt_field, (list, tuple, np.ndarray)):
        if len(prompt_field) == 0:
            return ""
        first = prompt_field[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(prompt_field, dict):
        return str(prompt_field.get("content", ""))
    return str(prompt_field)


def _extract_gt(reward_model_field) -> str:
    if isinstance(reward_model_field, dict):
        return str(reward_model_field.get("ground_truth", ""))
    return str(reward_model_field)


def main():
    df = pd.read_parquet(SRC)
    print(f"[load] {SRC}  rows={len(df)}  cols={list(df.columns)}")

    out = pd.DataFrame({
        "prompt":      df["prompt"].apply(_extract_prompt_text),
        "ground_truth": df["reward_model"].apply(_extract_gt),
        "data_source": df["data_source"],
    })

    # 校验: 没有空值
    n_empty_prompt = (out["prompt"].str.len() == 0).sum()
    n_empty_gt = (out["ground_truth"].str.len() == 0).sum()
    print(f"[check] empty prompt: {n_empty_prompt}  empty ground_truth: {n_empty_gt}")
    assert n_empty_prompt == 0 and n_empty_gt == 0, "存在空值, 中止"

    out.to_parquet(DST, index=False)
    print(f"[write] {DST}  rows={len(out)}  cols={list(out.columns)}")
    print("[sample] row 0:")
    print("  prompt[:200]:", out.iloc[0]["prompt"][:200])
    print("  ground_truth :", repr(out.iloc[0]["ground_truth"]))
    print("  data_source  :", out.iloc[0]["data_source"])


if __name__ == "__main__":
    main()
