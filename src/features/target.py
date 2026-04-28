"""타깃 레이블 생성.

target = 1  if  close(t+5) / close(t) - 1 >= 0.02
target = 0  otherwise

미래 데이터를 사용하므로 반드시 train/test 분리 후에 최종 NaN 제거.
"""
import pandas as pd


def add_target(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """
    5거래일 후 수익률이 threshold 이상이면 1, 아니면 0.
    마지막 horizon개 행은 target = NaN (미래 가격 없음).
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    df["target"] = ((future_close / df["close"] - 1) >= threshold).astype("float32")
    # 마지막 horizon개 행 — 미래 정보 없으므로 NaN 처리
    df.loc[df.index[-horizon:], "target"] = float("nan")
    return df
