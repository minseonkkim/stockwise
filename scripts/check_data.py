"""수집 현황 조회 스크립트."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.database import get_session


def main():
    with get_session() as session:

        # 1. Active stocks
        total_stocks = session.execute(text("SELECT COUNT(*) FROM stocks WHERE is_active = TRUE")).scalar()
        print(f"\n[Stocks Master]")
        print(f"  Active tickers : {total_stocks}")

        # 2. Total price rows
        total_rows = session.execute(text("SELECT COUNT(*) FROM daily_prices")).scalar()
        print(f"\n[Price Data]")
        print(f"  Total rows     : {total_rows:,}")

        # 3. Date range
        date_range = session.execute(text(
            "SELECT MIN(date), MAX(date) FROM daily_prices"
        )).fetchone()
        print(f"  Date range     : {date_range[0]} ~ {date_range[1]}")

        # 4. Unique tickers collected
        ticker_count = session.execute(text(
            "SELECT COUNT(DISTINCT ticker) FROM daily_prices"
        )).scalar()
        print(f"  Tickers stored : {ticker_count}")

        # 5. Recent sample
        print(f"\n[Recent Sample - AAPL / MSFT / GOOGL]")
        rows = session.execute(text("""
            SELECT ticker, date, open, high, low, close, volume
            FROM daily_prices
            WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
            ORDER BY ticker, date DESC
            LIMIT 9
        """)).fetchall()
        print(f"  {'ticker':<8} {'date':<12} {'open':>8} {'high':>8} {'low':>8} {'close':>8} {'volume':>14}")
        print(f"  {'-'*68}")
        for r in rows:
            print(f"  {r.ticker:<8} {str(r.date):<12} {r.open:>8.2f} {r.high:>8.2f} {r.low:>8.2f} {r.close:>8.2f} {r.volume:>14,}")

        # 6. Collection log summary
        print(f"\n[Collection Log]")
        log_stats = session.execute(text("""
            SELECT status, COUNT(*) as cnt
            FROM data_collection_logs
            GROUP BY status
            ORDER BY status
        """)).fetchall()
        for r in log_stats:
            print(f"  {r.status:<10}: {r.cnt}")

        # 7. Failed tickers
        failed = session.execute(text("""
            SELECT ticker, error_message
            FROM data_collection_logs
            WHERE status = 'failed'
            ORDER BY ticker
            LIMIT 10
        """)).fetchall()
        if failed:
            print(f"\n[Failed tickers (top 10)]")
            for r in failed:
                print(f"  {r.ticker}: {r.error_message[:60]}")

    print()


if __name__ == "__main__":
    main()
