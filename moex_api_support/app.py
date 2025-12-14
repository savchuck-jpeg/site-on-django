import asyncio
from datetime import date, timedelta
from flask import Flask, jsonify, request
from moex import AsyncMoex

app = Flask(__name__)

async def fetch_ticker_data(ticker):
    async with AsyncMoex() as amoex:
        template_id = None
        for tmpl in amoex.find_template("/candles"):
            template_id = tmpl.id
            break
        if template_id is None:
            return None

        end_date = date.today()
        start_date = end_date - timedelta(days=40)

        params = {
            "from": start_date.isoformat(),
            "till": (end_date + timedelta(days=1)).isoformat(),
            "interval": "24",
            "start": 0,
            "limit": 500,
        }

        url = amoex.render_url(template_id, engine="stock", market="shares", security=ticker)
        rows = await amoex.execute(url=url, **params)
        df = rows.to_df() if rows else None

        if df is not None and not df.empty:
            return df
        return None


def run_async(coro):
    return asyncio.run(coro)


@app.route('/get_ticker_data_28', methods=['GET'])
def get_ticker_data_28():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Тикер не указан"}), 400

    df = run_async(fetch_ticker_data(ticker)).tail(28)
    if df is None or df.empty:
        return jsonify({"error": "Нет данных для указанного тикера"}), 404

    response_data = df.to_dict(orient='records')
    return jsonify(response_data)

@app.route('/get_ticker_data', methods=['GET'])
def get_ticker_data():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Тикер не указан"}), 400

    df = run_async(fetch_ticker_data(ticker)).tail(1)
    if df is None or df.empty:
        return jsonify({"error": "Нет данных для указанного тикера"}), 404

    response_data = df.to_dict(orient='records')
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) 