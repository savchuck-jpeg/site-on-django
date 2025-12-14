from celery import shared_task
import requests
from .models import Ticker, RealCandle, PredictedCandle
from django.utils import timezone


def update_ticker_real_data():
    tickers = Ticker.objects.all()
    for ticker in tickers:
        try:
            response = requests.get(f'http://flask_moex_api:5000/get_ticker_data?ticker={ticker.slug}')
            print(f'Обновления данных для тикера {ticker.slug}')
            if response.status_code == 200:
                data = response.json()[0] 
                RealCandle.objects.create(
                    ticker=ticker,
                    open=data['open'],
                    close=data['close'],
                    high=data['high'],
                    low=data['low'],
                    volume=data['volume'],
                    value=data.get('value', 0),
                    begin=timezone.now(),
                    end=timezone.now()
                )
        except Exception as e:
            print(f"Ошибка при обновлении тикера {ticker.slug}: {e}")

def update_ticker_predict_data():
    tickers = Ticker.objects.all()
    for ticker in tickers:
        try:
            response = requests.get(f'http://moex_model:5000/predict?ticker={ticker.slug}')
            print(f'Обновления предсказаний для тикера {ticker.slug}')
            if response.status_code == 200:
                data = response.json()
                PredictedCandle.objects.create(
                    ticker=ticker,
                    open=data['open'],
                    close=data['close'],
                    high=data['high'],
                    low=data['low'],
                    volume=data['volume'],
                    value=data.get('value', 0),
                    begin=timezone.now(),
                    end=timezone.now()
                )
        except Exception as e:
            print(f"Ошибка при обновлении тикера {ticker.slug}: {e}")

@shared_task
def update_ticker_data():
    print(f'{'-'*17}Начало обновления')
    update_ticker_predict_data()
    update_ticker_real_data()
    print(f'{'-'*16}\nКонец обновления')