import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'interface.settings')

app = Celery('interface')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()

app.conf.beat_schedule = {
    'update-ticker-data-nightly': {
        'task': 'index.tasks.update_ticker_data',
        'schedule': crontab(hour=2, minute=00),
    },
}
