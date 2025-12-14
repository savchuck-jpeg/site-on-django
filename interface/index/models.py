from django.db import models


class Ticker(models.Model):
    title = models.CharField(max_length=100)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.title} ({self.code})"
    def code(self):
        return self.slug
    def get_real_candles(self):
        return self.realcandle_set.order_by('-end')

    def get_last_predicted_candle(self):
        return self.predictedcandle_set.order_by('-end').first()

    def get_last_real_candle(self):
        return self.realcandle_set.order_by('-end').first()

    def is_predicted_up_on_real(self):
        predict_close = self.get_last_predicted_candle()
        real_close = self.get_last_real_candle()
        print(predict_close, real_close)
        if predict_close.close > real_close.close:
            return 'up'
        elif predict_close.close == real_close.close:
            return 'equal'
        else:
            return 'down'


class RealCandle(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.IntegerField()
    value = models.FloatField()
    begin = models.DateTimeField()
    end = models.DateTimeField()

    class Meta:
        ordering = ('-end',)


class PredictedCandle(models.Model):
    ticker = models.ForeignKey(Ticker, on_delete=models.CASCADE)
    open = models.FloatField()
    close = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    volume = models.IntegerField()
    value = models.FloatField()
    begin = models.DateTimeField()
    end = models.DateTimeField()

    class Meta:
        ordering = ('-end',)
