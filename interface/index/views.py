from django.shortcuts import render
from django.views.generic import DetailView
from .models import Ticker


def main_view(request):
    context = {
        'tickers': Ticker.objects.all(),
    }
    return render(request, 'index/index.html', context=context)

class TickerDetailView(DetailView):
    model = Ticker
    context_object_name = 'ticker'
    template_name = 'index/detail.html'

class TickerArchiveDetailView(DetailView):
    model = Ticker
    context_object_name = 'ticker'
    template_name = 'index/archive.html'