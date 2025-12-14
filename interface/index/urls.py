from django.urls import path
from .views import main_view
from .views import TickerDetailView, TickerArchiveDetailView

app_name = 'index'

urlpatterns = [
    path('', main_view, name='index'),
    path('d/<str:slug>/', TickerDetailView.as_view(), name='detail'),
    path('a/<str:slug>/', TickerArchiveDetailView.as_view(), name='archive'),
]