# FLO Müşteri Segmentasyonu

# İş Problemi

# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri
# belirlemekistiyor. Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere
# göre gruplar oluşturulacak.

# Veri Seti Hikayesi
# Veri seti Flo’dan son alışverişlerini 2020 -2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarındaneldeedilenbilgilerdenoluşmaktadır.

# master_id : Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı(Android, ios, Desktop, Mobile)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

# Ilk olarak kutuphaneleri import edelim

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_ = pd.read_csv('C:/Users/pc/PycharmProjects/pythonProject2/3.Hafta/flo_data_20k.csv')
# dosyanin bir yedegini alalim
df = df_.copy()

# degiskenlere ait ilk 10 gozleme bakalim
df.head(10)

# kac degisken ve kac gozlem var
df.shape

# degiskenlerin isimleri nelerdir
df.columns

# betimsel istatistik degerleri

df.describe().T

# hangi degiskende ne kadar eksik deger var
df.isnull().sum()

# degisken tiplerini inceleyelim
df.dtypes

# Offline ve online musterilerin toplam alisveris sayisi ve harcamasi
df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_online'] + df['customer_value_total_ever_offline']
df.head()

# degisken tiplerine bakalim , tarih ifade eden degiskenlerin tipini date ye cevirelim


df.dtypes

df['first_order_date'] = df['first_order_date'].astype('datetime64[ns]')
df['last_order_date'] = df['last_o`rder_date'].astype('datetime64[ns]')
df['last_order_date_online'] = df['last_order_date_online'].astype('datetime64[ns]')
df['last_order_date_offline'] = df['last_order_date_offline'].astype('datetime64[ns]')
df.dtypes

# Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakalim.

df.groupby('order_channel').agg({'order_num_total': 'count',
                                 'customer_value_total': "mean"}).head()

# En fazla kazancı getiren ilk 10 müşteriyi sıralayalim.

df.groupby('master_id').agg({'customer_value_total': 'sum'}).sort_values(by='customer_value_total',
                                                                         ascending=False).head(10)

# En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.groupby('master_id').agg({'order_num_total': 'sum'}).sort_values(by='order_num_total',
                                                                    ascending=False).head(10)


# Veri ön hazırlık sürecini fonksiyonlaştıralim.

def create_rfm(dataframe):
    # Offline ve online musterilerin toplam alisveris sayisi ve harcamasi
    dataframe['order_num_total'] = dataframe['order_num_total_ever_online'] + dataframe['order_num_total_ever_offline']
    dataframe['customer_value_total'] = dataframe['customer_value_total_ever_online'] + dataframe[
        'customer_value_total_ever_offline']

    # degiskenlerin tipini date'e cevirelim
    dataframe['first_order_date'] = dataframe['first_order_date'].astype('datetime64[ns]')
    dataframe['last_order_date'] = dataframe['last_order_date'].astype('datetime64[ns]')
    dataframe['last_order_date_online'] = dataframe['last_order_date_online'].astype('datetime64[ns]')
    dataframe['last_order_date_offline'] = dataframe['last_order_date_offline'].astype('datetime64[ns]')

    # Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakalim.
    dataframe.groupby('order_channel').agg({'order_num_total': 'count',
                                            'customer_value_total': "mean"}).head()

    # En fazla kazancı getiren ilk 10 müşteriyi sıralayalim.
    dataframe.groupby('master_id').agg({'customer_value_total': 'sum'}).sort_values(by='customer_value_total',
                                                                                    ascending=False).head(10)

    # En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

    dataframe.groupby('master_id').agg({'order_num_total': 'sum'}).sort_values(by='order_num_total',
                                                                               ascending=False).head(10)

    return dataframe


rfm_new = create_rfm(df)
rfm_new

# Analiz yapilan tarihi son alisverisin yapildigi tarihten 2 gun sonrasini yapiyoruz
# Bunun icin ilk once son alisveris tarihini buluyoruz.

df['last_order_date'].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)
df.dtypes

# rfm adinda yeni bir df olusturup, bu df icine rfm degiskenlerini yerlestirelim
rfm = pd.DataFrame()
rfm['master_id'] = df['master_id']
rfm['recency'] = ((today_date - df['last_order_date']).astype('timedelta64[D]'))
rfm['frequency'] = df['order_num_total']
rfm['monetary'] = df['customer_value_total']

rfm.head()

# Recency, Frequencyve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çevirelim.
# Bu skorları recency_score, frequency_scoreve monetary_score olarak kaydedelim.

rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[1, 2, 3, 4, 5])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

# recency_scoreve frequency_score’u tek bir değişken olarak ifade edelim ve RF_SCORE olarak kaydedelim.

rfm['RF_SCORE'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

rfm.head()

# Oluşturulan RF skorları için segment tanımlamaları yapalim ve segmentleri skora cevirelim.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm[["segment", "RF_SCORE"]].groupby("segment").agg(["mean", "count"])

rfm.head()

# Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyelim.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# a.FLO bünyesine yeni bir kadın ayakkab ımarkası dahil ediyor.
# Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde.Bu nedenle markanın tanıtımı ve ürün
# satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçmek isteniliyor.Sadık müşterilerinden(champions,loyal_customers)
# ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydedelim.

first_target_a = (rfm[(rfm["segment"]=="champions") | (rfm["segment"]=="loyal_customers")])
first_target_b = df[(df["interested_in_categories_12"]).str.contains("KADIN")]

first_target_a
first_target_b.head()

ab = pd.merge(first_target_a , first_target_b[["interested_in_categories_12","master_id"]], on=["master_id"])
ab.head()

ab = ab.drop(ab.loc[:,'recency':'interested_in_categories_12'].columns,axis=1)
ab.head

ab.to_csv('first_target_customers_id.csv")


# Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun  süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,uykuda olanlar ve yeni gelen
# müşteriler özel olarak hedef alınmak isteniyor.Uygun profildeki müşterilerin id'lerini csv dosyasına kaydedelim.

second_target_c = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="about_to_sleep") | (rfm["segment"]=="new_customers")]
second_target_d = df[(df["interested_in_categories_12"]).str.contains("ERKEK|COCUK")]
cd = pd.merge(second_target_c, second_target_d[["interested_in_categories_12","master_id"]],on=["master_id"])
cd = cd.drop(cd.loc[:,'recency':'interested_in_categories_12'].columns,axis=1)
cd.head()

cd.to_csv('second_target_customers_id.csv")