import requests
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
url = "https://www.sherdog.com/stats/fightfinder?SearchTxt=Khamzat+Chimaev"

print("📡 Sherdog'un arka kapısına sızılıyor...")
res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")

print(f"🔥 Dönen Sayfa Başlığı: {soup.title.text.strip() if soup.title else 'Yok'}")

# Sherdog arama sonuçlarından dövüşçünün profiline dalıyoruz
for a in soup.select('a[href^="/fighter/"]'):
    if "Khamzat" in a.text:
        profile_url = "https://www.sherdog.com" + a['href']
        print(f"🔗 Profil bulundu: {profile_url}")

        p_res = requests.get(profile_url, headers=headers)
        p_soup = BeautifulSoup(p_res.text, "html.parser")

        # W-L-D Recordlarını kazıyoruz
        win = p_soup.select_one('.win span:nth-of-type(2)')
        loss = p_soup.select_one('.loss span:nth-of-type(2)')
        draw = p_soup.select_one('.draw span:nth-of-type(2)')

        w = win.text if win else "0"
        l = loss.text if loss else "0"
        d = draw.text if draw else "0"

        print(f"🎯 BINGO! Sherdog'dan Koparılan Record: {w}-{l}-{d}")
        break