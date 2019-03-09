import requests
from bs4 import BeautifulSoup
import warnings

warnings.filterwarnings("ignore")

templates = {
    "roman republic" : "https://www.coinshome.net/search.htm?q=&sortDirection=1&searchType=&pagingPage={}&countryFQ=countryID%3AhGjBwcI0zoYAAAEmV4oNrT2Z&pagingNumberPer=15&",
    "roman empire" : "https://www.coinshome.net/search.htm?q=&sortDirection=1&searchType=&pagingPage={}&countryFQ=countryID%3Ait_BwcI065UAAAEmP0A6TPGB&pagingNumberPer=15&",
    "italy" : "https://www.coinshome.net/search.htm?q=&sortDirection=1&searchType=&pagingPage={}&countryID=showAll&countryFQ=countryID%3AYtt_AAEBlQsAAAEjkuNucewv&pagingNumberPer=15&",
    "greece" : "https://www.coinshome.net/search.htm?q=&sortDirection=1&searchType=&pagingPage={}&countryID=showAll&countryFQ=countryID%3AawkKbzbiw_cAAAFMmEXGHhpg&pagingNumberPer=15&",
    "egypt" : "https://www.coinshome.net/search.htm?q=&sortDirection=1&searchType=&pagingPage={}&countryID=showAll&countryFQ=countryID%3ALRUKbzbixiQAAAFOxJi5liaM&pagingNumberPer=15&",
}

links = []
for country in templates.keys():
    urlTemplate = templates[country]
    i = 0
    while True:
        i += 1
        print(country, i)
        url = urlTemplate.format(i)
        try:
            text = requests.get(url, verify = False).text
        except:
            break
        page = BeautifulSoup(text, 'html.parser')
        f = 0
        for div in page.findAll("div", attrs={"align":"left"}):#, align = "left", class = "ol-lg-1 col-md-1 hidden-xs pd3"):
            if "header" in div.__dict__["attrs"]["class"] or "pd3" in div.__dict__["attrs"]["class"]:
                continue
            f += 1
            image = div.find("img", border="0")
            if image == None:
                continue
            try:
                links.append(image.__dict__["attrs"]["src"])
            except:
                continue
        if f == 0:
            break
out = open("data/links.txt", "w")
out.write("\n".join(links).encode("utf8"))
out.close()
