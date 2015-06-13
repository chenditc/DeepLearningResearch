import re
import os

import scrapy
from scrapy import log

class HexunSpider(scrapy.Spider):
    name = "hexun"
    allowed_domains = ["m.hexun.com"]
    start_urls = [
        "http://m.hexun.com/index.html"
    ]

    newsPatternUrl = re.compile(r'http://m.hexun.com/.*\d\d\d\d-\d\d-\d\d.*html')
    def parse(self, response):
        urlList = response.xpath('//a/@href').re('.*html')

        outputDir = '/home/ubuntu/news_data/news_data'

        if (self.newsPatternUrl.match(response.url)):
            date = response.xpath('//time/text()').extract()[0].split(' ')[0]
            time = response.xpath('//time/text()').extract()[0].split(' ')[1]
            articles = response.xpath("//div[@id='newsArticle']/p/text()").extract() 

            outputFile = open(os.path.join(outputDir, date+'-'+time), 'a')
            for line in articles:
		line.replace('\r\n', ' ')
		outputFile.write(line.encode('utf8'))

            outputFile.close()

        for url in urlList:
            yield scrapy.Request(url)


