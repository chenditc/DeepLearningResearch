#-*- coding: utf-8 -*-

import re
import os

import scrapy
from scrapy import log

class SohuSpider(scrapy.Spider):
    name = "sohu_finance"
    allowed_domains = ["m.sohu.com"]
    start_urls = [
            "http://m.sohu.com/cl/49/?page=1&v=2",  # ge gu
            "http://m.sohu.com/cl/42/?page=1&v=2",  # gong si
            "http://m.sohu.com/cl/447/?page=1&v=2", # yin hang
            "http://m.sohu.com/cl/33/?page=1&v=2", # hong guan
            "http://m.sohu.com/cl/135/?page=1&v=2", # dong tai
    ]

    searchUrl = re.compile(r'http://m.sohu.com/cl/(\d+)/\?page=(\d+)&v=2')
    def parse(self, response):
        match = self.searchUrl.match(response.url)
        catagory = match.groups()[0]
        pageNumber = int(match.groups()[1]) + 1
        nextPageUrl = 'http://m.sohu.com/cl/{0}/?page={1}&v=2'.format(catagory, pageNumber)



        urlList = response.xpath('//a/@href').re('(/n/\d+/)')
        urlList = [u'http://m.sohu.com' + url + u'?show_rest_pages=1&v=2' for url in urlList]

        # yield all news pages
        for url in urlList:
            yield scrapy.Request(url, callback=self.parseNews)

        if 'href="?page='+str(pageNumber) in response.body:
            # yield next page
            yield scrapy.Request(nextPageUrl)
        else:
            log.info("Page not exist:" + nextPageUrl)


    def parseNews(self, response):
        outputDir = '/home/ubuntu/news_data/finance_news_data'

        datetime = response.xpath('//p[@class="a3 f12 c2 pb1"]/text()').re('\d\d\d\d-\d\d-\d\d \d\d:\d\d')[0]
        date = datetime.split(' ')[0]
        time = datetime.split(' ')[1]
        articles = response.xpath('//div[@class=""]/text()').extract()

        outputFile = open(os.path.join(outputDir, 'sohu-'+date+'-'+time), 'a')
        for line in articles:
            line.replace('\r\n', ' ')
            outputFile.write(line.encode('utf8'))
        outputFile.write(u'\n')
        outputFile.close()

