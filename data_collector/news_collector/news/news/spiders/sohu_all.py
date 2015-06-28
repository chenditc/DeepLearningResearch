#-*- coding: utf-8 -*-

import re
import os

import scrapy
from scrapy import log

class SohuSpider(scrapy.Spider):
    name = "sohu_all"
    allowed_domains = ["m.sohu.com"]
    start_urls = open('/home/ubuntu/DeepLearningResearch/data_collector/news_collector/news/sohu_all_url.txt').read().split('\n')

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
        outputDir = '/home/ubuntu/news_data/sohu_news_data'

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

