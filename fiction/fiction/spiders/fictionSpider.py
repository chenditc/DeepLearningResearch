import scrapy

class FictionSpider(scrapy.Spider):
    name = "i7mu"
    allowed_domains = ["i7mu.cn"]
    start_urls = [
        "http://m.i7wu.cn/read/txt/11164-2.htm"
    ]

    def parse(self, response):
	# extract next url:
	urls = response.xpath('//a/@href').extract()

    def parseFiction(self, response):
        filename = response.url.split('/')[-1]
	filename = filename.split('-')[0]
	textLines = response.xpath('//div[@class="mc"]/text()').extract()
        with open('/home/ubuntu/data/' + filename, 'a') as f:
	    # Do not store the non-article text
	    for line in textLines[1:-3]:
		line.replace('\r\n', ' ')
		f.write(line.encode('utf8'))
		f.write('\n')
