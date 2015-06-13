import scrapy
import urlparse

class FictionSpider(scrapy.Spider):
    name = "i7wu"
    allowed_domains = ["m.i7wu.cn"]
    start_urls = [
        "http://m.i7wu.cn/read/txt/1-1.htm",
    ]

    def parse(self, response):
	# if there is no such page, quit
	if 'nopage.aspx' in response.url:
	    return

	print response.url

        filename = response.url.split('/')[-1]
	filename = filename.split('.')[0]
	fileId = filename.split('-')[0]
	partitionId = filename.split('-')[1]
	print fileId, partitionId

	textLines = response.xpath('//div[@class="mc"]/text()').extract()
        with open('/home/ubuntu/data/' + filename, 'w') as f:
	    # Do not store the non-article text
	    for line in textLines[1:-3]:
		line.replace('\r\n', ' ')
		f.write(line.encode('utf8'))
		f.write('\n')

	# put more url in
	intPartitionId = int(partitionId)
	for i in range(intPartitionId, intPartitionId + 3):
	    request = scrapy.Request(urlparse.urljoin(response.url, "/read/txt/" + fileId + "-" + str(i) + ".htm"))
	    yield request 

	if partitionId == '1':
	    intFileId = int(fileId)
	    for i in range(intFileId, intFileId + 100):
		request = scrapy.Request(urlparse.urljoin(response.url, "/read/txt/" + str(i) + "-1.htm"))
		yield request 


