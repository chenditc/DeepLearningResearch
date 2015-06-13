# Scrapy settings for news project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#

BOT_NAME = 'news'
BOT_VERSION = '1.0'

SPIDER_MODULES = ['news.spiders']
NEWSPIDER_MODULE = 'news.spiders'
USER_AGENT = '%s/%s' % (BOT_NAME, BOT_VERSION)

CONCURRENT_REQUESTS = 80
CONCURRENT_REQUESTS_PER_DOMAIN = 80

DOWNLOADER_MIDDLEWARES = {
    'scrapy.contrib.downloadermiddleware.httpproxy.HttpProxyMiddleware': 1,
}
