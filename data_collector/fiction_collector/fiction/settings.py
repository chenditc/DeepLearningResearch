# -*- coding: utf-8 -*-

# Scrapy settings for fiction project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'fiction1'

SPIDER_MODULES = ['fiction.spiders']
NEWSPIDER_MODULE = 'fiction.spiders'

CONCURRENT_REQUESTS = 30
CONCURRENT_REQUESTS_PER_DOMAIN = 30

DOWNLOADER_MIDDLEWARES = {
    'scrapy.contrib.downloadermiddleware.httpproxy.HttpProxyMiddleware': 1,
}

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'fiction (+http://www.yourdomain.com)'
